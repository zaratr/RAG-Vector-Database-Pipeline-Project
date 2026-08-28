"""Hermetic tests for the SQLite backup/restore scripts.

``scripts/backup_database.py`` and ``scripts/restore_database.py`` implement
the operational migration backup/recovery contract:

* backups are taken through the SQLite Online Backup API (``Connection.backup``),
  never a raw file copy, so WAL-mode pages not yet checkpointed into the main
  file are captured — proven here by backing up while a live writer connection
  keeps committed data WAL-resident;
* a backup then restore into a fresh path reproduces the exact rows and
  ``alembic_version`` head, and an explicit ``--allow-overwrite`` restore
  replaces a diverged target's rows with the backup's;
* a manifest/backup pair that disagrees (tampering) is refused;
* existing backup targets are never overwritten;
* restoring over a non-empty or configured-production target requires the
  explicit ``--allow-overwrite`` operator flag;
* every refusal exits ``2``; success exits ``0``.

All fixtures are disposable SQLite files under ``tmp_path``; no service, no
configured store, no network.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import backup_database, restore_database

_BACKUP_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "backup_database.py"
)

_HEAD = "b7f3d5a9c2e1"


def _make_source_db(path: Path, rows: int = 3) -> sqlite3.Connection:
    """Create a WAL-mode database and return its still-open writer.

    Committed pages live in the ``-wal`` file while the writer stays open
    (no close-time checkpoint), which is exactly the state an Online-Backup
    copy must capture.
    """
    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT)")
    connection.execute(
        "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
    )
    connection.execute("INSERT INTO alembic_version (version_num) VALUES (?)", (_HEAD,))
    connection.executemany(
        "INSERT INTO documents (title) VALUES (?)",
        [(f"document {index}",) for index in range(rows)],
    )
    connection.commit()
    return connection


def _read_state(path: Path) -> tuple[list[tuple], str]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT id, title FROM documents ORDER BY id"
        ).fetchall()
        head = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()[0]
    finally:
        connection.close()
    return rows, head


def _url(path: Path) -> str:
    return f"sqlite:///{path}"


def test_backup_round_trip_captures_wal_state_and_restores_identity(tmp_path):
    """Backup while committed data is WAL-resident, then restore into a fresh
    path: rows and head are identical end to end, and the copy provably came
    from the WAL (the source main file alone lacks the data)."""
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        # Premise: the committed rows physically live in the -wal file only.
        assert Path(str(source) + "-wal").stat().st_size > 0
        assert b"document 2" not in source.read_bytes()

        backup = tmp_path / "backup.db"
        code = backup_database.main(
            ["--database-url", _url(source), "--output", str(backup)]
        )
        assert code == 0
        manifest_path = Path(str(backup) + ".manifest.json")
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["backup_head"] == _HEAD
        assert manifest["backup_row_counts"] == {"alembic_version": 1, "documents": 3}
        assert manifest["source_row_counts"] == manifest["backup_row_counts"]
        assert len(manifest["source_sha256"]) == 64
        assert len(manifest["backup_sha256"]) == 64

        # The backup is a standalone readable snapshot of WAL-resident data.
        backup_rows, backup_head = _read_state(backup)
        assert backup_head == _HEAD

        restored = tmp_path / "restored.db"
        code = restore_database.main(
            ["--backup", str(backup), "--database-url", _url(restored)]
        )
        assert code == 0
        expected = [(index + 1, f"document {index}") for index in range(3)]
        assert backup_rows == expected
        source_rows, source_head = _read_state(source)
        restored_rows, restored_head = _read_state(restored)
        assert restored_rows == source_rows == expected
        assert restored_head == source_head == _HEAD
    finally:
        writer.close()


def test_backup_default_output_is_timestamped_next_to_database(tmp_path):
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        code = backup_database.main(["--database-url", _url(source)])
        assert code == 0
        produced = sorted(tmp_path.glob("source-backup-*.db"))
        assert len(produced) == 1
        assert Path(str(produced[0]) + ".manifest.json").is_file()
    finally:
        writer.close()


def test_backup_refusals_exit_2(tmp_path):
    """Refusals: existing output target, directory output, missing source,
    directory source, and non-file-based URLs all exit 2 without writing."""
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        backup = tmp_path / "backup.db"
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(backup)]
            )
            == 0
        )

        # Existing backup targets are never silently overwritten.
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(backup)]
            )
            == 2
        )
        # An existing directory as output is an existing target, not a file to create.
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(tmp_path)]
            )
            == 2
        )
        # Missing source database.
        assert (
            backup_database.main(
                [
                    "--database-url",
                    _url(tmp_path / "absent.db"),
                    "--output",
                    str(tmp_path / "out.db"),
                ]
            )
            == 2
        )
        # A directory is not a database file.
        assert (
            backup_database.main(
                ["--database-url", _url(tmp_path), "--output", str(tmp_path / "out2.db")]
            )
            == 2
        )
        # In-memory databases cannot be backed up to a file this way.
        assert (
            backup_database.main(
                [
                    "--database-url",
                    "sqlite:///:memory:",
                    "--output",
                    str(tmp_path / "out3.db"),
                ]
            )
            == 2
        )
        assert not (tmp_path / "out.db").exists()
        assert not (tmp_path / "out2.db").exists()
    finally:
        writer.close()


def test_backup_detects_torn_copy_by_row_counts(monkeypatch, tmp_path):
    """A lying Online-Backup copy whose tables/rows differ from the source
    must be detected and the partial backup removed — never reported as success."""

    def lying_online_backup(source, target):
        target.execute("CREATE TABLE torn (id INTEGER PRIMARY KEY)")
        target.execute("INSERT INTO torn (id) VALUES (1)")
        target.commit()

    monkeypatch.setattr(backup_database, "_online_backup", lying_online_backup)
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        backup = tmp_path / "backup.db"
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(backup)]
            )
            == 2
        )
        assert not backup.exists()
        assert not Path(str(backup) + ".manifest.json").exists()
    finally:
        writer.close()


def test_restore_refuses_tampered_backup_or_manifest(tmp_path):
    """Every tamper form is refused with exit 2 and never creates the target:
    edited checksum, appended backup bytes, corrupt manifest JSON, missing
    manifest, and a manifest missing required fields."""
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        backup = tmp_path / "backup.db"
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(backup)]
            )
            == 0
        )
    finally:
        writer.close()
    manifest_path = Path(str(backup) + ".manifest.json")
    original_manifest = manifest_path.read_text(encoding="utf-8")

    target = tmp_path / "target.db"

    def _restore():
        return restore_database.main(
            ["--backup", str(backup), "--database-url", _url(target)]
        )

    # Edited checksum in an otherwise valid manifest.
    manifest = json.loads(original_manifest)
    manifest["backup_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _restore() == 2
    # Backup file itself tampered (manifest restored to the truthful copy).
    manifest_path.write_text(original_manifest, encoding="utf-8")
    with backup.open("ab") as handle:
        handle.write(b"tampered")
    assert _restore() == 2
    # Corrupt manifest JSON.
    manifest_path.write_text("not json at all", encoding="utf-8")
    assert _restore() == 2
    # Manifest missing entirely.
    manifest_path.unlink()
    assert _restore() == 2
    # Manifest missing a required field.
    manifest = json.loads(original_manifest)
    del manifest["backup_row_counts"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _restore() == 2

    assert not target.exists()


def test_restore_refuses_non_empty_target_without_operator_flag(tmp_path):
    """A non-empty target is never silently replaced: without the flag the
    restore exits 2 and the target keeps its own rows; with the explicit
    flag the backup's rows replace them."""
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        backup = tmp_path / "backup.db"
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(backup)]
            )
            == 0
        )
    finally:
        writer.close()

    target = tmp_path / "target.db"
    other = sqlite3.connect(str(target))
    other.execute("CREATE TABLE other (id INTEGER PRIMARY KEY, note TEXT)")
    other.execute("INSERT INTO other (note) VALUES ('precious local data')")
    other.commit()
    other.close()

    assert (
        restore_database.main(
            ["--backup", str(backup), "--database-url", _url(target)]
        )
        == 2
    )
    check = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
    assert check.execute("SELECT note FROM other").fetchall() == [
        ("precious local data",)
    ]
    check.close()

    assert (
        restore_database.main(
            [
                "--backup",
                str(backup),
                "--database-url",
                _url(target),
                "--allow-overwrite",
            ]
        )
        == 0
    )
    rows, head = _read_state(target)
    assert rows == [(1, "document 0"), (2, "document 1"), (3, "document 2")]
    assert head == _HEAD


def test_restore_refuses_configured_production_target_without_operator_flag(
    monkeypatch, tmp_path
):
    """Mirrors the sibling scripts' production-path refusal: a target that
    equals the configured database is refused without the explicit flag even
    when it is an empty file."""
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        backup = tmp_path / "backup.db"
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(backup)]
            )
            == 0
        )
    finally:
        writer.close()

    production = tmp_path / "production.db"
    production.write_bytes(b"")
    monkeypatch.setattr(
        restore_database,
        "get_settings",
        lambda: SimpleNamespace(database_url=_url(production)),
    )

    assert (
        restore_database.main(
            ["--backup", str(backup), "--database-url", _url(production)]
        )
        == 2
    )
    assert production.read_bytes() == b""

    assert (
        restore_database.main(
            [
                "--backup",
                str(backup),
                "--database-url",
                _url(production),
                "--allow-overwrite",
            ]
        )
        == 0
    )
    rows, head = _read_state(production)
    assert rows == [(1, "document 0"), (2, "document 1"), (3, "document 2")]
    assert head == _HEAD


def test_restore_detects_post_restore_mismatch(monkeypatch, tmp_path):
    """A lying copy during restore must fail the post-restore verification
    (head/row counts vs the manifest) and exit 2."""

    def lying_online_backup(source, target):
        target.execute("CREATE TABLE wrong (id INTEGER PRIMARY KEY)")
        target.execute("INSERT INTO wrong (id) VALUES (1)")
        target.commit()

    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        # Take the backup with the REAL Online Backup API first.
        backup = tmp_path / "backup.db"
        assert (
            backup_database.main(
                ["--database-url", _url(source), "--output", str(backup)]
            )
            == 0
        )
    finally:
        writer.close()

    # Now make every restore-side Online Backup copy lie.
    monkeypatch.setattr(restore_database, "_online_backup", lying_online_backup)
    target = tmp_path / "target.db"
    assert (
        restore_database.main(
            ["--backup", str(backup), "--database-url", _url(target)]
        )
        == 2
    )


def test_backup_script_runs_standalone_in_subprocess(tmp_path):
    """Exit-code and bootstrap proof as a real CLI invocation: the script
    imports cleanly outside pytest and prints the manifest JSON on stdout."""
    source = tmp_path / "source.db"
    writer = _make_source_db(source)
    try:
        backup = tmp_path / "backup.db"
        result = subprocess.run(
            [
                sys.executable,
                str(_BACKUP_SCRIPT),
                "--database-url",
                _url(source),
                "--output",
                str(backup),
            ],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            env={**os.environ},
        )
        assert result.returncode == 0, result.stderr
        manifest = json.loads(result.stdout)
        assert manifest["backup_row_counts"] == {"alembic_version": 1, "documents": 3}
        assert Path(str(backup) + ".manifest.json").is_file()
    finally:
        writer.close()
