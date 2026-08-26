"""Back up a SQLite database through the SQLite Online Backup API.

The backup never copies the raw database file: it opens a live ``sqlite3``
connection to the source and uses ``Connection.backup()`` so the copy is a
consistent snapshot that includes WAL-mode pages not yet checkpointed into
the main file. A non-sensitive JSON manifest sidecar is written next to the
backup recording the source/backup paths, UTC timestamp, source and backup
SHA-256 fingerprints, the ``alembic_version`` head, and the per-table row
counts of both files (no text content, credentials, or query results).

Refusals (all exit ``2``):

* the source is missing or not a regular file;
* the output target already exists (a backup is never silently overwritten);
* the backup's post-copy head or per-table row counts disagree with the
  source (a torn or incomplete copy);
* any I/O or SQLite failure during the copy.

Exit codes:

* ``0`` — success; stdout is the manifest JSON.
* ``2`` — any refusal, verification, or infrastructure failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import make_url

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings  # noqa: E402

MANIFEST_SUFFIX = ".manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class BackupError(RuntimeError):
    """Raised for any condition that must exit 2."""


def _database_path(database_url: str) -> Path:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise BackupError(
            "--database-url must be a file-based SQLite database URL"
        )
    return Path(url.database).resolve()


def _default_output_path(database_path: Path) -> Path:
    """Timestamped backup filename next to the source database."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return database_path.parent / f"{database_path.stem}-backup-{timestamp}.db"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table_names(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def _row_counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in _table_names(connection)
    }


def _alembic_head(connection: sqlite3.Connection) -> str | None:
    try:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def _database_stats(connection: sqlite3.Connection) -> dict:
    return {"head": _alembic_head(connection), "row_counts": _row_counts(connection)}


def backup_database(*, database_url: str, output: Path) -> dict:
    """Copy the live database to ``output`` via the Online Backup API.

    Returns the manifest written at ``output + MANIFEST_SUFFIX``. Raises
    :class:`BackupError` on any refusal or verification failure, in which
    case no partial backup or manifest is left behind.
    """
    source_path = _database_path(database_url)
    if not source_path.is_file():
        raise BackupError(f"source database is not a file: {source_path}")

    output = output.resolve()
    if output.exists():
        raise BackupError(f"refusing to overwrite existing backup target: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(str(output) + MANIFEST_SUFFIX)
    try:
        source = sqlite3.connect(str(source_path))
        try:
            source_stats = _database_stats(source)
            target = sqlite3.connect(str(output))
            try:
                _online_backup(source, target)
            finally:
                target.close()
        finally:
            source.close()
    except sqlite3.Error as exc:
        _remove_partial(output)
        raise BackupError(f"SQLite backup failed: {exc}") from exc

    try:
        verification = sqlite3.connect(f"file:{output.as_posix()}?mode=ro", uri=True)
        try:
            backup_stats = _database_stats(verification)
        finally:
            verification.close()
    except sqlite3.Error as exc:
        _remove_partial(output)
        raise BackupError(f"backup verification unreadable: {exc}") from exc

    if backup_stats != source_stats:
        _remove_partial(output)
        raise BackupError(
            "backup verification failed: head or per-table row counts "
            f"disagree with the source (source={source_stats}, backup={backup_stats})"
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "backup_path": str(output),
        "source_sha256": _sha256(source_path),
        "backup_sha256": _sha256(output),
        "source_head": source_stats["head"],
        "backup_head": backup_stats["head"],
        "source_row_counts": source_stats["row_counts"],
        "backup_row_counts": backup_stats["row_counts"],
    }
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        _remove_partial(output)
        raise BackupError(f"cannot write manifest {manifest_path}: {exc}") from exc
    return manifest


def _online_backup(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """SQLite Online Backup API: consistent snapshot including WAL pages."""
    source.backup(target)


def _remove_partial(output: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(output) + suffix)
        try:
            path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default=None,
        help="file-based SQLite database URL (default: the configured database)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="backup file path (default: a timestamped file next to the database)",
    )
    args = parser.parse_args(argv)

    try:
        database_url = args.database_url or get_settings().database_url
        source_path = _database_path(database_url)
        output = (
            Path(args.output).resolve()
            if args.output
            else _default_output_path(source_path)
        )
        manifest = backup_database(database_url=database_url, output=output)
    except BackupError as error:
        print(f"backup_database: {error}", file=sys.stderr)
        return 2

    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
