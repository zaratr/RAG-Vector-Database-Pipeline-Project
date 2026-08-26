"""Restore a SQLite database from a verified backup.

The restore is the inverse of ``scripts/backup_database.py``: it verifies the
backup file's SHA-256 against the manifest sidecar created at backup time
(any tampering is refused), then copies the backup into the target database
using the same SQLite Online Backup API (never a raw file copy). After the
copy it re-opens the target and verifies the ``alembic_version`` head and
per-table row counts match the manifest; any mismatch exits ``2``.

The target is never silently destroyed: restoring over a non-empty database,
or over the configured production database, requires the explicit
``--allow-overwrite`` operator flag. When the flag is present, stale
``-wal``/``-shm`` sidecars of the overwritten target are removed first so
the restored file stands alone.

Exit codes:

* ``0`` — success; stdout is the post-restore verification JSON.
* ``2`` — tampered backup/manifest, missing inputs, refused overwrite, or
  post-restore verification failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

from sqlalchemy import make_url

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings  # noqa: E402

MANIFEST_SUFFIX = ".manifest.json"

_REQUIRED_MANIFEST_FIELDS = (
    "backup_path",
    "backup_sha256",
    "backup_head",
    "backup_row_counts",
)


class RestoreError(RuntimeError):
    """Raised for any condition that must exit 2."""


def _database_path(database_url: str) -> Path:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise RestoreError(
            "--database-url must be a file-based SQLite database URL"
        )
    return Path(url.database).resolve()


def _configured_production_path() -> Path | None:
    """Resolved path of the configured database, when it is SQLite-file based."""
    database_url = get_settings().database_url
    if not database_url or not database_url.startswith("sqlite:"):
        return None
    location = database_url.split("sqlite:///", 1)[1].split("?", 1)[0]
    if not location or location == ":memory:":
        return None
    try:
        return Path(location).resolve()
    except OSError:
        return None


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


def _is_production_path(target_path: Path) -> bool:
    """True when the target equals/symlinks the configured production database."""
    configured = _configured_production_path()
    if configured is None:
        return False
    try:
        if target_path.resolve() == configured:
            return True
        if target_path.is_symlink() and Path(os.readlink(target_path)).resolve() == configured:
            return True
    except OSError:
        return False
    return False


def _load_manifest(backup_path: Path) -> dict:
    manifest_path = Path(str(backup_path) + MANIFEST_SUFFIX)
    if not manifest_path.is_file():
        raise RestoreError(f"backup manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RestoreError(f"unreadable backup manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RestoreError("backup manifest must be a JSON object")
    missing = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise RestoreError(f"backup manifest missing fields: {sorted(missing)}")
    return manifest


def restore_database(
    *, backup_path: Path, database_url: str, allow_overwrite: bool = False
) -> dict:
    """Restore ``backup_path`` into the database at ``database_url``.

    Returns the post-restore verification summary. Raises
    :class:`RestoreError` on tampering, refusal, or verification failure.
    """
    backup_path = backup_path.resolve()
    if not backup_path.is_file():
        raise RestoreError(f"backup file not found: {backup_path}")
    target_path = _database_path(database_url)

    manifest = _load_manifest(backup_path)

    actual_sha256 = _sha256(backup_path)
    if actual_sha256 != manifest["backup_sha256"]:
        raise RestoreError(
            "backup checksum mismatch: the backup file does not match its "
            f"manifest (expected {manifest['backup_sha256']}, found {actual_sha256})"
        )

    non_empty_target = target_path.exists() and target_path.stat().st_size > 0
    production_target = _is_production_path(target_path)
    if (non_empty_target or production_target) and not allow_overwrite:
        reasons = []
        if non_empty_target:
            reasons.append("non-empty")
        if production_target:
            reasons.append("the configured production database")
        raise RestoreError(
            "refusing to restore over " + " and ".join(reasons) + f" target: {target_path} "
            "(pass --allow-overwrite to force)"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if allow_overwrite:
        # Stale sidecars of the destroyed target would otherwise mix pages
        # into the restored database.
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(target_path) + suffix)
            try:
                sidecar.unlink()
            except OSError:
                pass

    try:
        source = sqlite3.connect(str(backup_path))
        try:
            target = sqlite3.connect(str(target_path))
            try:
                _online_backup(source, target)
            finally:
                target.close()
        finally:
            source.close()
    except sqlite3.Error as exc:
        raise RestoreError(f"SQLite restore failed: {exc}") from exc

    try:
        verification = sqlite3.connect(
            f"file:{target_path.as_posix()}?mode=ro", uri=True
        )
        try:
            head = _alembic_head(verification)
            row_counts = _row_counts(verification)
        finally:
            verification.close()
    except sqlite3.Error as exc:
        raise RestoreError(f"restored database unreadable: {exc}") from exc

    if head != manifest["backup_head"]:
        raise RestoreError(
            f"post-restore head mismatch: expected {manifest['backup_head']!r}, "
            f"found {head!r}"
        )
    if row_counts != manifest["backup_row_counts"]:
        raise RestoreError(
            "post-restore row-count mismatch: expected "
            f"{manifest['backup_row_counts']}, found {row_counts}"
        )
    return {
        "restored_path": str(target_path),
        "backup_path": str(backup_path),
        "head": head,
        "row_counts": row_counts,
        "backup_sha256": manifest["backup_sha256"],
    }


def _online_backup(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    """SQLite Online Backup API in reverse: whole-file consistent copy."""
    source.backup(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backup",
        required=True,
        help="backup file created by scripts/backup_database.py",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="file-based SQLite database URL to restore into "
        "(default: the configured database)",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="explicit operator override permitting the restore to replace a "
        "non-empty or configured-production target database",
    )
    args = parser.parse_args(argv)

    try:
        database_url = args.database_url or get_settings().database_url
        summary = restore_database(
            backup_path=Path(args.backup),
            database_url=database_url,
            allow_overwrite=args.allow_overwrite,
        )
    except RestoreError as error:
        print(f"restore_database: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
