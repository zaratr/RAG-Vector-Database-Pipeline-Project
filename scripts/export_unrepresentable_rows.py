"""Export rows that block a downgrade because its predecessor cannot represent them.

Implements the 10A.3 half of the plan's unrepresentable-data recovery contract
(B-14): when the ``b7f3d5a9c2e1`` downgrade refuses with
``downgrade_skipped_rows_present``, this script exports the blocking
``skipped`` rows to a deterministic JSON file so the operator can review and
explicitly delete them; the downgrade then proceeds. Task 10C.4 later extends
the revision registry with ``d9b5f7c1e4a3`` for its safety-blocked rows.

The script is strictly read-only: export preserves every database row, never
deletes anything, asserts the exported row count against the database, and
requires explicit revision selection. ``--dry-run`` prints the deterministic
payload without writing the output file. Exits 2 on any error.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from sqlalchemy import make_url

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings  # noqa: E402

# Per-revision export definitions. 10C.4 adds "d9b5f7c1e4a3" later.
REVISION_EXPORTS: dict[str, dict[str, str]] = {
    "b7f3d5a9c2e1": {
        "table": "graph_extractions",
        "predicate": "status = 'skipped'",
        "reason": "skipped",
    },
}


class ExportError(RuntimeError):
    """Raised for any condition that must exit 2."""


def _database_path(database_url: str) -> Path:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise ExportError(
            "--revision export requires a file-based SQLite database URL"
        )
    return Path(url.database).resolve()


def export_unrepresentable_rows(
    *,
    database_url: str,
    revision: str,
    output_path: Path | None,
    dry_run: bool = False,
) -> dict:
    """Export the rows ``revision``'s downgrade cannot represent.

    Returns the deterministic payload dict. Writes ``output_path`` unless
    ``dry_run``. Never mutates the database.
    """
    export = REVISION_EXPORTS.get(revision)
    if export is None:
        raise ExportError(
            f"unsupported revision {revision!r}; supported revisions: "
            f"{sorted(REVISION_EXPORTS)}"
        )
    if not dry_run and output_path is None:
        raise ExportError("--output PATH is required unless --dry-run is set")

    database_path = _database_path(database_url)
    if not database_path.is_file():
        raise ExportError(f"database not found: {database_path}")

    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    try:
        current_revision = connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()
        if current_revision is None or current_revision[0] != revision:
            found = current_revision[0] if current_revision else None
            raise ExportError(
                f"database is at revision {found!r}, not the selected "
                f"revision {revision!r}; refusing to export"
            )

        cursor = connection.execute(
            f"SELECT * FROM {export['table']} WHERE {export['predicate']} "
            "ORDER BY id"
        )
        column_names = [description[0] for description in cursor.description]
        rows = [dict(zip(column_names, row)) for row in cursor.fetchall()]

        # Row-count assertion: the exported rows must be exactly the rows the
        # predicate selects at export time.
        expected_count = connection.execute(
            f"SELECT COUNT(*) FROM {export['table']} WHERE {export['predicate']}"
        ).fetchone()[0]
        if len(rows) != expected_count:
            raise ExportError(
                f"row-count assertion failed: exported {len(rows)} rows but "
                f"database reports {expected_count}"
            )
    finally:
        connection.close()

    payload = {
        "revision": revision,
        "table": export["table"],
        "reason": export["reason"],
        "row_count": len(rows),
        "rows": rows,
    }
    document = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if dry_run:
        sys.stdout.write(document)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8", newline="\n")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        required=True,
        help="revision whose downgrade refuses (currently b7f3d5a9c2e1)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="deterministic JSON output path (required unless --dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the payload without writing the output file",
    )
    args = parser.parse_args(argv)

    try:
        payload = export_unrepresentable_rows(
            database_url=get_settings().database_url,
            revision=args.revision,
            output_path=Path(args.output).resolve() if args.output else None,
            dry_run=args.dry_run,
        )
    except ExportError as error:
        print(f"export_unrepresentable_rows: {error}", file=sys.stderr)
        return 2

    print(
        f"exported {payload['row_count']} unrepresentable row(s) for "
        f"revision {payload['revision']} ({payload['table']}: {payload['reason']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
