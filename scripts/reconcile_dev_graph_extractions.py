"""One-off operator tool: reconcile a legacy-shaped ``graph_extractions``
table on a d9-stamped database to the remediated d9 head schema.

Context: dev volumes stamped at ``d9b5f7c1e4a3`` before the d9 remediation
carry ``graph_extractions`` in the old August shape (``input_sha256 DEFAULT
'000...0'``, ``ck_graph_extractions_skip_reason`` instead of the plan-exact
``ck_graph_extractions_lifecycle``, and ``is_identity_owner IN (0, 1)`` token
spacing). The migration chain is immutable history and no revision can re-run
on a stamped database, so ``app.core.migrations.upgrade_database`` refuses at
its pre-upgrade schema check ("Versioned database schema does not match
revision d9b5f7c1e4a3"). This tool applies the equivalent of d9's own
``graph_extractions`` rebuild directly to the live database:

1. refuses unless the database is stamped at exactly ``d9b5f7c1e4a3`` with
   ``graph_extractions`` present;
2. exits 0 up front (no mutation) if the schema already equals the expected
   d9 snapshot, making the tool idempotent;
3. takes a consistent in-volume backup via the SQLite online-backup API
   (captures WAL contents, unlike a plain file copy);
4. pre-validates every existing row against the head lifecycle CHECK
   (extracted verbatim from the d9 module) and refuses without coercion,
   listing offending rows, if any row violates;
5. rebuilds the table through the d9 module's own ``_rebuild_table``
   semantics and DDL constants (staging table, copy ORDER BY id, asserted
   row-count + PK-fingerprint equality, drop legacy default, exact index
   recreation);
6. additionally asserts a full-column row fingerprint before/after so
   losslessness is proven over all columns, not just the PK;
7. re-derives the expected snapshot exactly as the migrate service does and
   verifies equality, so ``python -m app.core.migrations`` is guaranteed to
   pass afterwards.

Run inside the compose topology (backup lands next to the database inside
the named volume; nothing is written on the host):

    docker compose run --rm --no-deps --entrypoint python api \
        scripts/reconcile_dev_graph_extractions.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, text as sa_text

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.migrations import (  # noqa: E402
    _alembic_config,
    _expected_sqlite_snapshot,
    _schema_snapshot,
    _sqlite_database_path,
)
from app.persistence.alembic.versions.d9b5f7c1e4a3_add_safety_reviews import (  # noqa: E402
    _drop_indexes,
    _GE_LIFECYCLE_D9,
    _GRAPH_EXTRACTION_COLUMNS,
    _GRAPH_EXTRACTIONS_D9_DDL,
    _GRAPH_EXTRACTIONS_INDEXES,
    _rebuild_table,
)

_REVISION = "d9b5f7c1e4a3"
_TABLE = "graph_extractions"


def _fail(message: str) -> None:
    sys.stderr.write(f"reconcile_dev_graph_extractions: {message}\n")
    raise SystemExit(1)


def _lifecycle_predicate() -> str:
    """The head lifecycle CHECK body, extracted verbatim from the d9 module."""
    marker = "CHECK "
    stripped = _GE_LIFECYCLE_D9.strip()
    index = stripped.find(marker)
    if index < 0 or not stripped.endswith(")"):
        _fail("cannot extract lifecycle CHECK body from d9 module constant")
    return stripped[index + len(marker):]


def _row_fingerprint(bind, table: str) -> str:
    """SHA-256 over every column of every row, ordered by primary key."""
    rows = bind.execute(
        sa_text(f"SELECT * FROM {table} ORDER BY id")
    ).fetchall()
    payload = json.dumps(
        [[repr(value) for value in row] for row in rows],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _backup(database_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = database_path.with_name(
        f"{database_path.name}.pre-reconcile-{timestamp}"
    )
    source = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    try:
        destination = sqlite3.connect(backup_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    return backup_path


def _current_revision(database_path: Path) -> tuple[str, ...]:
    connection = sqlite3.connect(
        f"file:{database_path.as_posix()}?mode=ro", uri=True
    )
    try:
        return tuple(
            row[0]
            for row in connection.execute(
                "SELECT version_num FROM alembic_version"
            ).fetchall()
        )
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile legacy-shaped graph_extractions to the d9 head "
        "schema (backup-first, lossless, asserted)."
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override the configured RAG_DATABASE_URL for this run.",
    )
    args = parser.parse_args(argv)

    if args.database_url:
        database_url = args.database_url
    else:
        from app.config import get_settings

        database_url = get_settings().database_url

    database_path = _sqlite_database_path(database_url)
    if database_path is None:
        _fail(f"requires a file-backed SQLite URL, got {database_url!r}")
    if not database_path.is_file():
        _fail(f"database not found: {database_path}")

    revisions = _current_revision(database_path)
    if revisions != (_REVISION,):
        _fail(
            f"refusing: alembic_version {revisions!r} != expected "
            f"{(_REVISION,)!r}; this tool only reconciles d9-stamped databases"
        )

    config = _alembic_config(database_url)
    actual = _schema_snapshot(database_url)
    expected = _expected_sqlite_snapshot(config, _REVISION)
    if actual == expected:
        print(f"already reconciled: schema equals expected {_REVISION} snapshot")
        return 0

    differing = sorted(
        name
        for name in set(actual["tables"]) | set(expected["tables"])
        if actual["tables"].get(name) != expected["tables"].get(name)
    )
    if differing != [_TABLE]:
        _fail(
            "refusing: schema differences are not isolated to "
            f"{_TABLE!r}; found {differing!r}"
        )

    predicate = _lifecycle_predicate()
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            violators = connection.exec_driver_sql(
                f"SELECT id, chunk_id, status, error_code, completed_at, "
                f"attempt_count FROM {_TABLE} WHERE NOT ({predicate})"
            ).fetchall()
    finally:
        engine.dispose()
    if violators:
        sys.stderr.write(
            "refusing: existing rows violate the head lifecycle CHECK "
            "(no coercion performed); offending rows "
            "(id, chunk_id, status, error_code, completed_at, attempt_count):\n"
        )
        for row in violators:
            sys.stderr.write(f"  {tuple(row)!r}\n")
        raise SystemExit(1)

    backup_path = _backup(database_path)
    print(f"backup written: {backup_path}")

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            before_count = connection.exec_driver_sql(
                f"SELECT COUNT(*) FROM {_TABLE}"
            ).fetchone()[0]
            before_rows = _row_fingerprint(connection, _TABLE)
        with engine.begin() as connection:
            _drop_indexes(connection, ("uq_graph_extractions_identity_owner",))
            _rebuild_table(
                connection,
                _TABLE,
                _GRAPH_EXTRACTION_COLUMNS,
                _GRAPH_EXTRACTIONS_D9_DDL,
                _GRAPH_EXTRACTIONS_INDEXES,
            )
        with engine.connect() as connection:
            after_count = connection.exec_driver_sql(
                f"SELECT COUNT(*) FROM {_TABLE}"
            ).fetchone()[0]
            after_rows = _row_fingerprint(connection, _TABLE)
    finally:
        engine.dispose()

    if (before_count, before_rows) != (after_count, after_rows):
        _fail(
            "rebuild was not lossless: "
            f"before=(count={before_count}, rows={before_rows}) "
            f"after=(count={after_count}, rows={after_rows}); "
            f"restore from {backup_path}"
        )

    actual = _schema_snapshot(database_url)
    expected = _expected_sqlite_snapshot(config, _REVISION)
    if actual != expected:
        _fail(
            "post-reconciliation schema still differs from expected "
            f"{_REVISION} snapshot; restore from {backup_path}"
        )

    print(
        f"reconciled {_TABLE} to the {_REVISION} head shape: "
        f"rows={after_count} (PK-order full-column fingerprint preserved), "
        f"backup={backup_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
