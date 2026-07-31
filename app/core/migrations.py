"""Database schema migration coordination."""
from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Column, ForeignKey, Index, Integer, MetaData, String, Table, Text, create_engine, inspect
from sqlalchemy.engine import make_url

from app.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
MIGRATION_SCRIPTS = PROJECT_ROOT / "app" / "persistence" / "alembic"
BASELINE_REVISION = "dee48bc24a7f"


def _build_baseline_metadata() -> MetaData:
    """Return the immutable schema represented by the baseline revision."""
    metadata = MetaData()
    documents = Table(
        "documents",
        metadata,
        Column("id", Integer, primary_key=True, nullable=False),
        Column("title", String, nullable=False),
        Column("source", String, nullable=True),
        Column("tags", String, nullable=True),
    )
    Index("ix_documents_id", documents.c.id)
    chunks = Table(
        "chunks",
        metadata,
        Column("id", Integer, primary_key=True, nullable=False),
        Column(
            "document_id",
            Integer,
            ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("index", Integer, nullable=False),
        Column("text", Text, nullable=False),
        Column("start_offset", Integer, nullable=False),
        Column("end_offset", Integer, nullable=False),
    )
    Index("ix_chunks_id", chunks.c.id)
    return metadata


BASELINE_METADATA = _build_baseline_metadata()


def _sqlite_database_path(database_url: str) -> Path | None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return None
    return Path(url.database).resolve()


def _ensure_database_parent(database_url: str) -> None:
    database_path = _sqlite_database_path(database_url)
    if database_path is not None:
        database_path.parent.mkdir(parents=True, exist_ok=True)


def import_legacy_sqlite(source_path: str | Path, target_database_url: str) -> None:
    """Copy a legacy SQLite database into an empty durable target safely."""
    source = Path(source_path).resolve()
    target = _sqlite_database_path(target_database_url)
    if target is None:
        raise ValueError("--legacy-sqlite-path requires a file-based SQLite target URL")
    if not source.is_file():
        raise FileNotFoundError(f"Legacy SQLite database not found: {source}")
    if source == target:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with sqlite3.connect(target) as target_connection:
            objects = target_connection.execute(
                "SELECT type, name FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall()
        if objects:
            raise RuntimeError(
                f"Refusing to overwrite non-empty target database: {target}; "
                f"existing schema objects: {objects!r}"
            )

    source_uri = f"file:{source.as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def _alembic_config(database_url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(MIGRATION_SCRIPTS))
    # ConfigParser treats percent signs as interpolation markers.
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    config.attributes["database_url_explicit"] = True
    return config


def _normalize_sql(value) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().split())


def _sqlite_schema_objects(database_url: str) -> tuple[tuple[str, str, str, str | None], ...]:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return tuple(
                (row.type, row.name, row.tbl_name, _normalize_sql(row.sql))
                for row in connection.exec_driver_sql(
                    "SELECT type, name, tbl_name, sql FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                )
            )
    finally:
        engine.dispose()


def _schema_snapshot(database_url: str) -> dict:
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        tables = sorted(
            table for table in inspector.get_table_names() if table != "alembic_version"
        )
        snapshot = {
            "tables": {},
            "views": tuple(
                sorted(
                    (
                        view_name,
                        _normalize_sql(inspector.get_view_definition(view_name)),
                    )
                    for view_name in inspector.get_view_names()
                )
            ),
        }
        for table in tables:
            columns = tuple(
                (
                    column["name"],
                    str(column["type"]).upper(),
                    bool(column["nullable"]),
                    int(column.get("primary_key", 0)),
                    _normalize_sql(column.get("default")),
                )
                for column in inspector.get_columns(table)
            )
            primary_key = tuple(
                inspector.get_pk_constraint(table).get("constrained_columns") or ()
            )
            foreign_keys = tuple(
                sorted(
                    (
                        tuple(foreign_key.get("constrained_columns") or ()),
                        foreign_key.get("referred_schema"),
                        foreign_key.get("referred_table"),
                        tuple(foreign_key.get("referred_columns") or ()),
                        (foreign_key.get("options") or {}).get("ondelete", "").upper(),
                        (foreign_key.get("options") or {}).get("onupdate", "").upper(),
                    )
                    for foreign_key in inspector.get_foreign_keys(table)
                )
            )
            indexes = tuple(
                sorted(
                    (
                        index.get("name"),
                        tuple(index.get("column_names") or ()),
                        bool(index.get("unique", False)),
                    )
                    for index in inspector.get_indexes(table)
                )
            )
            unique_constraints = tuple(
                sorted(
                    (
                        constraint.get("name") or "",
                        tuple(constraint.get("column_names") or ()),
                    )
                    for constraint in inspector.get_unique_constraints(table)
                )
            )
            check_constraints = tuple(
                sorted(
                    (
                        constraint.get("name") or "",
                        _normalize_sql(constraint.get("sqltext")),
                    )
                    for constraint in inspector.get_check_constraints(table)
                )
            )
            snapshot["tables"][table] = {
                "columns": columns,
                "primary_key": primary_key,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "unique_constraints": unique_constraints,
                "check_constraints": check_constraints,
            }

        if make_url(database_url).get_backend_name() == "sqlite":
            with engine.connect() as connection:
                snapshot["triggers"] = tuple(
                    (row.name, row.tbl_name, _normalize_sql(row.sql))
                    for row in connection.exec_driver_sql(
                        "SELECT name, tbl_name, sql FROM sqlite_master "
                        "WHERE type = 'trigger' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                    )
                )
        return snapshot
    finally:
        engine.dispose()


def _expected_sqlite_snapshot(config: Config, revision: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="rag-migration-schema-") as temp_dir:
        expected_url = f"sqlite:///{Path(temp_dir) / 'expected.db'}"
        expected_config = Config(str(ALEMBIC_INI))
        expected_config.set_main_option(
            "script_location", config.get_main_option("script_location")
        )
        expected_config.set_main_option("sqlalchemy.url", expected_url)
        expected_config.attributes["database_url_explicit"] = True
        command.upgrade(expected_config, revision)
        return _schema_snapshot(expected_url)


def _schema_differences(database_url: str, config: Config, revision: str) -> list:
    if make_url(database_url).get_backend_name() == "sqlite":
        actual = _schema_snapshot(database_url)
        expected = _expected_sqlite_snapshot(config, revision)
        return [] if actual == expected else [{"actual": actual, "expected": expected}]

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            migration_context = MigrationContext.configure(connection)
            return compare_metadata(migration_context, BASELINE_METADATA)
    finally:
        engine.dispose()


def _current_revisions(database_url: str) -> tuple[str, ...]:
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return tuple(MigrationContext.configure(connection).get_current_heads())
    finally:
        engine.dispose()


def upgrade_database(database_url: str) -> None:
    """Upgrade a fresh/versioned DB or safely adopt an exact legacy baseline.

    Databases created by the pre-Alembic application have the complete model
    schema but no ``alembic_version`` table. Such a database is stamped only
    after Alembic confirms that its schema exactly matches the frozen baseline.
    Partial or drifted unversioned schemas are rejected without stamping.
    """
    _ensure_database_parent(database_url)
    config = _alembic_config(database_url)
    engine = create_engine(database_url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    application_tables = tables - {"alembic_version"}
    if make_url(database_url).get_backend_name() == "sqlite":
        schema_is_nonempty = bool(_sqlite_schema_objects(database_url))
    else:
        schema_is_nonempty = bool(tables)

    if schema_is_nonempty and "alembic_version" not in tables:
        differences = _schema_differences(database_url, config, BASELINE_REVISION)
        if differences:
            raise RuntimeError(
                "Unversioned database schema does not match the Alembic baseline; "
                f"refusing to stamp. Differences: {differences!r}"
            )
        command.stamp(config, BASELINE_REVISION)
    elif "alembic_version" in tables:
        revisions = _current_revisions(database_url)
        if not revisions and not application_tables:
            differences = _schema_differences(database_url, config, "base")
            if differences:
                raise RuntimeError(
                    "Database at Alembic base contains unexpected schema objects; "
                    f"refusing to upgrade. Differences: {differences!r}"
                )
        elif len(revisions) != 1:
            raise RuntimeError(
                f"Expected exactly one current Alembic revision, found {revisions!r}"
            )
        else:
            ScriptDirectory.from_config(config).get_revision(revisions[0])
            differences = _schema_differences(database_url, config, revisions[0])
            if differences:
                raise RuntimeError(
                    f"Versioned database schema does not match revision {revisions[0]}; "
                    f"refusing to upgrade. Differences: {differences!r}"
                )

    command.upgrade(config, "head")
    heads = tuple(ScriptDirectory.from_config(config).get_heads())
    if len(heads) != 1:
        raise RuntimeError(f"Expected exactly one Alembic head, found {heads!r}")
    differences = _schema_differences(database_url, config, heads[0])
    if differences:
        raise RuntimeError(
            f"Database schema does not match Alembic head {heads[0]} after upgrade. "
            f"Differences: {differences!r}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the adoption-aware production migration command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-sqlite-path",
        help="Copy an exact pre-Alembic SQLite database into an empty configured target before upgrading",
    )
    args = parser.parse_args(argv)
    database_url = get_settings().database_url
    if args.legacy_sqlite_path:
        import_legacy_sqlite(args.legacy_sqlite_path, database_url)
    upgrade_database(database_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())