"""Alembic migration environment."""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, event, pool

# Ensure the project root is on sys.path so app.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import get_settings  # noqa: E402
from app.core.db import Base  # noqa: E402
from app.persistence import models  # noqa: E402, F401  (register tables on Base.metadata)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Programmatic callers mark their URL as explicit. Direct CLI commands use the
# same environment-driven URL as the application instead of the ini fallback.
if not config.attributes.get("database_url_explicit", False):
    database_url = get_settings().database_url
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    is_sqlite = connectable.dialect.name == "sqlite"
    if is_sqlite:
        # SQLAlchemy pysqlite transactional-DDL recipe: the driver's legacy
        # isolation opens a transaction only before DML, which leaves every
        # CREATE/ALTER/DROP running in autocommit. Switch the DBAPI to
        # autocommit and issue an explicit BEGIN when SQLAlchemy starts a
        # transaction, so alembic's migration transaction covers DDL and DML
        # alike and a mid-migration failure rolls back atomically.
        @event.listens_for(connectable, "connect")
        def _sqlite_transactional_connect(dbapi_connection, connection_record):
            dbapi_connection.isolation_level = None

        @event.listens_for(connectable, "begin")
        def _sqlite_transactional_begin(connection):
            connection.exec_driver_sql("BEGIN")

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            transactional_ddl=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
