"""Production-path tests for Alembic schema migrations."""
from __future__ import annotations

import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.core.db import create_database_engine
from app.core import migrations
from app.core.migrations import upgrade_database
from app.persistence import models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
BASELINE_REVISION = "dee48bc24a7f"
REVISION = "d9b5f7c1e4a3"
HEAD_TABLES = {
    "alembic_version",
    "chunks",
    "documents",
    "entity_mentions",
    "graph_edge_evidence",
    "graph_edges",
    "graph_entities",
    "graph_extractions",
    "retrieval_audits",
    "retrieval_candidate_decisions",
    "ingestion_rate_buckets",
    "safety_review_runs",
    "safety_findings",
}


def _db_url(path: Path) -> str:
    return f"sqlite:///{path}"


def _alembic_config(db_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "app" / "persistence" / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.attributes["database_url_explicit"] = True
    return cfg


def _run_migration_entrypoint(
    db_url: str, *extra_args: str
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["RAG_DATABASE_URL"] = db_url
    return subprocess.run(
        [sys.executable, "-m", "app.core.migrations", *extra_args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _revision(engine) -> str | None:
    with engine.connect() as conn:
        return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def _future_scripts_with_view_and_trigger(tmp_path: Path) -> tuple[Path, str]:
    scripts = tmp_path / "alembic-with-definitions"
    shutil.copytree(
        PROJECT_ROOT / "app" / "persistence" / "alembic",
        scripts,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    revision = "definition001"
    (scripts / "versions" / f"{revision}_view_and_trigger.py").write_text(
        f'''"""test-only view and trigger migration"""
from alembic import op
import sqlalchemy as sa

revision = "{revision}"
down_revision = "{REVISION}"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "document_audit",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("recorded_title", sa.String(), nullable=False),
    )
    op.execute("CREATE VIEW document_titles AS SELECT id, title FROM documents")
    op.execute(
        "CREATE TRIGGER document_title_audit AFTER INSERT ON documents "
        "BEGIN INSERT INTO document_audit (document_id, recorded_title) "
        "VALUES (NEW.id, NEW.title); END"
    )

def downgrade():
    op.execute("DROP TRIGGER document_title_audit")
    op.execute("DROP VIEW document_titles")
    op.drop_table("document_audit")
''',
        encoding="utf-8",
    )
    return scripts, revision


def test_baseline_creates_exact_schema_on_empty_db(tmp_path):
    db_url = _db_url(tmp_path / "fresh.db")
    upgrade_database(db_url)

    engine = create_engine(db_url)
    insp = inspect(engine)
    assert set(insp.get_table_names()) == HEAD_TABLES

    doc_cols = {column["name"]: column for column in insp.get_columns("documents")}
    expected_docs = {
        "id": ("INTEGER", False, 1),
        "title": ("VARCHAR", False, 0),
        "source": ("VARCHAR", True, 0),
        "tags": ("VARCHAR", True, 0),
        "ingestion_status": ("VARCHAR(20)", False, 0),
        "failure_code": ("VARCHAR(100)", True, 0),
        "trust_tier": ("VARCHAR(20)", False, 0),
        "trust_score": ("FLOAT", False, 0),
        "trust_policy_version": ("VARCHAR(50)", False, 0),
        "ingestion_origin": ("VARCHAR(50)", False, 0),
    }
    assert {
        name: (str(column["type"]), column["nullable"], column["primary_key"])
        for name, column in doc_cols.items()
    } == expected_docs
    assert insp.get_pk_constraint("documents")["constrained_columns"] == ["id"]

    chunk_cols = {column["name"]: column for column in insp.get_columns("chunks")}
    expected_chunks = {
        "id": ("INTEGER", False, 1),
        "document_id": ("INTEGER", False, 0),
        "index": ("INTEGER", False, 0),
        "text": ("TEXT", False, 0),
        "start_offset": ("INTEGER", False, 0),
        "end_offset": ("INTEGER", False, 0),
        "vector_id": ("VARCHAR(255)", True, 0),
        "media_type": ("VARCHAR(100)", False, 0),
    }
    assert {
        name: (str(column["type"]), column["nullable"], column["primary_key"])
        for name, column in chunk_cols.items()
    } == expected_chunks
    assert insp.get_pk_constraint("chunks")["constrained_columns"] == ["id"]

    foreign_keys = insp.get_foreign_keys("chunks")
    assert len(foreign_keys) == 1
    assert foreign_keys[0]["constrained_columns"] == ["document_id"]
    assert foreign_keys[0]["referred_table"] == "documents"
    assert foreign_keys[0]["referred_columns"] == ["id"]
    assert foreign_keys[0]["options"] == {"ondelete": "CASCADE"}

    assert {
        index["name"]: (index["column_names"], index["unique"])
        for index in insp.get_indexes("documents")
    } == {
        "ix_documents_id": (["id"], 0),
        "ix_documents_ingestion_status": (["ingestion_status"], 0),
        "ix_documents_trust_tier": (["trust_tier"], 0),
        "ix_documents_trust_policy_version": (["trust_policy_version"], 0),
    }
    assert {
        index["name"]: (index["column_names"], index["unique"])
        for index in insp.get_indexes("chunks")
    } == {
        "ix_chunks_id": (["id"], 0),
        "ix_chunks_vector_id": (["vector_id"], 1),
    }
    assert _revision(engine) == REVISION
    engine.dispose()


def test_graph_head_has_provenance_constraints_indexes_and_cascades(tmp_path):
    db_url = _db_url(tmp_path / "graph-head.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    inspector = inspect(engine)

    assert {
        constraint["name"]
        for constraint in inspector.get_check_constraints("documents")
    } == {"ck_documents_ingestion_status", "ck_documents_trust_tier", "ck_documents_trust_score"}

    assert {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("entity_mentions")
    } == {"uq_entity_mentions_entity_extraction"}
    assert {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("graph_edges")
    } == {"uq_graph_edges_triplet"}
    assert {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("graph_edge_evidence")
    } == {"uq_graph_edge_evidence_location"}
    assert {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("graph_entities")
    } == {"uq_graph_entities_name_type"}
    entity_indexes = {
        index["name"]: (index["column_names"], index["unique"])
        for index in inspector.get_indexes("graph_entities")
    }
    assert entity_indexes["ix_graph_entities_canonical_name"] == (
        ["canonical_name"],
        0,
    )
    edge_indexes = {
        index["name"] for index in inspector.get_indexes("graph_edges")
    }
    assert "ix_graph_edges_predicate" in edge_indexes

    for table in (
        "graph_extractions",
        "entity_mentions",
        "graph_edges",
        "graph_edge_evidence",
    ):
        foreign_keys = inspector.get_foreign_keys(table)
        assert foreign_keys
        assert all(
            foreign_key["options"] == {"ondelete": "CASCADE"}
            for foreign_key in foreign_keys
        )
    engine.dispose()


def test_graph_head_rejects_invalid_document_ingestion_status(tmp_path):
    db_url = _db_url(tmp_path / "invalid-document-status.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)

    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO documents (title, ingestion_status) "
                    "VALUES ('invalid', 'not_a_state')"
                )
            )

    engine.dispose()


def test_migration_entrypoint_migrates_configured_fresh_database(tmp_path):
    db_url = _db_url(tmp_path / "configured-fresh.db")

    result = _run_migration_entrypoint(db_url)

    assert result.returncode == 0, result.stderr
    engine = create_engine(db_url)
    assert set(inspect(engine).get_table_names()) == HEAD_TABLES
    assert _revision(engine) == REVISION
    engine.dispose()


def test_migration_entrypoint_adopts_matching_legacy_database_and_preserves_rows(tmp_path):
    db_url = _db_url(tmp_path / "legacy.db")
    command.upgrade(_alembic_config(db_url), BASELINE_REVISION)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO documents (id, title, source, tags) VALUES (7, 'Legacy', 'unit', 'kept')")
        )
        conn.execute(text("DROP TABLE alembic_version"))
    engine.dispose()

    result = _run_migration_entrypoint(db_url)

    assert result.returncode == 0, result.stderr
    engine = create_engine(db_url)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT title, source, tags, ingestion_status, failure_code "
                "FROM documents WHERE id = 7"
            )
        ).one()
    assert tuple(row) == ("Legacy", "unit", "kept", "ready", None)
    assert _revision(engine) == REVISION
    engine.dispose()


def test_legacy_adoption_stamps_baseline_then_applies_future_migrations(tmp_path, monkeypatch):
    db_url = _db_url(tmp_path / "legacy-future.db")
    command.upgrade(_alembic_config(db_url), BASELINE_REVISION)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO documents (id, title) VALUES (9, 'Before Future')"))
        conn.execute(text("DROP TABLE alembic_version"))
    engine.dispose()

    scripts = tmp_path / "alembic"
    shutil.copytree(
        PROJECT_ROOT / "app" / "persistence" / "alembic",
        scripts,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    future_revision = "future000001"
    (scripts / "versions" / f"{future_revision}_future_table.py").write_text(
        f'''"""test-only post-baseline migration"""
from alembic import op
import sqlalchemy as sa

revision = "{future_revision}"
down_revision = "{REVISION}"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table("future_entities", sa.Column("id", sa.Integer(), primary_key=True))

def downgrade():
    op.drop_table("future_entities")
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(migrations, "MIGRATION_SCRIPTS", scripts)

    upgrade_database(db_url)

    engine = create_engine(db_url)
    assert "future_entities" in inspect(engine).get_table_names()
    with engine.connect() as conn:
        assert conn.execute(text("SELECT title FROM documents WHERE id = 9")).scalar() == "Before Future"
    assert _revision(engine) == future_revision
    engine.dispose()


def test_migration_entrypoint_rejects_partial_unversioned_schema(tmp_path):
    db_url = _db_url(tmp_path / "partial.db")
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE documents (id INTEGER PRIMARY KEY, title VARCHAR NOT NULL)"))
    engine.dispose()

    result = _run_migration_entrypoint(db_url)

    assert result.returncode != 0
    assert "unversioned database schema does not match" in result.stderr.lower()
    engine = create_engine(db_url)
    assert "alembic_version" not in inspect(engine).get_table_names()
    engine.dispose()


def test_alembic_cli_honors_rag_database_url(tmp_path):
    db_url = _db_url(tmp_path / "cli-configured.db")
    env = os.environ.copy()
    env["RAG_DATABASE_URL"] = db_url

    upgrade = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), "upgrade", "head"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert upgrade.returncode == 0, upgrade.stderr

    current = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), "current"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert current.returncode == 0, current.stderr
    assert REVISION in current.stdout

    check = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(ALEMBIC_INI), "check"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    assert "No new upgrade operations detected" in check.stdout

    engine = create_engine(db_url)
    assert set(inspect(engine).get_table_names()) == HEAD_TABLES
    assert _revision(engine) == REVISION
    engine.dispose()


def test_migration_entrypoint_imports_legacy_sqlite_into_durable_target(tmp_path):
    legacy_path = tmp_path / "legacy-source.db"
    legacy_url = _db_url(legacy_path)
    command.upgrade(_alembic_config(legacy_url), BASELINE_REVISION)
    legacy_engine = create_engine(legacy_url)
    with legacy_engine.begin() as conn:
        conn.execute(text("INSERT INTO documents (id, title) VALUES (11, 'Moved Legacy')"))
        conn.execute(text("DROP TABLE alembic_version"))
    legacy_engine.dispose()

    durable_url = _db_url(tmp_path / "data" / "rag.db")
    result = _run_migration_entrypoint(
        durable_url, "--legacy-sqlite-path", str(legacy_path)
    )

    assert result.returncode == 0, result.stderr
    durable_engine = create_engine(durable_url)
    with durable_engine.connect() as conn:
        assert conn.execute(text("SELECT title FROM documents WHERE id = 11")).scalar() == "Moved Legacy"
    assert _revision(durable_engine) == REVISION
    durable_engine.dispose()


@pytest.mark.parametrize("target_object", ["table", "view"])
def test_legacy_import_refuses_any_existing_target_schema_object(tmp_path, target_object):
    legacy_path = tmp_path / "legacy-source.db"
    command.upgrade(_alembic_config(_db_url(legacy_path)), BASELINE_REVISION)
    legacy_engine = create_engine(_db_url(legacy_path))
    with legacy_engine.begin() as conn:
        conn.execute(text("DROP TABLE alembic_version"))
    legacy_engine.dispose()

    target_path = tmp_path / f"target-{target_object}.db"
    with sqlite3.connect(target_path) as connection:
        if target_object == "table":
            connection.execute("CREATE TABLE sentinel (value INTEGER NOT NULL)")
            connection.execute("INSERT INTO sentinel VALUES (73)")
        else:
            connection.execute("CREATE VIEW sentinel AS SELECT 73 AS value")

    result = _run_migration_entrypoint(
        _db_url(target_path), "--legacy-sqlite-path", str(legacy_path)
    )

    assert result.returncode != 0
    assert "refusing to overwrite non-empty target database" in result.stderr.lower()
    with sqlite3.connect(target_path) as connection:
        object_type = connection.execute(
            "SELECT type FROM sqlite_master WHERE name = 'sentinel'"
        ).fetchone()[0]
        value = connection.execute("SELECT value FROM sentinel").fetchone()[0]
    assert object_type == target_object
    assert value == 73


def test_unversioned_legacy_schema_with_extra_check_is_rejected(tmp_path):
    db_url = _db_url(tmp_path / "legacy-extra-check.db")
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE documents ("
                "id INTEGER NOT NULL PRIMARY KEY, title VARCHAR NOT NULL "
                "CHECK(length(title) > 2), source VARCHAR, tags VARCHAR)"
            )
        )
        conn.execute(text("CREATE INDEX ix_documents_id ON documents (id)"))
        conn.execute(
            text(
                'CREATE TABLE chunks (id INTEGER NOT NULL PRIMARY KEY, document_id INTEGER NOT NULL, '
                '"index" INTEGER NOT NULL, text TEXT NOT NULL, start_offset INTEGER NOT NULL, '
                'end_offset INTEGER NOT NULL, FOREIGN KEY(document_id) REFERENCES documents (id) '
                "ON DELETE CASCADE)"
            )
        )
        conn.execute(text("CREATE INDEX ix_chunks_id ON chunks (id)"))
    engine.dispose()

    with pytest.raises(RuntimeError, match="Unversioned database schema does not match"):
        upgrade_database(db_url)

    engine = create_engine(db_url)
    assert "alembic_version" not in inspect(engine).get_table_names()
    assert inspect(engine).get_check_constraints("documents")
    engine.dispose()


def test_view_only_unversioned_database_is_rejected_without_mutation(tmp_path):
    database_path = tmp_path / "view-only.db"
    db_url = _db_url(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE VIEW sentinel AS SELECT 73 AS value")
        before = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()

    with pytest.raises(RuntimeError):
        upgrade_database(db_url)

    with sqlite3.connect(database_path) as connection:
        after = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        assert connection.execute("SELECT value FROM sentinel").fetchone()[0] == 73
    assert after == before


@pytest.mark.parametrize("object_kind", ["view", "trigger"])
def test_versioned_schema_rejects_changed_view_or_trigger_definition(
    tmp_path, monkeypatch, object_kind
):
    scripts, future_revision = _future_scripts_with_view_and_trigger(tmp_path)
    monkeypatch.setattr(migrations, "MIGRATION_SCRIPTS", scripts)
    database_path = tmp_path / f"changed-{object_kind}.db"
    db_url = _db_url(database_path)
    upgrade_database(db_url)

    with sqlite3.connect(database_path) as connection:
        if object_kind == "view":
            connection.execute("DROP VIEW document_titles")
            connection.execute(
                "CREATE VIEW document_titles AS SELECT id, source AS title FROM documents"
            )
        else:
            connection.execute("DROP TRIGGER document_title_audit")
            connection.execute(
                "CREATE TRIGGER document_title_audit AFTER INSERT ON documents "
                "BEGIN INSERT INTO document_audit (document_id, recorded_title) "
                "VALUES (NEW.id, 'tampered'); END"
            )

    with pytest.raises(RuntimeError, match="Versioned database schema does not match"):
        upgrade_database(db_url)

    engine = create_engine(db_url)
    assert _revision(engine) == future_revision
    engine.dispose()


@pytest.mark.parametrize(
    "mutation",
    ["DROP TABLE chunks", "DROP INDEX ix_documents_id"],
)
def test_versioned_schema_rejects_missing_table_or_index(tmp_path, mutation):
    db_url = _db_url(tmp_path / f"versioned-drift-{mutation.split()[-1]}.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(mutation))
    engine.dispose()

    with pytest.raises(RuntimeError, match="Versioned database schema does not match"):
        upgrade_database(db_url)


def test_versioned_schema_rejects_column_and_fk_drift(tmp_path):
    database_path = tmp_path / "versioned-column-fk-drift.db"
    db_url = _db_url(database_path)
    upgrade_database(db_url)
    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("ALTER TABLE chunks RENAME TO chunks_old")
        connection.execute(
            'CREATE TABLE chunks (id INTEGER NOT NULL PRIMARY KEY, document_id INTEGER NOT NULL, '
            '"index" INTEGER NOT NULL, text TEXT, start_offset INTEGER NOT NULL, '
            'end_offset INTEGER NOT NULL, FOREIGN KEY(document_id) REFERENCES documents (id))'
        )
        connection.execute("DROP TABLE chunks_old")
        connection.execute("CREATE INDEX ix_chunks_id ON chunks (id)")

    with pytest.raises(RuntimeError, match="Versioned database schema does not match"):
        upgrade_database(db_url)


def test_importing_api_does_not_run_migrations(tmp_path):
    db_url = _db_url(tmp_path / "api-import.db")
    env = os.environ.copy()
    env["RAG_DATABASE_URL"] = db_url

    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    engine = create_engine(db_url)
    assert inspect(engine).get_table_names() == []
    engine.dispose()


def test_parallel_api_imports_succeed_after_one_shot_migration(tmp_path):
    db_url = _db_url(tmp_path / "parallel-api.db")
    migration = _run_migration_entrypoint(db_url)
    assert migration.returncode == 0, migration.stderr

    env = os.environ.copy()
    env["RAG_DATABASE_URL"] = db_url
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", "import app.main"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    results = [worker.communicate(timeout=30) + (worker.returncode,) for worker in workers]

    assert all(returncode == 0 for stdout, stderr, returncode in results), results
    engine = create_engine(db_url)
    assert _revision(engine) == REVISION
    engine.dispose()


def test_upgrade_head_is_idempotent(tmp_path):
    db_url = _db_url(tmp_path / "idempotent.db")
    upgrade_database(db_url)
    upgrade_database(db_url)

    engine = create_engine(db_url)
    assert set(inspect(engine).get_table_names()) == HEAD_TABLES
    assert _revision(engine) == REVISION
    engine.dispose()


def test_downgrade_base_then_upgrade_restores_schema(tmp_path):
    db_url = _db_url(tmp_path / "cycle.db")
    upgrade_database(db_url)
    command.downgrade(_alembic_config(db_url), "base")

    engine = create_engine(db_url)
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()

    upgrade_database(db_url)
    engine = create_engine(db_url)
    assert set(inspect(engine).get_table_names()) == HEAD_TABLES
    assert _revision(engine) == REVISION
    engine.dispose()


def test_production_engine_enforces_fk_rejection_and_cascade(tmp_path):
    db_url = _db_url(tmp_path / "fk.db")
    upgrade_database(db_url)
    engine = create_database_engine(db_url)

    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    'INSERT INTO chunks (document_id, "index", text, start_offset, end_offset) '
                    "VALUES (999, 0, 'orphan', 0, 6)"
                )
            )

    with engine.begin() as conn:
        conn.execute(text("INSERT INTO documents (id, title) VALUES (1, 'Parent')"))
        conn.execute(
            text(
                'INSERT INTO chunks (id, document_id, "index", text, start_offset, end_offset) '
                "VALUES (1, 1, 0, 'child', 0, 5)"
            )
        )
        conn.execute(text("DELETE FROM documents WHERE id = 1"))

    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar() == 0
    engine.dispose()


@pytest.mark.parametrize(
    ("statement", "missing_column"),
    [
        ("INSERT INTO documents (id, title) VALUES (1, NULL)", "documents.title"),
        (
            'INSERT INTO chunks (id, document_id, "index", text, start_offset, end_offset) '
            "VALUES (1, NULL, 0, 'text', 0, 4)",
            "chunks.document_id",
        ),
        (
            'INSERT INTO chunks (id, document_id, "index", text, start_offset, end_offset) '
            "VALUES (1, 1, NULL, 'text', 0, 4)",
            "chunks.index",
        ),
        (
            'INSERT INTO chunks (id, document_id, "index", text, start_offset, end_offset) '
            "VALUES (1, 1, 0, NULL, 0, 4)",
            "chunks.text",
        ),
        (
            'INSERT INTO chunks (id, document_id, "index", text, start_offset, end_offset) '
            "VALUES (1, 1, 0, 'text', NULL, 4)",
            "chunks.start_offset",
        ),
        (
            'INSERT INTO chunks (id, document_id, "index", text, start_offset, end_offset) '
            "VALUES (1, 1, 0, 'text', 0, NULL)",
            "chunks.end_offset",
        ),
    ],
)
def test_required_columns_reject_null(tmp_path, statement, missing_column):
    db_url = _db_url(tmp_path / f"not-null-{missing_column.replace('.', '-')}.db")
    upgrade_database(db_url)
    engine = create_database_engine(db_url)
    if missing_column.startswith("chunks."):
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO documents (id, title) VALUES (1, 'Parent')"))

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(statement))
    engine.dispose()


# ---------------------------------------------------------------------------
# 10C.4: d9b5f7c1e4a3 migration matrix
# ---------------------------------------------------------------------------

D9_HEAD = "d9b5f7c1e4a3"


def test_d9_upgrade_creates_safety_tables_with_exact_columns(tmp_path):
    db_url = _db_url(tmp_path / "d9.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    insp = inspect(engine)
    assert "safety_review_runs" in insp.get_table_names()
    assert "safety_findings" in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("safety_review_runs")}
    expected = {"id", "scope", "status", "document_id", "chunk_id",
                "document_id_snapshot", "chunk_id_snapshot",
                "retrieval_audit_id", "input_sha256", "policy_version",
                "detector_version", "provider", "model", "prompt_version",
                "schema_version", "llm_status", "final_action",
                "failure_code", "created_at", "completed_at"}
    assert cols == expected
    engine.dispose()


def test_d9_named_checks_exist(tmp_path):
    db_url = _db_url(tmp_path / "d9-checks.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    insp = inspect(engine)
    names = {c["name"] for c in insp.get_check_constraints("safety_review_runs")}
    expected = {"ck_safety_runs_scope", "ck_safety_runs_status",
                "ck_safety_runs_llm_status", "ck_safety_runs_action",
                "ck_safety_runs_hash", "ck_safety_runs_provenance",
                "ck_safety_runs_lifecycle"}
    assert expected.issubset(names)
    engine.dispose()


def test_d9_partial_unique_indexes_exist(tmp_path):
    db_url = _db_url(tmp_path / "d9-idx.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.connect() as conn:
        idx_rows = conn.execute(text(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND tbl_name='safety_review_runs' AND sql IS NOT NULL"
        )).fetchall()
    partial_sql = " ".join(r[0] for r in idx_rows)
    assert "CREATE UNIQUE INDEX" in partial_sql
    assert "scope = 'ingestion'" in partial_sql and "document_id_snapshot" in partial_sql
    assert "scope = 'context'" in partial_sql and "chunk_id_snapshot" in partial_sql
    assert "scope = 'answer'" in partial_sql
    engine.dispose()


def _d9_insert_extraction(engine, *, error_code, attempt_count, sha):
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, error_code, "
            "input_sha256, attempt_count, is_identity_owner, completed_at) "
            "VALUES (1, 'p', 'm', 'pv', 'sv', 'skipped', :code, :sha, "
            ":attempts, 1, '2026-08-01T00:00:00Z')"
        ), {"code": error_code, "sha": sha, "attempts": attempt_count})


def test_d9_graph_extractions_skip_check_includes_safety_blocked(tmp_path):
    db_url = _db_url(tmp_path / "d9-skip.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO chunks (document_id, \"index\", text, "
            "start_offset, end_offset) VALUES (0, 0, 'x', 0, 1)"))
    # valid at d9: skipped + safety_blocked + attempt_count == 0
    _d9_insert_extraction(engine, error_code="safety_blocked",
                          attempt_count=0, sha="a" * 64)
    # rejected: skipped requires attempt_count == 0 (distinct input_sha256 so
    # the skip-reason CHECK raises, not the identity-owner unique index)
    with pytest.raises(IntegrityError):
        _d9_insert_extraction(engine, error_code="safety_blocked",
                              attempt_count=1, sha="b" * 64)
    # rejected: only the three stable skip codes are permitted
    with pytest.raises(IntegrityError):
        _d9_insert_extraction(engine, error_code="unsupported_skip_code",
                              attempt_count=0, sha="c" * 64)
    engine.dispose()


def test_d9_candidate_decisions_check_includes_rejected_safety(tmp_path):
    db_url = _db_url(tmp_path / "d9-decision.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    insp = inspect(engine)
    decision_sql = " ".join(
        c.get("sqltext", "")
        for c in insp.get_check_constraints("retrieval_candidate_decisions")
    )
    assert "rejected_safety" in decision_sql
    assert "rejected_injection" in decision_sql
    assert "selected" in decision_sql
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO retrieval_candidate_decisions (decision) "
                "VALUES ('rejected_safety_typo')"
            ))
    engine.dispose()


def test_d9_downgrade_refuses_when_safety_blocked_rows_present(tmp_path):
    db_url = _db_url(tmp_path / "d9-refuse.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO chunks (document_id, \"index\", text, "
            "start_offset, end_offset) VALUES (0, 0, 'x', 0, 1)"))
    _d9_insert_extraction(engine, error_code="safety_blocked",
                          attempt_count=0, sha="a" * 64)
    with pytest.raises(RuntimeError):
        command.downgrade(_alembic_config(db_url), "c8a4e6b0d3f2")
    engine.dispose()


def test_d9_downgrade_refuses_when_rejected_safety_rows_present(tmp_path):
    db_url = _db_url(tmp_path / "d9-refuse-rs.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    sha = "a" * 64
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO retrieval_audits (id, query_sha256, retrieval_mode, "
            "status, provenance_policy_version, retrieval_policy_version, "
            "context_policy_version, completed_at) "
            "VALUES ('audit-1', :sha, 'vector', 'completed', 'p', 'r', 'c', "
            "'2026-08-01T00:00:00Z')"
        ), {"sha": "b" * 64})
        conn.execute(text(
            "INSERT INTO retrieval_candidate_decisions "
            "(audit_id, chunk_id_snapshot, document_id_snapshot, "
            "content_sha256, decision, reason_codes, provenance_score) "
            "VALUES ('audit-1', 1, 1, :sha, 'rejected_safety', '[]', 0.5)"
        ), {"sha": sha})
    with pytest.raises(RuntimeError):
        command.downgrade(_alembic_config(db_url), "c8a4e6b0d3f2")
    engine.dispose()


def test_d9_downgrade_representable_data_restores_c8_b7_exactly(tmp_path):
    db_url = _db_url(tmp_path / "d9-down-ok.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO chunks (document_id, \"index\", text, "
            "start_offset, end_offset) VALUES (0, 0, 'x', 0, 1)"))
    _d9_insert_extraction(engine, error_code="extraction_disabled",
                          attempt_count=0, sha="a" * 64)
    rows_before = engine.connect().execute(text(
        "SELECT id, status, error_code FROM graph_extractions ORDER BY id"
    )).fetchall()
    command.downgrade(_alembic_config(db_url), "c8a4e6b0d3f2")
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    assert "safety_review_runs" not in tables
    assert "safety_findings" not in tables
    assert _revision(engine) == "c8a4e6b0d3f2"
    ge_sql = " ".join(
        c.get("sqltext", "")
        for c in insp.get_check_constraints("graph_extractions")
    )
    assert "safety_blocked" not in ge_sql
    dec_sql = " ".join(
        c.get("sqltext", "")
        for c in insp.get_check_constraints("retrieval_candidate_decisions")
    )
    assert "rejected_safety" not in dec_sql
    rows_after = engine.connect().execute(text(
        "SELECT id, status, error_code FROM graph_extractions ORDER BY id"
    )).fetchall()
    assert rows_after == rows_before
    engine.dispose()


def test_d9_re_upgrade_after_downgrade_is_lossless(tmp_path):
    db_url = _db_url(tmp_path / "d9-roundtrip.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO chunks (document_id, \"index\", text, "
            "start_offset, end_offset) VALUES (0, 0, 'x', 0, 1)"))
    _d9_insert_extraction(engine, error_code="extraction_disabled",
                          attempt_count=0, sha="a" * 64)
    fingerprint_before = engine.connect().execute(text(
        "SELECT id, status, error_code, attempt_count, input_sha256 "
        "FROM graph_extractions ORDER BY id"
    )).fetchall()
    command.downgrade(_alembic_config(db_url), "c8a4e6b0d3f2")
    upgrade_database(db_url)
    assert _revision(engine) == D9_HEAD
    fingerprint_after = engine.connect().execute(text(
        "SELECT id, status, error_code, attempt_count, input_sha256 "
        "FROM graph_extractions ORDER BY id"
    )).fetchall()
    assert fingerprint_after == fingerprint_before
    insp = inspect(engine)
    assert "safety_review_runs" in insp.get_table_names()
    assert "safety_findings" in insp.get_table_names()
    engine.dispose()


def test_d9_upgrade_from_c8_with_data_preserves_all_rows(tmp_path):
    db_url = _db_url(tmp_path / "d9-from-c8.db")
    command.upgrade(_alembic_config(db_url), "c8a4e6b0d3f2")
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO chunks (document_id, \"index\", text, "
            "start_offset, end_offset) VALUES (0, 0, 'x', 0, 1)"))
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO graph_extractions (chunk_id, provider, model, "
            "prompt_version, schema_version, status, error_code, "
            "input_sha256, attempt_count, is_identity_owner, completed_at) "
            "VALUES (1, 'p', 'm', 'pv', 'sv', 'skipped', 'extraction_disabled', "
            ":sha, 0, 1, '2026-08-01T00:00:00Z')"
        ), {"sha": "a" * 64})
    rows_before = engine.connect().execute(text(
        "SELECT id, input_sha256 FROM graph_extractions ORDER BY id"
    )).fetchall()
    engine.dispose()
    upgrade_database(db_url)
    engine = create_engine(db_url)
    assert _revision(engine) == D9_HEAD
    rows_after = engine.connect().execute(text(
        "SELECT id, input_sha256 FROM graph_extractions ORDER BY id"
    )).fetchall()
    assert rows_after == rows_before
    insp = inspect(engine)
    assert "safety_review_runs" in insp.get_table_names()
    assert "safety_findings" in insp.get_table_names()
    assert engine.connect().execute(text(
        "SELECT COUNT(*) FROM safety_review_runs")).scalar() == 0
    assert engine.connect().execute(text(
        "SELECT COUNT(*) FROM safety_findings")).scalar() == 0
    engine.dispose()


def test_d9_upgrade_from_every_prior_head(tmp_path):
    for head in ["dee48bc24a7f", "4c9a8d7e6f5b", "a6e2c4f8b1d9",
                 "b7f3d5a9c2e1", "c8a4e6b0d3f2"]:
        db_url = _db_url(tmp_path / f"from-{head}.db")
        command.upgrade(_alembic_config(db_url), head)
        upgrade_database(db_url)
        engine = create_engine(db_url)
        assert _revision(engine) == D9_HEAD
        engine.dispose()


def test_d9_fk_cascade_deletes_safety_findings_on_run_delete(tmp_path):
    import sqlite3
    db_path = tmp_path / "d9-cascade.db"
    upgrade_database(_db_url(db_path))
    sha = "a" * 64
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.execute(
        "INSERT INTO safety_review_runs (scope, status, input_sha256, "
        "policy_version, detector_version, llm_status, document_id_snapshot) "
        "VALUES ('ingestion', 'pending', ?, 'safety-v1', 'det-1', 'skipped', 1)",
        (sha,))
    rid = cur.lastrowid
    conn.execute(
        "INSERT INTO safety_findings (review_run_id, category, severity, "
        "action, start_offset, end_offset, source_rule_ids, excerpt_sha256) "
        "VALUES (?, 'violence', 3, 'warn', 0, 4, ?, ?)",
        (rid, '["SAF001_violence"]', sha))
    conn.execute("DELETE FROM safety_review_runs WHERE id = ?", (rid,))
    remaining = conn.execute(
        "SELECT COUNT(*) FROM safety_findings WHERE review_run_id = ?", (rid,)
    ).fetchone()[0]
    conn.commit()
    conn.close()
    assert remaining == 0


def test_d9_fk_set_null_on_document_delete_preserves_snapshot(tmp_path):
    import sqlite3
    db_path = tmp_path / "d9-setnull.db"
    upgrade_database(_db_url(db_path))
    sha = "a" * 64
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.execute(
        "INSERT INTO documents (title, ingestion_status) VALUES ('D', 'failed')")
    did = cur.lastrowid
    conn.execute(
        "INSERT INTO safety_review_runs (scope, status, input_sha256, "
        "policy_version, detector_version, llm_status, document_id, "
        "document_id_snapshot) VALUES ('ingestion', 'pending', ?, "
        "'safety-v1', 'det-1', 'skipped', ?, ?)",
        (sha, did, did))
    conn.execute("DELETE FROM documents WHERE id = ?", (did,))
    row = conn.execute(
        "SELECT document_id, document_id_snapshot FROM safety_review_runs"
    ).fetchone()
    conn.commit()
    conn.close()
    assert row[0] is None
    assert row[1] == did


def test_d9_lifecycle_check_rejects_invalid_combinations(tmp_path):
    db_url = _db_url(tmp_path / "d9-life.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    sha = "a" * 64
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO safety_review_runs (scope, status, input_sha256, "
                "policy_version, detector_version, llm_status, document_id_snapshot, "
                "completed_at) VALUES ('ingestion', 'pending', :sha, 'safety-v1', "
                "'det-1', 'skipped', 1, '2026-08-01T00:00:00Z')"
            ), {"sha": sha})
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO safety_review_runs (scope, status, input_sha256, "
                "policy_version, detector_version, llm_status, document_id_snapshot, "
                "final_action, completed_at, failure_code) VALUES ('ingestion', "
                "'succeeded', :sha, 'safety-v1', 'det-1', 'skipped', 1, 'allow', "
                "'2026-08-01T00:00:00Z', 'boom')"
            ), {"sha": sha})
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO safety_review_runs (scope, status, input_sha256, "
                "policy_version, detector_version, llm_status, document_id_snapshot, "
                "completed_at) VALUES ('ingestion', 'failed', :sha, 'safety-v1', "
                "'det-1', 'failed', 1, '2026-08-01T00:00:00Z')"
            ), {"sha": sha})
    engine.dispose()


def test_d9_provenance_check_rejects_wrong_scope_column_combination(tmp_path):
    db_url = _db_url(tmp_path / "d9-prov.db")
    upgrade_database(db_url)
    engine = create_engine(db_url)
    sha = "a" * 64
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO safety_review_runs (scope, status, input_sha256, "
                "policy_version, detector_version, llm_status, document_id_snapshot, "
                "chunk_id_snapshot) VALUES ('ingestion', 'pending', :sha, "
                "'safety-v1', 'det-1', 'skipped', 1, 1)"
            ), {"sha": sha})
    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO safety_review_runs (scope, status, input_sha256, "
                "policy_version, detector_version, llm_status, retrieval_audit_id, "
                "document_id_snapshot) VALUES ('answer', 'pending', :sha, "
                "'safety-v1', 'det-1', 'skipped', 'audit-1', 1)"
            ), {"sha": sha})
    engine.dispose()
