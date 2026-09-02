"""Phase 10B — dev_seed provenance contract tests (B-18, D-42).

Subprocess-level tests running scripts/dev_seed.py against an isolated
database and vector store, asserting server-side trust assignment and the
non-operator non-bypass contract.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.db import Base
from app.persistence import models  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def seed_env(tmp_path, monkeypatch):
    """Isolated DB + policy path for the dev_seed CLI subprocess."""
    db_path = tmp_path / "dev-seed.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(
        url, connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    engine.dispose()

    monkeypatch.setenv("RAG_DATABASE_URL", url)
    monkeypatch.setenv("RAG_CHROMA_HOST", "")  # ephemeral client in-process
    yield url, tmp_path
    # Chroma ephemeral per process; nothing to clean outside the tmp db.


def _child_env() -> dict[str, str]:
    """Explicit environment for the dev_seed child process.

    The overrides must be applied IN THE DICT, not relied upon via
    ``monkeypatch.setenv`` + platform env inheritance: Windows drops
    empty-valued variables when a child inherits the parent environment, so a
    ``RAG_CHROMA_HOST=""`` override set that way never reaches the child and
    ``scripts/dev_seed.py`` then falls back to the tree's ``.env`` (selecting
    the Chroma HttpClient). Building the dict explicitly and passing it via
    ``env=`` keeps the child hermetic regardless of platform inheritance
    behavior or an ``.env`` file being present.

    ``RAG_CHROMA_HOST=""`` is the config's "no host" form — an explicitly set
    (empty) env var takes precedence over ``.env`` in pydantic-settings, and
    ``Settings.chroma_host`` is falsy so ``_create_client`` selects the
    EphemeralClient.
    """
    env = dict(os.environ)
    env["RAG_CHROMA_HOST"] = ""  # '' is falsy -> ephemeral client in the child
    env["RAG_EMBEDDING_PROVIDER"] = "local"  # no model downloads in the child
    return env


def _run_dev_seed() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "dev_seed.py")],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=False,
        env=_child_env(),
    )


def test_dev_seed_assigns_server_trust_untrusted_and_origin(seed_env):
    url, _tmp = seed_env
    result = _run_dev_seed()
    assert result.returncode == 0, result.stderr

    engine = create_engine(url, connect_args={"check_same_thread": False})
    session = sessionmaker(bind=engine)()
    try:
        doc = (
            session.query(models.Document)
            .filter(models.Document.ingestion_origin == "dev_seed_cli")
            .one()
        )
        assert doc.source == "seed"
        assert doc.trust_tier == "untrusted"  # never trusted for dev_seed_cli
        assert doc.trust_policy_version == "source-trust-v1"
        assert doc.ingestion_status == "ready"
    finally:
        session.close()
        engine.dispose()


def test_dev_seed_refuses_blocked_or_operator_only_sources(seed_env, monkeypatch):
    """A policy that blocks (or protects) the 'seed' source forces exit 2 and
    writes no rows — the CLI can never bypass server trust rules."""
    import json

    url, tmp = seed_env
    blocked = tmp / "blocked-policy.json"
    blocked.write_text(json.dumps({
        "version": "source-trust-v1",
        "default": {"tier": "untrusted", "score": 0.2},
        "rules": [
            {"rule_id": "SRC_BLOCK", "source": "seed", "tier": "blocked",
             "score": 0.0, "requires_operator": False},
        ],
    }))
    monkeypatch.setenv("RAG_SOURCE_TRUST_POLICY_PATH", str(blocked))

    result = _run_dev_seed()
    assert result.returncode == 2
    assert "blocked" in result.stderr

    engine = create_engine(url, connect_args={"check_same_thread": False})
    session = sessionmaker(bind=engine)()
    try:
        assert session.query(models.Document).count() == 0
    finally:
        session.close()
        engine.dispose()
