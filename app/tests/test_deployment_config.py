"""Phase 10B plan-body config assertions (B-19 and compose contract).

The ``migrate`` service must remain database-only: no ``env_file`` (so
``.env`` provider/operator secrets can never reach the migration process)
and an explicit environment allowlist of exactly ``RAG_DATABASE_URL``. The
``api`` service keeps its documented env structure. These tests parse the
committed YAML so the security property is regression-pinned rather than
only read-verified.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"

FORBIDDEN_ENV_KEY = re.compile(
    r"(?i)(token|secret|password|credential|api[_-]?key)"
)

API_ENVIRONMENT_KEYS = {
    "RAG_DATABASE_URL",
    "RAG_EXTRACTION_LEASE_SECONDS",
    "RAG_SOURCE_TRUST_POLICY_PATH",
    "RAG_OPERATOR_API_ENABLED",
    "RAG_OPERATOR_TOKEN",
    "RAG_SECURITY_AUDIT_RETENTION_DAYS",
    "RAG_RETRIEVAL_SECURITY_POLICY_PATH",
    "RAG_CONTEXT_SECURITY_POLICY_PATH",
    "RAG_INGESTION_REQUEST_MAX_BYTES",
    "RAG_INGESTION_FILE_MAX_BYTES",
    "RAG_INGESTION_EXTRACTED_MAX_BYTES",
    "RAG_INGEST_RATE_LIMIT_REQUESTS",
    "RAG_INGEST_RATE_LIMIT_WINDOW_SECONDS",
}


def _compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


def test_migrate_service_has_no_env_file():
    """B-19: removing ``env_file`` from migrate is the security control that
    keeps provider/operator secrets out of the migration process."""
    migrate = _compose()["services"]["migrate"]
    assert "env_file" not in migrate


def test_migrate_environment_is_exactly_the_database_url():
    migrate_env = _compose()["services"]["migrate"]["environment"]
    assert set(migrate_env.keys()) == {"RAG_DATABASE_URL"}
    assert not any(FORBIDDEN_ENV_KEY.search(key) for key in migrate_env)


def test_migrate_runs_only_the_migration_entrypoint():
    migrate = _compose()["services"]["migrate"]
    assert migrate["command"] == ["python", "-m", "app.core.migrations"]


def test_api_service_keeps_env_file_and_documented_environment():
    api = _compose()["services"]["api"]
    assert api.get("env_file") == [".env"]
    assert set(api["environment"].keys()) == API_ENVIRONMENT_KEYS


def test_compose_rendered_defaults_match_documented_values():
    """The ``${VAR:-default}`` fallbacks in the committed file equal the
    documented defaults (operator API disabled by default, empty operator
    token, 30-day audit retention)."""
    api_env = _compose()["services"]["api"]["environment"]
    assert api_env["RAG_OPERATOR_API_ENABLED"] == "${RAG_OPERATOR_API_ENABLED:-false}"
    assert api_env["RAG_OPERATOR_TOKEN"] == "${RAG_OPERATOR_TOKEN:-}"
    assert api_env["RAG_SECURITY_AUDIT_RETENTION_DAYS"] == "${RAG_SECURITY_AUDIT_RETENTION_DAYS:-30}"
    assert api_env["RAG_SOURCE_TRUST_POLICY_PATH"] == (
        "${RAG_SOURCE_TRUST_POLICY_PATH:-/app/config/source-trust-policy.json}")
    assert api_env["RAG_RETRIEVAL_SECURITY_POLICY_PATH"] == (
        "${RAG_RETRIEVAL_SECURITY_POLICY_PATH:-/app/config/retrieval-security-policy.json}")
    assert api_env["RAG_CONTEXT_SECURITY_POLICY_PATH"] == (
        "${RAG_CONTEXT_SECURITY_POLICY_PATH:-/app/config/context-security-policy.json}")
    assert _compose()["services"]["migrate"]["environment"]["RAG_DATABASE_URL"] == (
        "${RAG_DATABASE_URL:-sqlite:////data/rag.db}")
