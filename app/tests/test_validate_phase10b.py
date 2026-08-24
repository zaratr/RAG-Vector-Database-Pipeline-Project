"""Tests for the Phase 10B validator's embedding-regime precondition (R2b).

``scripts/validate_phase10b.py`` judges exact distance-calibrated retrieval
outcomes (its corpus fixtures are near-verbatim variants of the
calibration-corpus answer so the LIVE embedding model keeps every candidate
inside the calibrated ``max_distance``). Those verdicts are only meaningful
when the runtime resolves the embedding regime the retrieval-security policy
was calibrated for; any other regime (e.g. the deterministic local hash
provider, whose l2 distances sit far outside real-model calibrations) must
fail FAST with a machine-readable ``provider_mismatch`` (exit 2) instead of
producing meaningless distance verdicts. These tests pin the precondition
in-process without needing a live API.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_POLICY = PROJECT_ROOT / "config" / "retrieval-security-policy.json"


def _settings_for(monkeypatch, provider, model=None):
    """Resolve fresh settings against the committed policy under one regime."""
    monkeypatch.setenv(
        "RAG_RETRIEVAL_SECURITY_POLICY_PATH", str(COMMITTED_POLICY)
    )
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", provider)
    if model is not None:
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", model)
    else:
        monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        return get_settings()
    finally:
        get_settings.cache_clear()


def test_regime_precondition_passes_under_calibrated_fastembed(monkeypatch):
    """fastembed + the policy's calibrated model (the dev-stack regime) is the
    regime every distance verdict below the precondition was calibrated for."""
    from scripts.validate_phase10b import _assert_embedding_regime

    settings = _settings_for(monkeypatch, "fastembed", "jinaai/jina-clip-v1")

    assert _assert_embedding_regime(settings) is None


def test_regime_precondition_fails_fast_under_local_provider(
    monkeypatch, capsys
):
    """local (hash) regime + fastembed-calibrated policy -> exit 2 with a
    machine-readable provider_mismatch naming BOTH regimes."""
    from scripts.validate_phase10b import _assert_embedding_regime

    settings = _settings_for(monkeypatch, "local")

    with pytest.raises(SystemExit) as excinfo:
        _assert_embedding_regime(settings)
    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    assert "provider_mismatch" in message
    # Both regimes are named: the calibrated model and the runtime regime.
    assert "jinaai/jina-clip-v1" in message
    assert "local" in message


def test_regime_precondition_fails_fast_on_model_mismatch(monkeypatch, capsys):
    """fastembed provider with a DIFFERENT model is still a regime mismatch:
    the thresholds were calibrated for exactly one model identity."""
    from scripts.validate_phase10b import _assert_embedding_regime

    settings = _settings_for(monkeypatch, "fastembed", "other-org/other-model")

    with pytest.raises(SystemExit) as excinfo:
        _assert_embedding_regime(settings)
    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    assert "provider_mismatch" in message
    assert "jinaai/jina-clip-v1" in message
    assert "other-org/other-model" in message
