"""Tests for the hardened Phase 10A.7 acceptance validator (F9 remediation).

``scripts/validate_phase10a.py`` is load-bearing inside the recorded phase10a
gate: its exit code is parsed by ``run_recorded_gate.py``. These tests prove
the remediation contract:

* the deterministic lane ASSERTS exact document/entity/evidence counts,
  directional 1/2/3-hop paths with citations, genuinely executed hybrid
  retrieval (sources derived from real results), and RRF-60 fusion scores
  derived from the real per-side rankings — never hardcoded output;
* the live lane persists its unique document through the production
  ``ingest_text`` path with the real Ollama/Gemma extractor, asserts
  schema/grounded/non-canned from the persisted SQL rows, and fails with a
  NON-ZERO exit when the provider is unavailable or output is invalid;
* disposable-lane fingerprints restore, configured production SQL/Chroma
  fingerprints are asserted unchanged, and ``"restored": true`` is emitted
  only when genuinely verified.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.services import graph_extraction as ge
from app.services.graph_extraction import (
    GraphProviderUnavailable,
    OllamaGraphExtractor,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "validate_phase10a.py"
)

PINNED_DETERMINISTIC = {
    "documents": 3,
    "entities": 4,
    "evidence": 3,
    "graph_hops": [1, 2, 3],
    "hybrid_sources": ["graph", "vector"],
}
PINNED_LIVE = {"schema_valid": True, "grounded": True, "non_canned": True}
PINNED_FULL = {
    "deterministic": PINNED_DETERMINISTIC,
    "live": PINNED_LIVE,
    "restored": True,
}


def _install_permissive_retrieval_policy(monkeypatch):
    """Neutralize 10B's retrieval-security distance/cap controls in-process.

    The merged tree applies 10B's security layer inside ``retrieve_contexts``;
    the 10A deterministic lane uses HashEmbeddingProvider whose l2 distances
    (~1e2) exceed the production-calibrated max_distance. The lane pins 10A
    topology/fusion semantics, so install a permissive policy through the
    shipped ``_POLICY_CACHE`` hook (restored automatically by monkeypatch).
    """
    from app.services import retrieval as retrieval_module
    from app.services.retrieval_security import RetrievalSecurityPolicy

    monkeypatch.setattr(
        retrieval_module,
        "_POLICY_CACHE",
        RetrievalSecurityPolicy(
            version="retrieval-security-v1",
            metric="l2",
            max_distance=1e9,
            per_source_cap=1000,
            per_document_cap=1000,
            max_candidates=1000,
            near_duplicate_jaccard=1.0,
            calibration_fixture_sha256="phase10a-deterministic-lane",
            calibration_clean_recall=1.0,
            calibration_poison_share=0.0,
            calibration_tool_version="calibrate-v1",
            calibration_embedding_model="jinaai/jina-clip-v1",
        ),
    )


# ---------------------------------------------------------------------------
# Live-extractor fakes shared by the in-process tests
# ---------------------------------------------------------------------------


async def _grounded_extract(self, text):
    """Mirror a real non-canned provider: build one relation whose evidence
    is an exact substring of the actual unique text passed in."""
    sentence = text.split(".")[0] + "."
    left, right = sentence.split(" manages ")
    target = right.rstrip(".")
    start = text.index(sentence)
    return [
        ge.ExtractedRelation(
            source=ge.ExtractedEntity(
                name=left, canonical_name=left.casefold(), entity_type="person"
            ),
            predicate="manages",
            target=ge.ExtractedEntity(
                name=target, canonical_name=target.casefold(), entity_type="project"
            ),
            evidence=sentence,
            evidence_start=start,
            evidence_end=start + len(sentence),
            confidence=0.9,
        )
    ]


# ---------------------------------------------------------------------------
# Subprocess lane: proves exit codes, sys.path bootstrap, and pinned JSON
# ---------------------------------------------------------------------------

# Startup shim loaded via PYTHONPATH inside the subprocess. It patches the
# source modules at the moment they finish importing, BEFORE the script binds
# their names, so the script under test exercises the patched production path.
_SITECUSTOMIZE = textwrap.dedent(
    '''
    import builtins
    import os
    import sys

    _mode = os.environ.get("PHASE10A_TEST_MODE")
    _real_import = builtins.__import__
    _patched = set()

    def _patch(name, module):
        if name == "app.services.graph_extraction":
            if _mode == "unavailable":

                async def extract(self, text):
                    raise module.GraphProviderUnavailable("connection refused")

            else:

                async def extract(self, text):
                    sentence = text.split(".")[0] + "."
                    left, right = sentence.split(" manages ")
                    target = right.rstrip(".")
                    start = text.index(sentence)
                    return [
                        module.ExtractedRelation(
                            source=module.ExtractedEntity(
                                name=left,
                                canonical_name=left.casefold(),
                                entity_type="person",
                            ),
                            predicate="manages",
                            target=module.ExtractedEntity(
                                name=target,
                                canonical_name=target.casefold(),
                                entity_type="project",
                            ),
                            evidence=sentence,
                            evidence_start=start,
                            evidence_end=start + len(sentence),
                            confidence=0.9,
                        )
                    ]

            module.OllamaGraphExtractor.extract = extract
        elif name == "app.services.retrieval":
            # Merged-tree adaptation: the deterministic lane drives hybrid
            # retrieval with HashEmbeddingProvider, whose l2 distances
            # (~1e2) exceed the production-calibrated max_distance. The lane
            # pins 10A topology/fusion semantics, so neutralize ONLY the
            # 10B distance/cap controls via the shipped _POLICY_CACHE hook.
            from app.services.retrieval_security import RetrievalSecurityPolicy

            module._POLICY_CACHE = RetrievalSecurityPolicy(
                version="retrieval-security-v1",
                metric="l2",
                max_distance=1e9,
                per_source_cap=1000,
                per_document_cap=1000,
                max_candidates=1000,
                near_duplicate_jaccard=1.0,
                calibration_fixture_sha256="phase10a-deterministic-lane",
                calibration_clean_recall=1.0,
                calibration_poison_share=0.0,
                calibration_tool_version="calibrate-v1",
                calibration_embedding_model="jinaai/jina-clip-v1",
            )
        elif (
            name == "app.persistence.graph_repository"
            and _mode == "persist_broken"
        ):
            def broken_persist(session, **kwargs):
                return None  # persist nothing: count assertions must fail

            module.persist_chunk_extraction = broken_persist

    def _patching_import(name, *args, **kwargs):
        module = _real_import(name, *args, **kwargs)
        if not _mode:
            return module
        if name in _patched:
            return module
        target = sys.modules.get(name)
        # Only patch once the module body has finished executing; during its
        # own nested imports the target attribute may not exist yet.
        if target is None:
            return module
        if name == "app.services.graph_extraction" and not hasattr(
            target, "OllamaGraphExtractor"
        ):
            return module
        if name == "app.services.retrieval" and not hasattr(
            target, "_apply_security_filter"
        ):
            return module
        if name == "app.persistence.graph_repository" and not hasattr(
            target, "persist_chunk_extraction"
        ):
            return module
        _patched.add(name)
        _patch(name, target)
        return module

    builtins.__import__ = _patching_import
    '''
)


def _run_validator_script(monkeypatch, tmp_path, mode="grounded"):
    """Run the validator as a real subprocess with hermetic environment.

    Production stores are made unconfigured (missing sqlite path, no Chroma
    host) so the configured-production fingerprint lane is exercised in its
    "nothing configured to protect" form; cwd is tmp_path so no .env applies.
    """
    (tmp_path / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    monkeypatch.setenv("PHASE10A_TEST_MODE", mode)
    monkeypatch.setenv("RAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("RAG_GRAPH_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:///{tmp_path}/absent.db")
    monkeypatch.delenv("RAG_CHROMA_HOST", raising=False)
    monkeypatch.delenv("RAG_CHROMA_PERSIST_DIRECTORY", raising=False)
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    return subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )


def test_validator_happy_path_exits_0_with_pinned_json(monkeypatch, tmp_path):
    """Grounded live provider + intact deterministic topology -> exit 0 and
    the exact plan-pinned summary shape."""
    result = _run_validator_script(monkeypatch, tmp_path, "grounded")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == PINNED_FULL
    assert "Traceback" not in result.stderr


def test_validator_deterministic_violation_exits_nonzero(monkeypatch, tmp_path):
    """A topology violation (persistence writes nothing) must FAIL the
    deterministic lane: non-zero exit and a machine-readable error, never the
    success JSON."""
    result = _run_validator_script(monkeypatch, tmp_path, "persist_broken")

    assert result.returncode != 0, result.stdout
    assert result.returncode == 1
    assert result.stdout.strip() == ""
    error = json.loads(result.stderr.strip().splitlines()[-1])
    assert error["lane"] == "deterministic"
    assert "check" in error
    assert "Traceback" not in result.stderr


def test_validator_provider_unavailable_exits_nonzero(monkeypatch, tmp_path):
    """Provider unavailability must fail the explicit live lane (exit 2),
    not be swallowed into a passing run."""
    result = _run_validator_script(monkeypatch, tmp_path, "unavailable")

    assert result.returncode != 0, result.stdout
    assert result.returncode == 2
    assert result.stdout.strip() == ""
    error = json.loads(result.stderr.strip().splitlines()[-1])
    assert error["error"] == "provider_unavailable"
    assert error["lane"] == "live"
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# In-process lane: direct unit access to the hardened internals
# ---------------------------------------------------------------------------


def _db_url(tmp_path, name):
    return f"sqlite:///{tmp_path / name}"


def _assert_disposable_db_empty(db_url):
    """Every application table must be back to zero rows after lane cleanup."""
    from sqlalchemy import create_engine, inspect

    engine = create_engine(db_url)
    inspector = inspect(engine)
    with engine.connect() as conn:
        for table in inspector.get_table_names():
            if table == "alembic_version":
                continue
            count = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar()
            assert count == 0, f"lane left {count} rows in {table}"
    engine.dispose()


@pytest.mark.asyncio
async def test_deterministic_lane_asserts_topology_and_cleans_up(monkeypatch, tmp_path):
    """Direct lane run returns the pinned summary with restored=True and
    leaves the disposable DB empty (self-cleaning proven)."""
    from scripts import validate_phase10a as validator

    _install_permissive_retrieval_policy(monkeypatch)
    summary, restored = await validator._run_deterministic(
        _db_url(tmp_path, "det.db"), "deadbeefdeadbeef"
    )

    assert summary == PINNED_DETERMINISTIC
    assert restored is True
    _assert_disposable_db_empty(_db_url(tmp_path, "det.db"))


@pytest.mark.asyncio
async def test_deterministic_lane_fails_on_broken_topology(
    monkeypatch, tmp_path
):
    """A seeded violation inside the production persistence path must raise a
    machine-readable failure instead of printing counts."""
    from scripts import validate_phase10a as validator

    monkeypatch.setattr(
        validator, "persist_chunk_extraction", lambda session, **kwargs: None
    )

    with pytest.raises(validator.ValidatorFailure) as excinfo:
        await validator._run_deterministic(
            _db_url(tmp_path, "det-broken.db"), "cafebabecafebabe"
        )
    assert excinfo.value.lane == "deterministic"
    assert excinfo.value.detail["check"]
    _assert_disposable_db_empty(_db_url(tmp_path, "det-broken.db"))


@pytest.mark.asyncio
async def test_deterministic_lane_requires_distinct_seed_sources(
    monkeypatch, tmp_path
):
    """Regression proof for the post-10B seeding fix: all three documents
    under ONE shared source must FAIL the lane — the merged 10B layer's
    ``per_source_cap: 2`` is applied at its production value inside the lane,
    so a shared-source fixture silently loses the third document again."""
    from scripts import validate_phase10a as validator

    monkeypatch.setattr(
        validator,
        "_doc_sources",
        lambda run_id: ["validate-phase10a-regression"]
        * validator.EXPECTED_DOCUMENTS,
    )

    with pytest.raises(validator.ValidatorFailure) as excinfo:
        await validator._run_deterministic(
            _db_url(tmp_path, "det-single-source.db"), "0f0f0f0f0f0f0f0f"
        )
    assert excinfo.value.lane == "deterministic"
    assert excinfo.value.detail["check"] in {
        "seeded_sources_distinct",
        "hybrid_executed",
    }
    _assert_disposable_db_empty(_db_url(tmp_path, "det-single-source.db"))


@pytest.mark.asyncio
async def test_live_lane_persists_through_production_ingest_path(
    monkeypatch, tmp_path
):
    """The live document must go through the production ingest_text path with
    a REAL (non-disabled) extractor, then verify from persisted SQL rows."""
    from scripts import validate_phase10a as validator

    calls = {}
    real_ingest = validator.ingest_text

    async def recording_ingest(**kwargs):
        calls["extractor_type"] = type(kwargs["graph_extractor"]).__name__
        calls["title"] = kwargs["title"]
        calls["tags"] = kwargs["tags"]
        return await real_ingest(**kwargs)

    monkeypatch.setattr(validator, "ingest_text", recording_ingest)
    monkeypatch.setattr(
        OllamaGraphExtractor, "extract", _grounded_extract
    )

    summary, restored = await validator._run_live(
        _db_url(tmp_path, "live.db"), "0123456789abcdef"
    )

    assert summary == PINNED_LIVE
    assert restored is True
    assert calls["extractor_type"] == "OllamaGraphExtractor"
    assert "0123456789abcdef" in calls["title"]
    _assert_disposable_db_empty(_db_url(tmp_path, "live.db"))


@pytest.mark.asyncio
async def test_live_lane_provider_unavailable_raises(monkeypatch, tmp_path):
    from scripts import validate_phase10a as validator

    async def unavailable(self, text):
        raise GraphProviderUnavailable("connection refused")

    monkeypatch.setattr(OllamaGraphExtractor, "extract", unavailable)

    with pytest.raises(validator.ValidatorFailure) as excinfo:
        await validator._run_live(
            _db_url(tmp_path, "live-down.db"), "ffffffffffffffff"
        )
    assert excinfo.value.code == "provider_unavailable"
    assert excinfo.value.lane == "live"
    _assert_disposable_db_empty(_db_url(tmp_path, "live-down.db"))


@pytest.mark.asyncio
async def test_live_lane_rejects_canned_output(monkeypatch, tmp_path):
    """A provider echoing a CANNED, token-free relation (schema-valid and
    even grounded as a substring) must fail the non_canned check rather than
    pass the live lane."""
    from app.services import graph_extraction as ge
    from scripts import validate_phase10a as validator

    async def canned(self, text):
        start = text.index("Project Helios")
        return [
            ge.ExtractedRelation(
                source=ge.ExtractedEntity(
                    name="Aria", canonical_name="aria", entity_type="person"
                ),
                predicate="manages",
                target=ge.ExtractedEntity(
                    name="Project Helios",
                    canonical_name="project helios",
                    entity_type="project",
                ),
                evidence="Project Helios",
                evidence_start=start,
                evidence_end=start + len("Project Helios"),
                confidence=0.9,
            )
        ]

    monkeypatch.setattr(OllamaGraphExtractor, "extract", canned)

    with pytest.raises(validator.ValidatorFailure) as excinfo:
        await validator._run_live(
            _db_url(tmp_path, "live-canned.db"), "aaaaaaaabbbbbbbb"
        )
    assert excinfo.value.lane == "live"
    assert excinfo.value.detail["check"] == "non_canned"
    _assert_disposable_db_empty(_db_url(tmp_path, "live-canned.db"))


def test_main_fails_nonzero_when_production_fingerprint_changes(
    monkeypatch, tmp_path, capsys
):
    """restored:true must be withheld when configured production state does
    not restore exactly; exit code must be non-zero with a machine-readable
    error."""
    from scripts import validate_phase10a as validator

    _install_permissive_retrieval_policy(monkeypatch)
    monkeypatch.setattr(OllamaGraphExtractor, "extract", _grounded_extract)
    fingerprints = iter(
        [
            {"sql": {"documents": [1, 1]}, "chroma": None},
            {"sql": {"documents": [2, 1]}, "chroma": None},
        ]
    )
    monkeypatch.setattr(
        validator, "_production_fingerprints", lambda: next(fingerprints)
    )

    exit_code = validator.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out.strip() == ""
    error = json.loads(captured.err.strip().splitlines()[-1])
    assert error["lane"] == "restoration"
    assert "check" in error


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def test_sql_fingerprint_missing_database_returns_none(tmp_path):
    from scripts import validate_phase10a as validator

    assert validator._sql_fingerprint(f"sqlite:///{tmp_path}/missing.db") is None


def test_sql_fingerprint_reads_counts_and_max_ids_read_only(tmp_path):
    from scripts import validate_phase10a as validator

    db_path = tmp_path / "prod-like.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT);"
        "CREATE TABLE alembic_version (version_num TEXT);"
        "INSERT INTO documents (id, title) VALUES (5, 'a'), (9, 'b');"
        "INSERT INTO alembic_version VALUES ('x');"
    )
    conn.commit()
    conn.close()
    before_stat = db_path.stat()

    fingerprint = validator._sql_fingerprint(f"sqlite:///{db_path}")

    assert fingerprint == {"documents": [2, 9]}
    assert "alembic_version" not in fingerprint
    # read-only: file mtime/size unchanged
    after_stat = db_path.stat()
    assert (before_stat.st_mtime_ns, before_stat.st_size) == (
        after_stat.st_mtime_ns,
        after_stat.st_size,
    )


def test_chroma_fingerprint_without_configured_host_returns_none():
    from scripts import validate_phase10a as validator

    assert validator._chroma_fingerprint(None, 8000, None) is None


def test_rrf_constant_is_sixty():
    from scripts import validate_phase10a as validator

    # A chunk ranked 1st on the vector side and 2nd on the graph side fuses to
    # 1/(60+1) + 1/(60+2): the plan 10A.6 fusion constant is pinned at 60.
    # Production emits the exact RRF sum (no rounding); the helper mirrors it.
    assert validator._rrf_score({"vector": 1, "graph": 2}) == pytest.approx(
        1 / 61 + 1 / 62
    )
    assert validator.RRF_K == 60
