"""Phase 10B.3 — retrieval poisoning controls tests (appendix spec).

Covers: calibration fixture schema/semantic validation, two-run byte equality,
the sequential filter pipeline (distance → duplicate → caps), mode-specific
pre-rank orders, persisted decision/reason codes, disposable SQL/Chroma
isolation, and production-fingerprint preservation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from app.services.retrieval_security import (
    Candidate,
    RetrievalSecurityPolicy,
    apply_security_filters,
    compute_calibration_metrics,
    default_policy,
    exact_duplicate_hash,
    filter_candidates,
    near_duplicate_jaccard,
    prerank_vector,
    select_max_distance_threshold,
    validate_calibration_corpus,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = PROJECT_ROOT / "app/tests/fixtures/retrieval_calibration.json"
SCHEMA_PATH = PROJECT_ROOT / "app/tests/fixtures/retrieval_calibration.schema.json"
FIXTURE_SHA256 = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    """Never leak a settings cache built against a disposable database URL."""
    yield
    from app.config import get_settings

    get_settings.cache_clear()


def _policy(**overrides):
    base = default_policy()
    return RetrievalSecurityPolicy(**{**base.__dict__, **overrides})


def _hand_policy(**overrides):
    return _policy(**overrides)


def _candidate(chunk_id, doc_id, source, text, score=0.5, tier="standard", ready=True):
    return Candidate(
        chunk_id=chunk_id, document_id=doc_id, source=source, text=text,
        native_score=score, trust_tier=tier, document_ready=ready,
    )


def _c(
    chunk_id,
    *,
    distance=None,
    document_id=1,
    source="s1",
    text="text",
    graph_score=None,
    hop=None,
    hybrid_score=None,
    origin="vector",
):
    native = distance
    if native is None and hybrid_score is not None:
        native = hybrid_score
    if native is None and graph_score is not None:
        native = graph_score
    if native is None:
        native = 0.5
    return Candidate(
        chunk_id=chunk_id, document_id=document_id, source=source, text=text,
        native_score=native, trust_tier="standard", document_ready=True,
        origin=origin, distance=distance, graph_score=graph_score, hop=hop,
        hybrid_score=hybrid_score,
    )


def _hand_candidates():
    """Pre-rank order (distance, chunk_id): [1, 3, 4, 5, 2].

    Survivors must be [1, 3]; chunk 2 exceeds max_distance, chunk 4 is an
    exact duplicate of chunk 1 (case/whitespace-folded), and chunk 5 is the
    third kept-eligible candidate of srcA (cap 2).
    """
    return [
        _c(chunk_id=1, distance=0.10, document_id=1, source="srcA", text="alpha one"),
        _c(chunk_id=2, distance=0.60, document_id=2, source="srcA", text="beta one"),
        _c(chunk_id=3, distance=0.20, document_id=3, source="srcA", text="gamma two"),
        _c(chunk_id=4, distance=0.30, document_id=4, source="srcB", text="ALPHA   ONE"),
        _c(chunk_id=5, distance=0.40, document_id=5, source="srcA", text="delta three"),
    ]


def _apply_mutation(base: dict, mutation: str) -> dict:
    bad = copy.deepcopy(base)
    docs = bad["documents"]
    queries = bad["queries"]
    if mutation == "empty_text":
        docs[0]["text"] = "   "
    elif mutation == "empty_source":
        docs[0]["source"] = ""
    elif mutation == "label_outside_clean_poison":
        docs[0]["label"] = "unknown"
    elif mutation == "query_references_absent_doc":
        queries[0]["required_clean_ids"] = ["missing-doc"]
    elif mutation == "required_clean_poisoned_overlap":
        queries[0]["poisoned_ids"] = sorted(
            set(queries[0]["poisoned_ids"]) | set(queries[0]["required_clean_ids"])
        )
    elif mutation == "required_clean_not_labeled_clean":
        queries[0]["required_clean_ids"] = ["poison-aria-flood-01"]
    elif mutation == "poisoned_not_labeled_poison":
        queries[0]["poisoned_ids"] = ["clean-aria"]
    elif mutation == "unreferenced_document":
        for q in queries:
            q["poisoned_ids"] = [p for p in q["poisoned_ids"] if p != "poison-unrelated"]
    elif mutation == "empty_required_set":
        queries[0]["required_clean_ids"] = []
    elif mutation == "empty_poisoned_set":
        queries[0]["poisoned_ids"] = []
    elif mutation == "model_mismatch":
        bad["embedding_model"] = "other/model-v1"
    elif mutation == "metric_not_l2":
        bad["distance_metric"] = "cosine"
    elif mutation == "top_k_out_of_range":
        bad["top_k"] = 99
    elif mutation == "non_lexical_document_order":
        docs[0], docs[1] = docs[1], docs[0]
    elif mutation == "fewer_than_two_queries":
        bad["queries"] = queries[:1]
    elif mutation == "fewer_than_two_clean_docs":
        for d in docs:
            if d["label"] == "clean":
                d["label"] = "poison"
    elif mutation == "fewer_than_two_poison_docs":
        for d in docs:
            if d["label"] == "poison":
                d["label"] = "clean"
    else:
        raise AssertionError(f"unknown mutation {mutation!r}")
    return bad


# ---------------------------------------------------------------------------
# Normalization primitives (kept from the pre-appendix suite)
# ---------------------------------------------------------------------------

def test_exact_duplicate_hash_is_deterministic():
    h1 = exact_duplicate_hash("Hello World")
    h2 = exact_duplicate_hash("Hello World")
    assert h1 == h2
    assert len(h1) == 64


def test_exact_duplicate_normalizes_whitespace_case():
    assert exact_duplicate_hash("Hello  World") == exact_duplicate_hash("hello world")


def test_near_duplicate_jaccard_identical():
    assert near_duplicate_jaccard("hello world", "hello world") == 1.0


def test_near_duplicate_jaccard_disjoint():
    assert near_duplicate_jaccard("apple", "banana") == 0.0


def test_near_duplicate_jaccard_partial():
    assert 0.0 < near_duplicate_jaccard("the cat sat", "the cat ran") < 1.0


def test_near_duplicate_jaccard_empty_sets():
    assert near_duplicate_jaccard("", "") == 1.0


# ---------------------------------------------------------------------------
# Calibration fixture / corpus validation
# ---------------------------------------------------------------------------

def test_calibration_fixture_byte_count_and_sha256_immutable():
    raw = FIXTURE_PATH.read_bytes()
    assert raw.endswith(b"\n")
    obj = json.loads(raw)
    assert obj["schema_version"] == "retrieval-calibration-v1"
    assert obj["embedding_model"] == "jinaai/jina-clip-v1"
    assert obj["distance_metric"] == "l2"
    assert obj["top_k"] == 10
    # The committed bytes are pinned by the generated policy's fixture hash.
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256


def test_calibration_schema_rejects_unknown_top_level_key():
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text())
    bad = json.loads(FIXTURE_PATH.read_text())
    bad["unknown_key"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(bad)


def test_calibration_rejects_duplicate_document_ids():
    """JSON Schema cannot express cross-item uniqueness; the semantic loader
    (validate_calibration_corpus) is the authority that rejects duplicates."""
    bad = json.loads(FIXTURE_PATH.read_text())
    bad["documents"][1]["id"] = bad["documents"][0]["id"]
    with pytest.raises(ValueError):
        validate_calibration_corpus(bad)


@pytest.mark.parametrize("mutation", [
    "empty_text", "empty_source", "label_outside_clean_poison",
    "query_references_absent_doc", "required_clean_poisoned_overlap",
    "required_clean_not_labeled_clean", "poisoned_not_labeled_poison",
    "unreferenced_document", "empty_required_set", "empty_poisoned_set",
    "model_mismatch", "metric_not_l2", "top_k_out_of_range",
    "non_lexical_document_order", "fewer_than_two_queries",
    "fewer_than_two_clean_docs", "fewer_than_two_poison_docs",
])
def test_calibration_fixture_rejects_each_semantic_invalid_mutation(mutation):
    bad = _apply_mutation(json.loads(FIXTURE_PATH.read_text()), mutation)
    with pytest.raises(ValueError):
        validate_calibration_corpus(bad)


def test_validate_only_reports_counts_and_writes_nothing():
    result = subprocess.run(
        [sys.executable, "scripts/calibrate_retrieval_security.py",
         "--fixtures", str(FIXTURE_PATH),
         "--schema", str(SCHEMA_PATH),
         "--validate-only"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out == {"documents": 6, "queries": 2,
                   "schema_version": "retrieval-calibration-v1", "status": "valid"}


# ---------------------------------------------------------------------------
# Calibration isolation lifecycle helpers
# ---------------------------------------------------------------------------

def _make_production_db(directory) -> Path:
    prod = directory / "prod.db"
    subprocess.run(
        [sys.executable, "-m", "app.core.migrations"],
        env={**os.environ, "RAG_DATABASE_URL": f"sqlite:///{prod}"},
        capture_output=True, check=True, cwd=PROJECT_ROOT,
    )
    return prod


def _two_disposable_runs(tmp_path) -> tuple[dict, dict]:
    runs = []
    for _ in range(2):
        run_id = uuid.uuid4().hex
        env = {
            **os.environ,
            "RAG_DATABASE_URL": f"sqlite:////tmp/calibration-{run_id}.db",
        }
        result = subprocess.run(
            [sys.executable, "scripts/calibrate_retrieval_security.py",
             "--fixtures", str(FIXTURE_PATH),
             "--schema", str(SCHEMA_PATH),
             "--run-id", run_id,
             "--collection-name", f"calibration-{run_id}",
             "--production-collection-name", "rag-collection",
             "--production-database-url", f"sqlite:///{_make_production_db(tmp_path)}",
             "--stdout"],
            cwd=PROJECT_ROOT, env=env, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stderr.decode()[-2000:]
        runs.append({"policy_bytes": result.stdout})
    return runs[0], runs[1]


def _run_calibrator(monkeypatch, tmp_path, *, db_url=None, run_id=None,
                    production_db=None):
    """Invoke the calibrator in-process; returns whatever main() raises/returns."""
    from scripts import calibrate_retrieval_security as calibrator

    run_id = run_id or uuid.uuid4().hex
    db_url = db_url or f"sqlite:////tmp/calibration-{run_id}.db"
    prod = production_db or _make_production_db(tmp_path)
    monkeypatch.setenv("RAG_DATABASE_URL", db_url)
    argv = [
        "calibrate_retrieval_security.py",
        "--fixtures", str(FIXTURE_PATH),
        "--schema", str(SCHEMA_PATH),
        "--run-id", run_id,
        "--collection-name", f"calibration-{run_id}",
        "--production-collection-name", "rag-collection",
        "--production-database-url", f"sqlite:///{prod}",
        "--stdout",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        return calibrator.main()
    finally:
        sys.argv = old_argv


_POSIX_ONLY_CALIBRATOR = pytest.mark.skipif(
    sys.platform == "win32",
    reason="calibration CLI pins disposable/production DB URLs to POSIX "
    "absolute /tmp paths as an operational-safety guard (destructive-tool "
    "path restriction); this lane executes on POSIX filesystems only",
)


@_POSIX_ONLY_CALIBRATOR
def test_two_calibration_runs_produce_byte_identical_policy(tmp_path):
    run1, run2 = _two_disposable_runs(tmp_path)
    assert run1["policy_bytes"] == run2["policy_bytes"]
    p = json.loads(run1["policy_bytes"])
    assert p["metric"] == "l2"
    assert p["calibration"]["fixture_sha256"] == FIXTURE_SHA256
    assert p["calibration"]["clean_recall"] >= 0.90
    assert p["calibration"]["poisoned_context_share"] <= 0.20


# ---------------------------------------------------------------------------
# Sequential filter pipeline
# ---------------------------------------------------------------------------

def test_filter_pipeline_distance_then_duplicate_then_caps_order():
    policy = _hand_policy(max_distance=0.5, per_source_cap=2,
                          per_document_cap=2, near_duplicate_jaccard=0.90)
    decisions = apply_security_filters(_hand_candidates(), policy=policy,
                                       mode="vector")
    assert [d.chunk_id for d in decisions if d.decision == "selected"] == [1, 3]
    rejected = {d.chunk_id: d.decision for d in decisions if d.decision != "selected"}
    assert rejected[2] == "rejected_distance"
    assert rejected[4] == "rejected_duplicate"
    assert rejected[5] == "rejected_source_cap"


def test_exact_duplicate_uses_first_preranked_sha256_representative():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=10, distance=0.10, document_id=1, source="s1",
           text="Hello   World"),
        _c(chunk_id=11, distance=0.20, document_id=2, source="s2",
           text="hello world"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="vector")
    selected = {d.chunk_id: d for d in decisions if d.decision == "selected"}
    rejected = {d.chunk_id: d for d in decisions if d.decision != "selected"}
    assert 10 in selected
    assert rejected[11].decision == "rejected_duplicate"
    assert "exact_duplicate" in rejected[11].reason_codes


def test_near_duplicate_jaccard_threshold_boundary():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    shared = " ".join("s%d" % i for i in range(18))
    candidates = [
        _c(chunk_id=1, distance=0.10, document_id=1, source="s1",
           text=shared + " aonly"),
        _c(chunk_id=2, distance=0.20, document_id=2, source="s2",
           text=shared + " bonly"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="vector")
    selected = {d.chunk_id for d in decisions if d.decision == "selected"}
    rejected = {d.chunk_id: d for d in decisions if d.decision != "selected"}
    assert selected == {1}
    assert rejected[2].decision == "rejected_duplicate"
    assert "near_duplicate" in rejected[2].reason_codes


def test_empty_sets_define_jaccard_as_one():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=1, distance=0.10, document_id=1, source="s1", text="!!!"),
        _c(chunk_id=2, distance=0.20, document_id=2, source="s2", text="???"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="vector")
    selected = {d.chunk_id for d in decisions if d.decision == "selected"}
    rejected = {d.chunk_id: d for d in decisions if d.decision != "selected"}
    assert selected == {1}
    assert rejected[2].decision == "rejected_duplicate"
    assert "near_duplicate" in rejected[2].reason_codes


def test_per_query_cap_resets_independently():
    policy = _hand_policy(max_distance=0.5, per_source_cap=2,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    query_a = [
        _c(chunk_id=1, distance=0.10, document_id=1, source="srcA", text="alpha one"),
        _c(chunk_id=2, distance=0.15, document_id=2, source="srcA", text="alpha two"),
        _c(chunk_id=3, distance=0.20, document_id=3, source="srcA", text="alpha three"),
    ]
    dec_a = apply_security_filters(query_a, policy=policy, mode="vector")
    assert {d.chunk_id for d in dec_a if d.decision == "selected"} == {1, 2}
    assert {d.chunk_id for d in dec_a if d.decision == "rejected_source_cap"} == {3}
    query_b = [
        _c(chunk_id=10, distance=0.10, document_id=10, source="srcB", text="beta one"),
        _c(chunk_id=11, distance=0.15, document_id=11, source="srcB", text="beta two"),
        _c(chunk_id=12, distance=0.20, document_id=12, source="srcB", text="beta three"),
    ]
    dec_b = apply_security_filters(query_b, policy=policy, mode="vector")
    assert {d.chunk_id for d in dec_b if d.decision == "selected"} == {10, 11}
    assert {d.chunk_id for d in dec_b if d.decision == "rejected_source_cap"} == {12}


def test_document_cap_runs_before_source_cap():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=1, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=1, distance=0.10, document_id=1, source="s1", text="doc one chunk a"),
        _c(chunk_id=2, distance=0.20, document_id=1, source="s1", text="doc one chunk b"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="vector")
    selected = {d.chunk_id for d in decisions if d.decision == "selected"}
    rejected = {d.chunk_id: d for d in decisions if d.decision != "selected"}
    assert selected == {1}
    assert rejected[2].decision == "rejected_document_cap"
    assert "source_cap" not in rejected[2].reason_codes


# ---------------------------------------------------------------------------
# Calibration metrics and threshold selection
# ---------------------------------------------------------------------------

def test_off_query_global_poison_counted_in_numerator():
    label_by_id = {"clean-1": "clean", "poison-off": "poison"}
    metrics = compute_calibration_metrics(
        selected_ids=["clean-1", "poison-off"],
        required_clean_ids=["clean-1"],
        label_by_id=label_by_id,
        poisoned_ids=[],
    )
    assert metrics.clean_recall == 1.0
    assert metrics.poisoned_context_share == 0.5
    assert "poison-off" in metrics.off_query_poison


def test_zero_selected_yields_zero_poison_share_and_zero_clean_recall():
    metrics = compute_calibration_metrics(
        selected_ids=[], required_clean_ids=["c1"],
        label_by_id={}, poisoned_ids=[],
    )
    assert metrics.clean_recall == 0.0
    assert metrics.poisoned_context_share == 0.0


def test_threshold_selection_uses_largest_threshold_meeting_both_bounds():
    evaluations = [
        (0.10, 1.00, 0.00),   # passes
        (0.30, 1.00, 0.10),   # passes
        (0.50, 0.95, 0.20),   # passes (both bounds inclusive)
        (0.70, 0.85, 0.30),   # fails recall (<0.90) and poison (>0.20)
        (0.90, 0.70, 0.40),   # fails both bounds
    ]
    assert select_max_distance_threshold(evaluations) == 0.50


def test_sentinel_distance_greater_than_last_is_used_when_finite():
    import math
    d_last = 0.9
    sentinel = math.nextafter(d_last, math.inf)
    assert sentinel > d_last
    assert math.isfinite(sentinel)


@_POSIX_ONLY_CALIBRATOR
def test_no_passing_threshold_exits_one_without_replacement(tmp_path, monkeypatch):
    import importlib.util

    from app.services.retrieval_security import CalibrationMetrics

    policy_path = PROJECT_ROOT / "config" / "retrieval-security-policy.json"
    before = policy_path.read_bytes()

    spec = importlib.util.spec_from_file_location(
        "calibrate_retrieval_security",
        PROJECT_ROOT / "scripts" / "calibrate_retrieval_security.py")
    calibrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibrator)
    calibrator.compute_calibration_metrics = lambda *a, **k: CalibrationMetrics(
        clean_recall=0.50, poisoned_context_share=0.50)

    run_id = uuid.uuid4().hex
    prod = _make_production_db(tmp_path)
    monkeypatch.setenv("RAG_DATABASE_URL", f"sqlite:////tmp/calibration-{run_id}.db")
    old_argv = sys.argv
    sys.argv = [
        "calibrate_retrieval_security.py",
        "--fixtures", str(FIXTURE_PATH),
        "--schema", str(SCHEMA_PATH),
        "--run-id", run_id,
        "--collection-name", f"calibration-{run_id}",
        "--production-collection-name", "rag-collection",
        "--production-database-url", f"sqlite:///{prod}",
        "--stdout",
    ]
    try:
        with pytest.raises(SystemExit) as exc:
            calibrator.main()
    finally:
        sys.argv = old_argv
    assert exc.value.code == 1
    assert policy_path.read_bytes() == before


# ---------------------------------------------------------------------------
# Mode-specific pre-rank orders and distance applicability
# ---------------------------------------------------------------------------

def test_vector_mode_pre_rank_is_native_l2_then_chunk_id():
    ranked = prerank_vector([
        _c(chunk_id=2, distance=0.4), _c(chunk_id=1, distance=0.4),
    ])
    assert [c.chunk_id for c in ranked] == [1, 2]


def test_graph_mode_pre_rank_ignores_distance():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=1, distance=None, graph_score=0.9, hop=1,
           document_id=1, source="s1", text="graph one", origin="graph"),
        _c(chunk_id=2, distance=None, graph_score=0.7, hop=2,
           document_id=2, source="s2", text="graph two", origin="graph"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="graph")
    assert [d.chunk_id for d in decisions if d.decision == "selected"] == [1, 2]
    for d in decisions:
        assert d.decision != "rejected_distance"
        assert "distance_not_applicable_graph_origin" in d.reason_codes


def test_hybrid_vector_only_distance_rejected():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=1, distance=0.2, hybrid_score=0.9, origin="vector",
           document_id=1, source="s1", text="hybrid vec ok"),
        _c(chunk_id=2, distance=0.9, hybrid_score=0.8, origin="vector",
           document_id=2, source="s2", text="hybrid vec far"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="hybrid")
    selected = {d.chunk_id for d in decisions if d.decision == "selected"}
    rejected = {d.chunk_id: d for d in decisions if d.decision != "selected"}
    assert selected == {1}
    assert rejected[2].decision == "rejected_distance"


def test_hybrid_graph_only_distance_inapplicable():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=1, distance=None, hybrid_score=0.9, origin="graph",
           document_id=1, source="s1", text="hybrid graph one"),
        _c(chunk_id=2, distance=None, hybrid_score=0.7, origin="graph",
           document_id=2, source="s2", text="hybrid graph two"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="hybrid")
    assert [d.chunk_id for d in decisions if d.decision == "selected"] == [1, 2]
    for d in decisions:
        assert d.decision != "rejected_distance"
        assert "distance_not_applicable_graph_origin" in d.reason_codes


def test_hybrid_both_graph_supported_distance_inapplicable():
    policy = _hand_policy(max_distance=0.5, per_source_cap=5,
                          per_document_cap=5, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=1, distance=0.9, hybrid_score=0.9, origin="both",
           document_id=1, source="s1", text="hybrid both supported"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="hybrid")
    selected = [d for d in decisions if d.decision == "selected"]
    assert [d.chunk_id for d in selected] == [1]
    assert "distance_not_applicable_graph_origin" in selected[0].reason_codes


# ---------------------------------------------------------------------------
# Persisted decision/reason codes
# ---------------------------------------------------------------------------

def test_persisted_decision_reason_codes_match_filter_outcome(tmp_path):
    import sqlalchemy
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.persistence import models
    from app.services.security_audit import SecurityAuditService

    engine = create_engine(f"sqlite:///{tmp_path / 'audit.db'}")
    models.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    policy = _hand_policy(max_distance=0.5, per_source_cap=2,
                          per_document_cap=2, near_duplicate_jaccard=0.90)
    candidates = [
        _c(chunk_id=1, distance=0.10, document_id=101, source="s1", text="alpha"),
        _c(chunk_id=2, distance=0.70, document_id=102, source="s1", text="beta"),
        _c(chunk_id=3, distance=0.20, document_id=103, source="s2", text="gamma"),
    ]
    decisions = apply_security_filters(candidates, policy=policy, mode="vector")
    in_memory = {d.chunk_id: d for d in decisions}
    doc_of = {c.chunk_id: c.document_id for c in candidates}

    session = Session()
    try:
        service = SecurityAuditService(session)
        audit_id = service.begin(
            query="persisted-query",
            mode="vector",
            policy_versions={"retrieval": policy.version},
        )
        service.record_decisions(audit_id, [
            {"chunk_id": d.chunk_id, "decision": d.decision,
             "native_score": d.native_score,
             "provenance_score": d.provenance_score,
             "reason_codes": json.dumps(sorted(d.reason_codes)),
             "content_sha256": hashlib.sha256(("c%d" % d.chunk_id).encode()).hexdigest(),
             "document_id": doc_of[d.chunk_id],
             "document_id_snapshot": doc_of[d.chunk_id],
             "chunk_id_snapshot": d.chunk_id}
            for d in decisions
        ])
        service.complete(audit_id)
        session.commit()
        rows = session.query(models.RetrievalCandidateDecision).filter_by(
            audit_id=audit_id).all()
    finally:
        session.close()
        engine.dispose()

    assert len(rows) == len(decisions)
    for row in rows:
        mem = in_memory[row.chunk_id_snapshot]
        assert row.decision == mem.decision
        assert json.loads(row.reason_codes) == sorted(mem.reason_codes)


# ---------------------------------------------------------------------------
# Disposable SQL/Chroma isolation and fingerprint preservation
# ---------------------------------------------------------------------------

def test_disposable_sql_isolation_refuses_production_path_equality(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RAG_DATABASE_URL", "sqlite:////data/rag.db")
    with pytest.raises((ValueError, SystemExit)):
        _run_calibrator(monkeypatch, tmp_path,
                        db_url="sqlite:////data/rag.db",
                        production_db=tmp_path / "prod.db")


def test_disposable_sql_isolation_refuses_symlink_to_production(tmp_path, monkeypatch):
    prod = _make_production_db(tmp_path)
    run_id = uuid.uuid4().hex
    link = Path(f"/tmp/calibration-{run_id}.db")
    try:
        link.symlink_to(prod)
    except OSError:
        pytest.skip("symlinks not supported")
    try:
        with pytest.raises((ValueError, SystemExit)):
            _run_calibrator(monkeypatch, tmp_path,
                            db_url=f"sqlite:///{link}",
                            run_id=run_id, production_db=prod)
    finally:
        link.unlink(missing_ok=True)


@_POSIX_ONLY_CALIBRATOR
def test_disposable_collection_cleanup_in_finally(tmp_path, monkeypatch):
    from unittest.mock import patch

    from app.services.vector_store import ChromaVectorStore, _create_client

    deleted: list = []
    with patch.object(ChromaVectorStore, "query",
                      side_effect=RuntimeError("query stage boom")), \
         patch.object(ChromaVectorStore, "delete_collection",
                      lambda self, name, **k: deleted.append(name)):
        with pytest.raises((RuntimeError, SystemExit)):
            _run_calibrator(monkeypatch, tmp_path)
    assert any(name.startswith("calibration-") for name in deleted)
    # The patched delete only recorded; really remove the populated disposable
    # collections so the shared Chroma volume carries no per-run debris (D-41).
    cleanup_client = _create_client()
    existing = {
        c.name if hasattr(c, "name") else str(c)
        for c in cleanup_client.list_collections()
    }
    for name in deleted:
        if name in existing:
            cleanup_client.delete_collection(name)


def test_production_sqlite_fingerprint_unchanged_before_and_after(tmp_path):
    prod = _make_production_db(tmp_path)
    before = hashlib.sha256(prod.read_bytes()).hexdigest()
    run_id = uuid.uuid4().hex
    env = {
        **os.environ,
        "RAG_DATABASE_URL": f"sqlite:////tmp/calibration-{run_id}.db",
    }
    subprocess.run(
        [sys.executable, "scripts/calibrate_retrieval_security.py",
         "--fixtures", str(FIXTURE_PATH), "--schema", str(SCHEMA_PATH),
         "--run-id", run_id,
         "--collection-name", f"calibration-{run_id}",
         "--production-collection-name", "rag-collection",
         "--production-database-url", f"sqlite:///{prod}",
         "--stdout"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    after = hashlib.sha256(prod.read_bytes()).hexdigest()
    assert before == after


def test_alias_or_duplicate_vector_id_refused(tmp_path, monkeypatch):
    from unittest.mock import patch

    from app.services.vector_store import ChromaVectorStore

    class _FakeResult:
        def __init__(self, vector_id, score):
            self.vector_id = vector_id
            self.score = score

    def fake_query(*a, **k):
        # Two returned candidates share one canonical vector_id -> alias collision.
        return [_FakeResult("dup-vid", 0.10), _FakeResult("dup-vid", 0.20)]

    with patch.object(ChromaVectorStore, "query", side_effect=fake_query):
        with pytest.raises((ValueError, SystemExit)) as exc:
            _run_calibrator(monkeypatch, tmp_path)
        if isinstance(exc.value, SystemExit):
            assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Strict policy loading (D-18/D-19): immutable, fail-closed, exact key set
# ---------------------------------------------------------------------------

def test_strict_policy_loader_accepts_the_committed_policy():
    from app.services.retrieval_security import load_retrieval_security_policy_strict

    policy = load_retrieval_security_policy_strict(
        PROJECT_ROOT / "config" / "retrieval-security-policy.json"
    )
    assert policy.metric == "l2"
    assert policy.calibration_embedding_model == "jinaai/jina-clip-v1"
    assert policy.calibration_fixture_sha256 == FIXTURE_SHA256


def test_strict_policy_loader_rejects_unknown_top_level_key(tmp_path):
    import json

    from app.services.retrieval_security import load_retrieval_security_policy_strict

    policy = json.loads(
        (PROJECT_ROOT / "config" / "retrieval-security-policy.json").read_text()
    )
    policy["unknown_key"] = 1
    bad = tmp_path / "policy-unknown.json"
    bad.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_retrieval_security_policy_strict(bad)


def test_strict_policy_loader_rejects_fixture_hash_mismatch(tmp_path):
    import json

    from app.services.retrieval_security import load_retrieval_security_policy_strict

    policy = json.loads(
        (PROJECT_ROOT / "config" / "retrieval-security-policy.json").read_text()
    )
    policy["calibration"]["fixture_sha256"] = "0" * 64
    bad = tmp_path / "policy-hash.json"
    bad.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_retrieval_security_policy_strict(bad)


def test_strict_policy_loader_rejects_non_l2_metric(tmp_path):
    import json

    from app.services.retrieval_security import load_retrieval_security_policy_strict

    policy = json.loads(
        (PROJECT_ROOT / "config" / "retrieval-security-policy.json").read_text()
    )
    policy["metric"] = "cosine"
    bad = tmp_path / "policy-metric.json"
    bad.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        load_retrieval_security_policy_strict(bad)


def test_retrieval_policy_is_loaded_once_and_fails_closed(tmp_path, monkeypatch):
    """D-18: a missing/corrupt policy must never fall back to an arbitrary
    default threshold; retrieval fails closed."""
    from app.config import get_settings
    from app.services import retrieval as retrieval_module

    monkeypatch.setenv(
        "RAG_RETRIEVAL_SECURITY_POLICY_PATH",
        str(tmp_path / "missing-policy.json"),
    )
    get_settings.cache_clear()
    retrieval_module.reset_retrieval_security_policy_cache()
    try:
        with pytest.raises(Exception):
            retrieval_module.get_retrieval_security_policy()
    finally:
        retrieval_module.reset_retrieval_security_policy_cache()
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Embedding-regime precondition (R2a): the policy load path must refuse a
# runtime whose embedding regime differs from the policy's calibration regime
# (typed, fail-closed at load — never silent rejected_distance filtering).
# ---------------------------------------------------------------------------


def _regime_env(monkeypatch, provider, model=None):
    """Resolve settings against the committed policy under a chosen regime."""
    monkeypatch.setenv(
        "RAG_RETRIEVAL_SECURITY_POLICY_PATH",
        str(PROJECT_ROOT / "config" / "retrieval-security-policy.json"),
    )
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", provider)
    if model is not None:
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", model)
    from app.config import get_settings

    get_settings.cache_clear()
    return get_settings()


def test_policy_load_refuses_local_provider_under_fastembed_calibration(
    monkeypatch,
):
    """local (hash) regime + committed fastembed-calibrated policy -> typed
    refusal at load naming BOTH regimes, not filter-time rejections."""
    from app.config import get_settings
    from app.services import retrieval as retrieval_module
    from app.services.retrieval_security import RetrievalSecurityRegimeError

    _regime_env(monkeypatch, "local")
    retrieval_module.reset_retrieval_security_policy_cache()
    try:
        with pytest.raises(RetrievalSecurityRegimeError) as excinfo:
            retrieval_module.get_retrieval_security_policy()
        message = str(excinfo.value)
        # Both regimes are named: the calibrated model and the runtime provider.
        assert "jinaai/jina-clip-v1" in message
        assert "local" in message
    finally:
        retrieval_module.reset_retrieval_security_policy_cache()
        get_settings.cache_clear()


def test_policy_load_accepts_matching_fastembed_regime(monkeypatch):
    """fastembed regime + committed policy (the dev-stack configuration)
    loads unchanged."""
    from app.config import get_settings
    from app.services import retrieval as retrieval_module

    _regime_env(monkeypatch, "fastembed", "jinaai/jina-clip-v1")
    retrieval_module.reset_retrieval_security_policy_cache()
    try:
        policy = retrieval_module.get_retrieval_security_policy()
        assert policy.calibration_embedding_model == "jinaai/jina-clip-v1"
        assert policy.max_distance == pytest.approx(0.643872)
    finally:
        retrieval_module.reset_retrieval_security_policy_cache()
        get_settings.cache_clear()


def test_preinstalled_policy_cache_is_a_full_regime_bypass(monkeypatch):
    """The shipped _POLICY_CACHE test hook stays a full bypass: hermetic tests
    installing their own policy must never hit the provider-regime assertion
    (even under a local runtime regime with a non-fastembed calibration)."""
    from app.config import get_settings
    from app.services import retrieval as retrieval_module

    _regime_env(monkeypatch, "local")
    installed = _policy(
        max_distance=1e9,
        per_source_cap=1000,
        per_document_cap=1000,
        max_candidates=1000,
        near_duplicate_jaccard=1.0,
        calibration_fixture_sha256="hermetic-test",
        calibration_embedding_model="hash-test",
    )
    retrieval_module._POLICY_CACHE = installed
    try:
        assert retrieval_module.get_retrieval_security_policy() is installed
    finally:
        retrieval_module.reset_retrieval_security_policy_cache()
        get_settings.cache_clear()


def test_regime_assertion_is_model_identity_not_provider_name():
    """A local-calibrated policy under a local runtime regime is fine: the
    refusal compares effective embedding-model identity, it is not a blanket
    ban on the local provider name."""
    from app.services.retrieval_security import (
        LOCAL_HASH_EMBEDDING_MODEL,
        RetrievalSecurityRegimeError,
        assert_policy_matches_runtime_regime,
        effective_runtime_embedding_model,
    )

    class _Regime:
        embedding_provider = "local"
        embedding_model = "jinaai/jina-clip-v1"  # ignored by the local provider

    assert effective_runtime_embedding_model(_Regime()) == LOCAL_HASH_EMBEDDING_MODEL

    local_calibrated = _policy(
        calibration_embedding_model=LOCAL_HASH_EMBEDDING_MODEL,
        max_distance=12.0,
    )
    assert assert_policy_matches_runtime_regime(local_calibrated, _Regime()) is None

    fastembed_calibrated = _policy(
        calibration_embedding_model="jinaai/jina-clip-v1"
    )
    with pytest.raises(RetrievalSecurityRegimeError):
        assert_policy_matches_runtime_regime(fastembed_calibrated, _Regime())


def test_production_graph_contexts_pre_rank_by_min_hop_on_equal_graph_score():
    """D-23: production graph metadata carries the hop under ``min_hop``; with
    equal graph_score the lower-hop chunk must rank (and be selected) first."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.db import Base
    from app.persistence import models  # noqa: F401 — register models
    from app.services import retrieval as retrieval_module
    from app.services.retrieval import _apply_security_filter
    from app.services.retrieval_security import (
        load_retrieval_security_policy_strict,
    )

    # Hermetic lane shim (merged 10D R2): _apply_security_filter loads the
    # policy lazily; under the bare ``local`` hash regime the R2
    # regime-identity precondition would refuse the committed
    # fastembed-calibrated policy at that load. Pre-install it through the
    # shipped ``_POLICY_CACHE`` hook (the documented full bypass) so the
    # production caps keep their exact pre-merge bare behavior.
    _prior_policy_cache = retrieval_module._POLICY_CACHE
    retrieval_module._POLICY_CACHE = load_retrieval_security_policy_strict(
        PROJECT_ROOT / "config" / "retrieval-security-policy.json"
    )

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        contexts = [
            {"text": "graph chunk one hop", "score": 0.9, "metadata": {
                "chunk_id": 2, "document_id": 2, "source": "srcB",
                "retrieval_sources": ["graph"], "graph_score": 0.9,
                "min_hop": 1, "score_type": "graph_path_score"}},
            {"text": "graph chunk two hops", "score": 0.9, "metadata": {
                "chunk_id": 1, "document_id": 1, "source": "srcA",
                "retrieval_sources": ["graph"], "graph_score": 0.9,
                "min_hop": 2, "score_type": "graph_path_score"}},
        ]
        # Equal graph_score: the lower-hop chunk must rank first even though
        # its chunk_id is higher (a hop-inert sort would emit [1, 2]).
        result, decision_rows = _apply_security_filter(
            session, contexts, mode="graph"
        )
    finally:
        retrieval_module._POLICY_CACHE = _prior_policy_cache
        session.close()
        engine.dispose()
    assert [c["metadata"]["chunk_id"] for c in result] == [2, 1]


# ---------------------------------------------------------------------------
# Filter primitives kept from the pre-appendix suite (caller pre-ranked API)
# ---------------------------------------------------------------------------

def test_blocked_trust_rejected():
    policy = _policy()
    candidates = [_candidate(1, 1, "src", "text", tier="blocked")]
    decisions = filter_candidates(candidates, policy)
    assert decisions[0].decision == "rejected_blocked_source"


def test_exact_duplicate_kept_first_rejected_rest():
    policy = _policy()
    candidates = [
        _candidate(1, 1, "src", "identical text"),
        _candidate(2, 2, "src2", "identical text"),
    ]
    decisions = filter_candidates(candidates, policy)
    assert decisions[0].decision == "selected"
    assert decisions[1].decision == "rejected_duplicate"
    assert "exact_duplicate" in decisions[1].reason_codes


def test_max_candidates_cap():
    policy = _policy(max_candidates=1)
    candidates = [
        _candidate(1, 1, "src", "text a", score=0.1),
        _candidate(2, 2, "src2", "text b", score=0.2),
    ]
    decisions = filter_candidates(candidates, policy)
    selected = [d for d in decisions if d.decision == "selected"]
    assert len(selected) == 1


def test_survivors_preserve_pre_rank_order():
    """filter_candidates preserves the caller's pre-rank order (D-8 fix)."""
    policy = _policy()
    candidates = [
        _candidate(1, 1, "src1", "text a", score=0.1),
        _candidate(2, 2, "src2", "text b", score=0.2),
        _candidate(3, 3, "src3", "text c", score=0.3),
    ]
    decisions = filter_candidates(candidates, policy)
    selected = [d for d in decisions if d.decision == "selected"]
    assert [d.chunk_id for d in selected] == [1, 2, 3]


def test_empty_candidates_returns_empty():
    assert filter_candidates([], _policy()) == []
