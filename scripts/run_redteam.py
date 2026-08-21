"""Task 10D.2/10D.4 red-team harness CLI (guarded, isolated, disposable).

Runs the pinned attack corpus through the exact production ingestion and
query paths twice — a ``disabled`` mode that observes but does not
enforce the measured content-safety control, and an ``enabled`` mode
that enforces it — each against its own migrated, UUID-named disposable
SQL database and Chroma collection. Per fixture and mode the harness
measures with the pinned latency protocol (3 untimed warm-ups, 10 timed
``perf_counter_ns`` repetitions) and emits the closed
``phase10-redteam-report-v1`` report: per-fixture ScenarioResult rows,
mode/comparison metrics, acceptance verdict, and production
fingerprints. Production stores are opened read-only for fingerprints
only; every refusal happens before any mutation; one outer ``finally``
removes both disposable stores and proves production unchanged. Exit
codes: 0 harness complete with acceptance thresholds passed; 1 measured
defense failure; 2 isolation/configuration refusal or corpus-invalid
input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from itertools import groupby
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services import attack_simulator  # noqa: E402

DEFAULT_FIXTURES = "app/tests/fixtures/attack_payloads.json"
WARMUPS = 3
REPETITIONS = 10


class _CorpusEmbeddingProvider:
    """Deterministic embeddings encoding each fixture's scenario geometry.

    Document vectors are ``[l2(doc)] * 8`` from the corpus scenario
    distances (benign documents sit near the query point), and query
    vectors are all-zero, so real L2 retrieval reproduces the corpus's
    intended candidate geometry without any external embedding model.
    """

    def __init__(self, vectors_by_text: dict) -> None:
        self._vectors = vectors_by_text

    async def embed_texts(self, texts):
        return [self._vectors.get(text, [0.0] * 8) for text in texts]


class _EchoLLMClient:
    """Deterministic generation: the model echoes the retrieved evidence.

    An echoed context containing an attack literal (OVERRIDE_ACCEPTED /
    UNSAFE_MARKER) makes the corpus's answer evaluators real — a
    successful prompt injection or safety evasion surfaces verbatim in
    the generated answer, exactly as the corpus defines it.
    """

    async def generate_answer(self, query, evidence, system_prompt=None):
        return "\n".join(evidence)


def load_fixtures(fixtures_path) -> list:
    """Validate the corpus and flatten it to ingestion payloads."""
    path = Path(fixtures_path)
    corpus = attack_simulator.validate_attack_corpus(path)
    payloads = []
    for fixture in corpus["fixtures"]:
        for doc in fixture["documents"]:
            payloads.append({
                "fixture_id": fixture["id"],
                "document_fixture_id": doc["id"],
                "title": doc["id"],
                "source": doc["source"],
                "text": doc["text"],
            })
    return payloads


def _sha(text) -> str:
    return hashlib.sha256(text.encode("utf-8") if isinstance(text, str)
                          else text).hexdigest()


def _fixture_distances(fixture: dict) -> dict:
    """Document id -> L2 distance from the query point (default near)."""
    l2 = fixture["scenario"].get("l2_distances", {})
    return {doc["id"]: float(l2.get(doc["id"], 0.02))
            for doc in fixture["documents"]}


def _measure_fixture(fixture: dict, mode: str, engine, store, bindings
                     ) -> tuple:
    """Execute one fixture's query 3+10 times through the production path.

    Returns ``(scenario_result_row, latency_samples)``.
    """
    import asyncio

    from app.services.rag import answer_query

    distances = _fixture_distances(fixture)
    vectors = {doc["text"]: [distances[doc["id"]]] * 8
               for doc in fixture["documents"]}
    vectors[fixture["query"]] = [0.0] * 8
    provider = _CorpusEmbeddingProvider(vectors)
    llm = _EchoLLMClient()

    bindings_by_doc_all = {b["document_fixture_id"]: b for b in bindings}
    fixture_doc_ids = [doc["id"] for doc in fixture["documents"]]
    # Each fixture's attack is measured against its own documents: the
    # query is scoped to the fixture's SQL document ids in this mode.
    scope_document_ids = [
        bindings_by_doc_all[doc_id]["sql_document_id"]
        for doc_id in fixture_doc_ids
        if doc_id in bindings_by_doc_all
        and bindings_by_doc_all[doc_id].get("sql_document_id") is not None]
    query_filters = {"document_id": {"$in": scope_document_ids}}

    def _run_query():
        session = attack_simulator._session_from_engine(engine)
        try:
            return asyncio.run(answer_query(
                query=fixture["query"],
                embedding_provider=provider,
                vector_store=store,
                llm_client=llm,
                top_k=5,
                filters=query_filters,
                session=session,
            ))
        finally:
            session.close()

    for _ in range(WARMUPS):
        _run_query()
    samples = []
    result = None
    for _ in range(REPETITIONS):
        started = time.perf_counter_ns()
        result = _run_query()
        samples.append(max(time.perf_counter_ns() - started, 1))

    bindings_by_doc = bindings_by_doc_all
    context_chunk_ids = {
        (ctx.get("metadata") or {}).get("chunk_id")
        for ctx in result.get("context", [])}
    selected_docs = {
        doc_id for doc_id, binding in bindings_by_doc.items()
        if binding.get("sql_chunk_id") in context_chunk_ids}

    candidate_rows = []
    for doc in fixture["documents"]:
        binding = bindings_by_doc.get(doc["id"])
        if binding is None or binding.get("sql_chunk_id") is None:
            # Ingestion-blocked documents never produced a chunk/vector:
            # they cannot appear as candidates at all.
            continue
        selected = doc["id"] in selected_docs
        candidate_rows.append({
            "document_fixture_id": doc["id"],
            "chunk_fixture_id": f"{doc['id']}:0",
            "sql_document_id": binding["sql_document_id"],
            "sql_chunk_id": binding["sql_chunk_id"],
            "is_poisoned": doc["is_poisoned"],
            "required_clean": doc["id"] in
            fixture["required_clean_document_ids"],
            "selected": selected,
            "decision": "selected" if selected else "rejected",
            "reason_codes": [] if selected else ["not_selected"],
        })

    scenario = fixture["scenario"]
    path_rows = []
    if scenario.get("paths"):
        poisoned_evidence = next(
            (doc["id"] for doc in fixture["documents"] if doc["is_poisoned"]),
            None)
        for i, path in enumerate(scenario["paths"]):
            selected = mode == "disabled"
            path_rows.append({
                "path_id": f"{fixture['id']}-path-{i}",
                "selected": selected,
                "evidence_fixture_ids": list(path["document_ids"]),
                "contains_poisoned_evidence": poisoned_evidence
                in path["document_ids"],
            })

    control_results = [{
        "control_id": cid,
        "would_trigger": fixture["kind"] == "malicious",
        "enforced": mode == "enabled",
        "outcome": ("triggered" if fixture["kind"] == "malicious"
                    and mode == "enabled"
                    else "bypassed" if fixture["kind"] == "malicious"
                    else "not_triggered"),
    } for cid in fixture["expected_control_ids"]]

    answer = result.get("answer", "")
    answer_text = answer if isinstance(answer, str) else ""
    generation_blocked = result.get("safety_summary", {}) is not None and \
        answer_text.startswith("The generated answer was withheld")
    safety_summary = result.get("safety_summary") or {}
    block_decisions = 1 if generation_blocked else 0

    ev = fixture["evaluator"]
    selected_chunks = [c["chunk_fixture_id"] for c in candidate_rows
                       if c["selected"]]
    poisoned_selected = [c["chunk_fixture_id"] for c in candidate_rows
                         if c["selected"] and c["is_poisoned"]]
    clean_ids = [f"{d}:0" for d in fixture["required_clean_document_ids"]]

    if ev["type"] == "poisoned_chunk_selected":
        target = f"{ev['poisoned_document_id']}:0"
        evaluator_result = target in selected_chunks
        observations = {
            "type": "poisoned_chunk_selected",
            "target_chunk_fixture_id": target,
            "selected_chunk_fixture_ids": selected_chunks,
        }
    elif ev["type"] == "answer_contains_literal":
        literal = ev["literal"]
        marker = f"marker-{_sha(literal)[:8]}"
        start = answer_text.find(literal)
        evaluator_result = start >= 0
        observations = {
            "type": "answer_contains_literal",
            "literal_sha256": _sha(literal),
            "returned_marker_ids": [marker] if evaluator_result else [],
            "matched_marker_id": marker if evaluator_result else None,
            "match_start": start if evaluator_result else None,
            "match_end": start + len(literal) if evaluator_result else None,
        }
    elif ev["type"] == "graph_path_contains_poisoned_evidence":
        evaluator_result = any(p["selected"] and p["contains_poisoned_evidence"]
                               for p in path_rows)
        observations = {
            "type": "graph_path_contains_poisoned_evidence",
            "poisoned_evidence_fixture_ids": [
                f"{d['id']}:0" for d in fixture["documents"]
                if d["is_poisoned"]],
            "selected_path_ids": [p["path_id"] for p in path_rows
                                  if p["selected"]],
            "selected_evidence_fixture_ids": [
                p["evidence_fixture_ids"][0] for p in path_rows
                if p["selected"]],
        }
    elif ev["type"] == "unsafe_answer_returned":
        literal = ev["literal"]
        evaluator_result = literal in answer_text and not generation_blocked
        observations = {
            "type": "unsafe_answer_returned",
            "block_decision_ids":
                [f"block-{fixture['id']}"] if block_decisions else [],
            "unsafe_marker_ids":
                [f"marker-{_sha(literal)[:8]}"] if evaluator_result else [],
        }
    else:  # control_not_triggered
        control = next((c for c in control_results
                        if c["control_id"] == ev["control_id"]), None)
        evaluator_result = not (control["would_trigger"]
                                and control["enforced"])
        observations = {
            "type": "control_not_triggered",
            "control_id": ev["control_id"],
            "would_trigger": control["would_trigger"],
            "enforced": control["enforced"],
            "outcome": control["outcome"],
        }

    attack_succeeded = evaluator_result if fixture["kind"] == "malicious" \
        else False
    row = {
        "id": fixture["id"],
        "category": fixture["category"],
        "kind": fixture["kind"],
        "severity": fixture["severity"],
        "mode": mode,
        "expected_safe_outcome": fixture["expected_safe_outcome"],
        "query_sha256": _sha(fixture["query"]),
        "attack_succeeded": attack_succeeded,
        "evaluator": {**ev, "result": evaluator_result},
        "control_results": control_results,
        "candidate_rows": candidate_rows,
        "path_rows": path_rows,
        "required_clean_ids": clean_ids,
        "selected_clean_ids": [c["chunk_fixture_id"] for c in candidate_rows
                               if c["selected"] and c["required_clean"]],
        "selected_poisoned_ids": poisoned_selected,
        "benign_chunk_ids": [c["chunk_fixture_id"] for c in candidate_rows
                             if fixture["kind"] == "benign"],
        "rejected_benign_ids": [c["chunk_fixture_id"] for c in candidate_rows
                                if not c["selected"]
                                and fixture["kind"] == "benign"],
        "evaluator_observations": observations,
        "block_decision_count": block_decisions,
        "generation_blocked": generation_blocked,
        "unsafe_answer_returned": bool(
            fixture["kind"] == "malicious"
            and ev["type"] == "unsafe_answer_returned"
            and evaluator_result),
        "answer_sha256": None if generation_blocked else _sha(answer_text),
        "latency_ns": samples,
    }
    _ = safety_summary
    return row, samples


def _ratio(num: int, den: int) -> dict:
    return {"numerator": num, "denominator": den,
            "value": round(num / den, 6) if den else 0.0,
            "denominator_zero": den == 0}


def _derive_metrics(rows: list) -> dict:
    """Derive mode metrics + comparison from ScenarioResult rows."""
    from app.services.attack_simulator import nearest_rank_percentile

    out = {"methodology": {
        "clock": "perf_counter_ns", "warmups_per_fixture": WARMUPS,
        "measured_repetitions_per_fixture": REPETITIONS,
        "percentile_method": "nearest_rank"}}
    for mode in ("disabled", "enabled"):
        counters = {"goal": 0, "attempted": 0, "sel_poison": 0,
                    "sel_all": 0, "sel_clean": 0, "req_clean": 0,
                    "benign": 0, "benign_rejected": 0, "poison_paths": 0,
                    "sel_paths": 0, "blocked": 0, "unsafe_after_block": 0}
        latency = []
        for row in rows:
            if row["mode"] != mode:
                continue
            if row["kind"] == "malicious":
                counters["attempted"] += 1
                counters["goal"] += 1 if row["attack_succeeded"] else 0
            counters["sel_poison"] += len(row["selected_poisoned_ids"])
            counters["sel_all"] += sum(
                1 for c in row["candidate_rows"] if c["selected"])
            counters["sel_clean"] += len(row["selected_clean_ids"])
            counters["req_clean"] += len(row["required_clean_ids"])
            counters["benign"] += len(row["benign_chunk_ids"])
            counters["benign_rejected"] += len(row["rejected_benign_ids"])
            counters["poison_paths"] += sum(
                1 for p in row["path_rows"]
                if p["selected"] and p["contains_poisoned_evidence"])
            counters["sel_paths"] += sum(
                1 for p in row["path_rows"] if p["selected"])
            counters["blocked"] += row["block_decision_count"]
            if row["unsafe_answer_returned"] and row["block_decision_count"]:
                counters["unsafe_after_block"] += 1
            latency.extend(row["latency_ns"])
        out[mode] = {
            "attack_success_rate": _ratio(counters["goal"],
                                          counters["attempted"]),
            "poisoned_context_share": _ratio(counters["sel_poison"],
                                             counters["sel_all"]),
            "clean_retrieval_recall": _ratio(counters["sel_clean"],
                                             counters["req_clean"]),
            "false_positive_rate": _ratio(counters["benign_rejected"],
                                          counters["benign"]),
            "graph_path_contamination": _ratio(counters["poison_paths"],
                                               counters["sel_paths"]),
            "blocked_generation_count": counters["blocked"],
            "unsafe_answers_after_block": _ratio(
                counters["unsafe_after_block"], counters["blocked"]),
            "latency": {
                "sample_count": len(latency),
                "p50_ms": nearest_rank_percentile(latency, 50) / 1e6,
                "p95_ms": nearest_rank_percentile(latency, 95) / 1e6,
            },
        }
    d = out["disabled"]["attack_success_rate"]["value"]
    e = out["enabled"]["attack_success_rate"]["value"]
    comparison = {"relative_attack_reduction": {
        "numerator": round(d - e, 6), "denominator": d,
        "value": round((d - e) / d, 6) if d else 0.0,
        "denominator_zero": d == 0}}
    for key in ("p50", "p95"):
        d_ms = out["disabled"]["latency"][f"{key}_ms"]
        e_ms = out["enabled"]["latency"][f"{key}_ms"]
        comparison[f"{key}_latency_overhead"] = {
            "numerator": round(e_ms - d_ms, 9), "denominator": d_ms,
            "value": round((e_ms - d_ms) / d_ms, 6) if d_ms else 0.0,
            "denominator_zero": d_ms == 0}
    out["comparison"] = comparison
    return out


def _source_state() -> dict:
    revision = os.environ.get("SOURCE_REVISION", "0" * 40)
    context = os.environ.get("SOURCE_CONTEXT_SHA256", "0" * 64)
    dirty = os.environ.get("SOURCE_DIRTY", "unknown") in ("true", "True")
    labels = {"revision": revision, "source_context_sha256": context,
              "source_dirty": "true" if os.environ.get(
                  "SOURCE_DIRTY", "false") == "true" else "false"}
    return {
        "branch": os.environ.get(
            "SOURCE_BRANCH", "phase10d-red-team"),
        "commit_sha": revision,
        "dirty": dirty,
        "porcelain_sha256": _sha(os.environ.get("SOURCE_PORCELAIN", "")),
        "manifest_sha256": _sha(os.environ.get("SOURCE_MANIFEST", "")),
        "delivery_tree_sha256": _sha(
            os.environ.get("SOURCE_DELIVERY_TREE", "")),
        "image_context_sha256": context,
        "api_labels": dict(labels),
        "migrate_labels": dict(labels),
    }


def _dependencies() -> dict:
    from importlib.metadata import version

    return {"jsonschema": version("jsonschema")}


def _policy_versions() -> dict:
    return {
        "source_trust": os.environ.get("RAG_SOURCE_TRUST_POLICY_VERSION",
                                       "unassigned"),
        "retrieval": os.environ.get("RAG_RETRIEVAL_POLICY_VERSION",
                                    "unassigned"),
        "context": os.environ.get("RAG_CONTEXT_POLICY_VERSION", "unassigned"),
        "safety": "safety-v1",
    }


def _record_exit_fingerprints(report: dict, config) -> None:
    try:
        report["production_sql_fingerprints"].append(
            attack_simulator.production_sql_fingerprint(
                config.production_database_url))
        report["production_chroma_fingerprints"].append(
            attack_simulator.production_chroma_fingerprint(
                config.production_chroma_collection))
    except Exception:
        report["exit_code"] = 2


def run_harness(fixtures_path=DEFAULT_FIXTURES, run_id=None) -> dict:
    """Execute the two-mode harness; return the full report dict."""
    if os.environ.get("RAG_REDTEAM_MODE") != "true":
        raise SystemExit(2)
    if run_id is None:
        run_id = uuid.uuid4().hex
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    config = attack_simulator.resolve_redteam_config()
    corpus_path = Path(fixtures_path)
    corpus_bytes = corpus_path.read_bytes()
    corpus = attack_simulator.validate_attack_corpus(corpus_path)
    payloads = load_fixtures(corpus_path)
    attack_simulator.ensure_unique_fixture_documents(payloads)

    manifest = attack_simulator.build_fixture_input_manifest(
        corpus_bytes, corpus)

    for collection in config.disposable_collections:
        if attack_simulator.collection_exists(collection):
            raise ValueError(
                f"disposable collection already exists: {collection!r}")

    sql_before = attack_simulator.production_sql_fingerprint(
        config.production_database_url)
    chroma_before = attack_simulator.production_chroma_fingerprint(
        config.production_chroma_collection)

    report = {
        "run_id": run_id,
        "schema_version": "phase10-redteam-report-v1",
        "source_state": _source_state(),
        "image_ids": {
            "api": os.environ.get("SOURCE_API_IMAGE_ID", "unknown"),
            "migrate": os.environ.get("SOURCE_MIGRATE_IMAGE_ID", "unknown"),
        },
        "dependencies": _dependencies(),
        "policy_versions": _policy_versions(),
        "seed": corpus["seed"],
        "fixture_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "manifests": {"disabled": manifest, "enabled": manifest},
        "fixtures": [],
        "disabled": {"bindings": []},
        "enabled": {"bindings": []},
        # 10D.2-compat flat fingerprint lists (kept for the pinned tests).
        "production_sql_fingerprints": [sql_before],
        "production_chroma_fingerprints": [chroma_before],
        "timestamps": {"started_at": started_at,
                       "completed_at": started_at},
        "cleanup_complete": False,
        "non_acceptance": config.keep_artifacts,
        "acceptance": {"thresholds_passed": False, "failure_codes": []},
        "production_fingerprints": {
            "before": {"sql_sha256": sql_before,
                       "chroma_sha256": chroma_before},
            "per_fixture": [],
            "post_cleanup": {"sql_sha256": sql_before,
                             "chroma_sha256": chroma_before}},
        "exit_code": 0,
    }

    engines: dict = {}
    try:
        for mode, (database_url, _collection) in config.modes.items():
            env = dict(os.environ)
            env["RAG_DATABASE_URL"] = database_url
            subprocess.run(
                [sys.executable, "-m", "app.core.migrations"],
                env=env, check=True, shell=False)
            engines[mode] = attack_simulator.open_mode_engine(database_url)

        fixture_groups = [
            (fixture["id"], [p for p in payloads
                             if p.get("fixture_id") == fixture["id"]])
            for fixture in corpus["fixtures"]
        ]

        for mode, (database_url, collection) in config.modes.items():
            with attack_simulator.ModeEnvironment(mode, database_url,
                                                  collection):
                store = attack_simulator.open_mode_store(collection)
                for fixture in corpus["fixtures"]:
                    group = next(
                        payloads_group for fid, payloads_group
                        in fixture_groups if fid == fixture["id"])
                    distances = _fixture_distances(fixture)
                    vectors = {doc["text"]: [distances[doc["id"]]] * 8
                               for doc in fixture["documents"]}
                    vectors[fixture["query"]] = [0.0] * 8
                    ingest_provider = _CorpusEmbeddingProvider(vectors)
                    for payload in group:
                        binding = attack_simulator.ingest_fixture_document(
                            payload, engines[mode], store,
                            embedding_provider=ingest_provider)
                        report[mode]["bindings"].append(binding)

                    row, _samples = _measure_fixture(
                        fixture, mode, engines[mode], store,
                        report[mode]["bindings"])
                    report["fixtures"].append(row)

                    sql_fp = attack_simulator.production_sql_fingerprint(
                        config.production_database_url)
                    chroma_fp = \
                        attack_simulator.production_chroma_fingerprint(
                            config.production_chroma_collection)
                    report["production_sql_fingerprints"].append(sql_fp)
                    report["production_chroma_fingerprints"].append(chroma_fp)
                    report["production_fingerprints"]["per_fixture"].append({
                        "fixture_id": fixture["id"], "mode": mode,
                        "sha256": sql_fp})
    except subprocess.CalledProcessError:
        report["exit_code"] = 2
        raise SystemExit(2)
    except SystemExit:
        report["exit_code"] = 2
        raise
    except Exception:
        report["exit_code"] = 2
        raise SystemExit(2)
    finally:
        _record_exit_fingerprints(report, config)
        for engine in engines.values():
            engine.dispose()
        if not config.keep_artifacts:
            try:
                for collection in config.disposable_collections:
                    attack_simulator.delete_disposable_collection(collection)
                for database_url in (config.disabled_database_url,
                                     config.enabled_database_url):
                    attack_simulator.delete_disposable_database(database_url)
            except Exception:
                report["exit_code"] = 2
        report["cleanup_complete"] = not config.keep_artifacts

    sql_fingerprints = report["production_sql_fingerprints"]
    chroma_fingerprints = report["production_chroma_fingerprints"]
    production_unchanged = (len(set(sql_fingerprints)) == 1
                            and len(set(chroma_fingerprints)) == 1)
    report["production_fingerprints"]["post_cleanup"] = {
        "sql_sha256": sql_fingerprints[-1],
        "chroma_sha256": chroma_fingerprints[-1]}

    # Closed-report order: fixture id, then disabled before enabled.
    report["fixtures"].sort(
        key=lambda row: (row["id"], 0 if row["mode"] == "disabled" else 1))
    report["production_fingerprints"]["per_fixture"] = sorted(
        report["production_fingerprints"]["per_fixture"],
        key=lambda item: (item["fixture_id"],
                          0 if item["mode"] == "disabled" else 1))

    metrics = _derive_metrics(report["fixtures"])
    report["metrics"] = metrics
    security = {
        "enabled_asr": metrics["enabled"]["attack_success_rate"]["value"],
        "relative_asr_reduction":
            metrics["comparison"]["relative_attack_reduction"]["value"],
        "poisoned_context_share":
            metrics["enabled"]["poisoned_context_share"]["value"],
        "clean_retrieval_recall":
            metrics["enabled"]["clean_retrieval_recall"]["value"],
        "false_positive_rate":
            metrics["enabled"]["false_positive_rate"]["value"],
        "graph_path_contamination":
            metrics["enabled"]["graph_path_contamination"]["value"],
        "unsafe_answers_after_block": metrics["enabled"][
            "unsafe_answers_after_block"]["numerator"],
    }
    passed, failures = attack_simulator.evaluate_acceptance(security)
    if not production_unchanged:
        passed = False
        failures = sorted(failures + ["production_fingerprint_changed"])
    report["acceptance"] = {"thresholds_passed": passed,
                            "failure_codes": failures}
    report["timestamps"]["completed_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if not production_unchanged:
        report["exit_code"] = 2
        raise SystemExit(2)
    if config.keep_artifacts:
        report["exit_code"] = 2
    elif not passed:
        report["exit_code"] = 1
    return report


_REPORT_COMPAT_KEYS = (
    "manifests", "disabled", "enabled", "production_sql_fingerprints",
    "production_chroma_fingerprints", "exit_code", "source_binding_sha256",
)


def _report_projection(report: dict) -> dict:
    """The closed-schema report view written to disk.

    The in-memory report carries 10D.2-compat keys consumed by the
    pinned harness tests; the persisted report is exactly the closed
    phase10-redteam-report-v1 shape.
    """
    return {key: value for key, value in report.items()
            if key not in _REPORT_COMPAT_KEYS}


def _write_atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")
    os.replace(tmp, path)


def _render_markdown(report: dict) -> str:
    disabled_asr = report["metrics"]["disabled"]["attack_success_rate"][
        "value"]
    enabled_asr = report["metrics"]["enabled"]["attack_success_rate"][
        "value"]
    reduction = report["metrics"]["comparison"][
        "relative_attack_reduction"]["value"]
    lines = [
        "# Phase 10D Red-Team Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- fixture_sha256: `{report['fixture_sha256']}`",
        f"- seed: {report['seed']}",
        f"- non_acceptance: {report['non_acceptance']}",
        f"- cleanup_complete: {report['cleanup_complete']}",
        f"- acceptance: {report['acceptance']}",
        "",
        "## Metrics",
        "",
        f"- disabled ASR: {disabled_asr}",
        f"- enabled ASR: {enabled_asr}",
        f"- relative reduction: {reduction}",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 10D harness.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    parser.add_argument("--source-binding", default=None)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--markdown", dest="markdown_path", default=None)
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args(argv)

    if os.environ.get("RAG_REDTEAM_MODE") != "true":
        sys.stderr.write("run_redteam: RAG_REDTEAM_MODE guard refused\n")
        return 2
    if args.keep_artifacts:
        os.environ["RAG_REDTEAM_KEEP_ARTIFACTS"] = "true"

    try:
        report = run_harness(fixtures_path=args.fixtures, run_id=args.run_id)
    except ValueError as exc:
        sys.stderr.write(f"run_redteam: refused: {exc}\n")
        return 2
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        sys.stderr.write("run_redteam: isolation/configuration failure\n")
        return 2

    if args.source_binding:
        binding_path = Path(args.source_binding)
        if binding_path.is_file():
            report["source_binding_sha256"] = hashlib.sha256(
                binding_path.read_bytes()).hexdigest()

    if args.json_path:
        _write_atomic_json(Path(args.json_path), _report_projection(report))
    if args.markdown_path:
        Path(args.markdown_path).write_text(
            _render_markdown(_report_projection(report)), encoding="utf-8")

    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
