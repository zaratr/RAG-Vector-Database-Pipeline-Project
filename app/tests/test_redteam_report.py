"""Phase 10D.4 — red-team report schema/validator tests.

Covers: report schema validation, cross-field equations, evaluator rerun,
threshold acceptance, validator argv/network contracts, normalization byte
equality.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import jsonschema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA_PATH = Path("app/tests/fixtures/redteam-report.schema.json")
CORPUS_PATH = Path("app/tests/fixtures/attack_payloads.json")

_FAKE_SHA = "b" * 64
_CORPUS_BYTES = CORPUS_PATH.read_bytes()
_CORPUS = json.loads(_CORPUS_BYTES)
_CORPUS_SHA = hashlib.sha256(_CORPUS_BYTES).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _valid_report(run_id: str = "1" * 32) -> dict:
    """A complete, internally consistent report over the real corpus.

    Disabled rows let every malicious attack succeed; enabled rows block
    them (controls triggered, poison rejected, generation blocked). All
    metrics/acceptance/fingerprint blocks are derived with the same
    equations the validator recomputes.
    """
    rows = []
    sql_counter = {"disabled": 0, "enabled": 0}
    counters = {
        mode: {"goal": 0, "attempted": 0, "sel_poison": 0, "sel_all": 0,
               "sel_clean": 0, "req_clean": 0, "benign": 0,
               "benign_rejected": 0, "poison_paths": 0, "sel_paths": 0,
               "blocked": 0, "unsafe_after_block": 0, "latency": []}
        for mode in ("disabled", "enabled")
    }
    per_fixture = []

    ordered = sorted(
        ((f, mode) for f in _CORPUS["fixtures"]
         for mode in ("disabled", "enabled")),
        key=lambda pair: (pair[0]["id"],
                          0 if pair[1] == "disabled" else 1))

    for fixture, mode in ordered:
        blocked = mode == "enabled" and fixture["kind"] == "malicious"
        counter = counters[mode]
        docs = {d["id"]: d for d in fixture["documents"]}
        clean_ids = fixture["required_clean_document_ids"]
        poison_docs = [d["id"] for d in fixture["documents"] if d["is_poisoned"]]

        candidate_rows = []
        for doc in fixture["documents"]:
            sql_counter[mode] += 1
            sql_id = sql_counter[mode]
            is_poison = doc["is_poisoned"]
            required_clean = doc["id"] in clean_ids
            if fixture["kind"] == "benign":
                selected, decision, reasons = True, "selected", []
            elif blocked and is_poison:
                selected, decision, reasons = False, "rejected", ["rejected_safety"]
            else:
                selected, decision, reasons = True, "selected", []
            candidate_rows.append({
                "document_fixture_id": doc["id"],
                "chunk_fixture_id": f"{doc['id']}:0",
                "sql_document_id": sql_id,
                "sql_chunk_id": sql_id,
                "is_poisoned": is_poison,
                "required_clean": required_clean,
                "selected": selected,
                "decision": decision,
                "reason_codes": reasons,
            })
        selected_chunks = [c["chunk_fixture_id"] for c in candidate_rows
                           if c["selected"]]

        # Graph scenarios carry one path; disabled selects it (poisoned
        # evidence rides along), enabled does not.
        path_rows = []
        if fixture["scenario"].get("paths"):
            poisoned_evidence = poison_docs[0]
            path_rows.append({
                "path_id": f"{fixture['id']}-path-1",
                "selected": mode == "disabled",
                "evidence_fixture_ids": list(
                    fixture["scenario"]["paths"][0]["document_ids"]),
                "contains_poisoned_evidence": True,
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

        ev = fixture["evaluator"]
        succeeded = fixture["kind"] == "malicious" and mode == "disabled"
        generation_blocked = blocked
        answer = None if generation_blocked else (
            fixture["scenario"].get("disabled_answer")
            or f"answer-{fixture['id']}-{mode}")
        unsafe_returned = fixture["kind"] == "malicious" and mode == "disabled" \
            and ev["type"] == "unsafe_answer_returned"

        if ev["type"] == "poisoned_chunk_selected":
            observations = {
                "type": "poisoned_chunk_selected",
                "target_chunk_fixture_id": f"{ev['poisoned_document_id']}:0",
                "selected_chunk_fixture_ids": selected_chunks,
            }
        elif ev["type"] == "answer_contains_literal":
            literal = ev["literal"]
            observations = {
                "type": "answer_contains_literal",
                "literal_sha256": _sha(literal),
                "returned_marker_ids": [f"marker-{_sha(literal)[:8]}"],
                "matched_marker_id":
                    f"marker-{_sha(literal)[:8]}" if succeeded else None,
                "match_start": 0 if succeeded else None,
                "match_end": len(literal) if succeeded else None,
            }
        elif ev["type"] == "graph_path_contains_poisoned_evidence":
            observations = {
                "type": "graph_path_contains_poisoned_evidence",
                "poisoned_evidence_fixture_ids":
                    [f"{d}:0" for d in poison_docs],
                "selected_path_ids":
                    [p["path_id"] for p in path_rows if p["selected"]],
                "selected_evidence_fixture_ids":
                    [p["evidence_fixture_ids"][0]
                     for p in path_rows if p["selected"]],
            }
        elif ev["type"] == "unsafe_answer_returned":
            observations = {
                "type": "unsafe_answer_returned",
                "block_decision_ids":
                    [f"block-{fixture['id']}"] if blocked else [],
                "unsafe_marker_ids":
                    [f"marker-{_sha(ev['literal'])[:8]}"]
                    if unsafe_returned else [],
            }
        else:  # control_not_triggered (benign controls)
            observations = {
                "type": "control_not_triggered",
                "control_id": ev["control_id"],
                "would_trigger": False,
                "enforced": mode == "enabled",
                "outcome": "benign_preserved",
            }

        block_count = 1 if blocked else 0
        row = {
            "id": fixture["id"],
            "category": fixture["category"],
            "kind": fixture["kind"],
            "severity": fixture["severity"],
            "mode": mode,
            "expected_safe_outcome": fixture["expected_safe_outcome"],
            "query_sha256": _sha(fixture["query"]),
            "attack_succeeded": succeeded,
            "evaluator": {**ev, "result": succeeded
                          if fixture["kind"] == "malicious" else True},
            "control_results": control_results,
            "candidate_rows": candidate_rows,
            "path_rows": path_rows,
            "required_clean_ids": [f"{d}:0" for d in clean_ids],
            "selected_clean_ids": [
                c["chunk_fixture_id"] for c in candidate_rows
                if c["selected"] and c["required_clean"]],
            "selected_poisoned_ids": [
                c["chunk_fixture_id"] for c in candidate_rows
                if c["selected"] and c["is_poisoned"]],
            "benign_chunk_ids": [
                c["chunk_fixture_id"] for c in candidate_rows
                if fixture["kind"] == "benign"],
            "rejected_benign_ids": [],
            "evaluator_observations": observations,
            "block_decision_count": block_count,
            "generation_blocked": generation_blocked,
            "unsafe_answer_returned": unsafe_returned,
            "answer_sha256": None if generation_blocked else _sha(answer),
            "latency_ns": [10 ** 6 + i for i in range(10)],
        }
        rows.append(row)
        per_fixture.append({"fixture_id": fixture["id"], "mode": mode,
                            "sha256": _FAKE_SHA})

        if fixture["kind"] == "malicious":
            counter["attempted"] += 1
            counter["goal"] += 1 if succeeded else 0
        counter["sel_poison"] += len(row["selected_poisoned_ids"])
        counter["sel_all"] += len(selected_chunks)
        counter["sel_clean"] += len(row["selected_clean_ids"])
        counter["req_clean"] += len(row["required_clean_ids"])
        counter["benign"] += len(row["benign_chunk_ids"])
        counter["benign_rejected"] += len(row["rejected_benign_ids"])
        counter["poison_paths"] += sum(
            1 for p in path_rows if p["selected"] and p["contains_poisoned_evidence"])
        counter["sel_paths"] += sum(1 for p in path_rows if p["selected"])
        counter["blocked"] += block_count
        if unsafe_returned and block_count:
            counter["unsafe_after_block"] += 1
        counter["latency"].extend(row["latency_ns"])

    def ratio(num, den):
        return {"numerator": num, "denominator": den,
                "value": round(num / den, 6) if den else 0.0,
                "denominator_zero": den == 0}

    def nearest_rank(samples, pct):
        ordered = sorted(samples)
        rank = min(max(-(-pct * len(ordered) // 100) or 1, 1), len(ordered))
        return ordered[rank - 1]

    mode_metrics = {}
    for mode in ("disabled", "enabled"):
        c = counters[mode]
        mode_metrics[mode] = {
            "attack_success_rate": ratio(c["goal"], c["attempted"]),
            "poisoned_context_share": ratio(c["sel_poison"], c["sel_all"]),
            "clean_retrieval_recall": ratio(c["sel_clean"], c["req_clean"]),
            "false_positive_rate": ratio(c["benign_rejected"], c["benign"]),
            "graph_path_contamination": ratio(c["poison_paths"], c["sel_paths"]),
            "blocked_generation_count": c["blocked"],
            "unsafe_answers_after_block": ratio(c["unsafe_after_block"], c["blocked"]),
            "latency": {
                "sample_count": len(c["latency"]),
                "p50_ms": nearest_rank(c["latency"], 50) / 1e6,
                "p95_ms": nearest_rank(c["latency"], 95) / 1e6,
            },
        }
    d_asr = mode_metrics["disabled"]["attack_success_rate"]["value"]
    e_asr = mode_metrics["enabled"]["attack_success_rate"]["value"]
    comparison = {"relative_attack_reduction": {
        "numerator": round(d_asr - e_asr, 6), "denominator": d_asr,
        "value": round((d_asr - e_asr) / d_asr, 6),
        "denominator_zero": d_asr == 0}}
    for key in ("p50", "p95"):
        d_ms = mode_metrics["disabled"]["latency"][f"{key}_ms"]
        e_ms = mode_metrics["enabled"]["latency"][f"{key}_ms"]
        comparison[f"{key}_latency_overhead"] = {
            "numerator": round(e_ms - d_ms, 9), "denominator": d_ms,
            "value": round((e_ms - d_ms) / d_ms, 6),
            "denominator_zero": d_ms == 0}

    return {
        "schema_version": "phase10-redteam-report-v1",
        "run_id": run_id,
        "source_state": {
            "branch": "phase10d-red-team",
            "commit_sha": "a" * 40,
            "dirty": False,
            "porcelain_sha256": _FAKE_SHA,
            "manifest_sha256": _FAKE_SHA,
            "delivery_tree_sha256": _FAKE_SHA,
            "image_context_sha256": _FAKE_SHA,
            "api_labels": {"revision": "a" * 40,
                           "source_context_sha256": _FAKE_SHA,
                           "source_dirty": "false"},
            "migrate_labels": {"revision": "a" * 40,
                               "source_context_sha256": _FAKE_SHA,
                               "source_dirty": "false"},
        },
        "image_ids": {"api": "img-api", "migrate": "img-migrate"},
        "dependencies": {"jsonschema": "4.25.1"},
        "policy_versions": {"source_trust": "v1"},
        "seed": 42,
        "fixture_sha256": _CORPUS_SHA,
        "fixtures": rows,
        "metrics": {
            "methodology": {
                "clock": "perf_counter_ns", "warmups_per_fixture": 3,
                "measured_repetitions_per_fixture": 10,
                "percentile_method": "nearest_rank"},
            "disabled": mode_metrics["disabled"],
            "enabled": mode_metrics["enabled"],
            "comparison": comparison,
        },
        "timestamps": {"started_at": "2026-08-20T00:00:00Z",
                       "completed_at": "2026-08-20T00:05:00Z"},
        "cleanup_complete": True,
        "non_acceptance": False,
        "acceptance": {"thresholds_passed": True, "failure_codes": []},
        "production_fingerprints": {
            "before": {"sql_sha256": _FAKE_SHA, "chroma_sha256": _FAKE_SHA},
            "per_fixture": per_fixture,
            "post_cleanup": {"sql_sha256": _FAKE_SHA,
                             "chroma_sha256": _FAKE_SHA}},
    }


def _load_valid_report() -> dict:
    return _valid_report()


def _run_validator(report: dict, tmp_path=None):
    """Validate a report; semantic failures surface as SystemExit(1).

    With ``tmp_path`` the validator runs through its real main() (argv
    and exit code end to end); otherwise validate_report is called
    directly with its typed failure converted to the CLI contract.
    """
    from scripts.validate_redteam_report import _Invalid, main, validate_report
    schema = json.loads(REPORT_SCHEMA_PATH.read_text())
    if tmp_path is not None:
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report))
        raise SystemExit(main(
            [str(report_path),
             "--schema", "app/tests/fixtures/redteam-report.schema.json"]))
    try:
        return validate_report(report, schema)
    except _Invalid:
        raise SystemExit(1)


def _normalize(path: Path) -> Path:
    from scripts.normalize_redteam_report import normalize_report
    payload = json.loads(path.read_text())
    out = path.with_suffix(".normalized.json")
    out.write_text(json.dumps(normalize_report(payload), sort_keys=True,
                              separators=(",", ":")))
    return out


def test_report_schema_is_draft2020_12_with_id():
    schema = json.loads(REPORT_SCHEMA_PATH.read_text())
    assert schema["$schema"] == "https://jsonschema-12/draft/2020-12/schema"  # exact str per impl
    assert schema["$id"] == "phase10-redteam-report-v1"


def test_report_schema_additional_properties_false_at_every_level():
    schema = json.loads(REPORT_SCHEMA_PATH.read_text())
    # Walk every object definition and assert additionalProperties is False
    # (except anchored patternProperties maps).
    def _walk(node, path):
        if isinstance(node, dict):
            if node.get("type") == "object" and "patternProperties" not in node:
                assert node.get("additionalProperties") is False, (
                    "additionalProperties not False at " + ".".join(map(str, path))
                )
            for key, child in node.items():
                _walk(child, path + [key])
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                _walk(child, path + [idx])
    _walk(schema, ["$"])
    # anchored dynamic maps use patternProperties with additionalProperties False
    for name in ("dependencies", "policy_versions"):
        node = schema["$defs"][name]
        assert "patternProperties" in node
        assert node["additionalProperties"] is False


def test_count_ratio_schema_cross_field_numerator_le_denominator():
    schema = json.loads(REPORT_SCHEMA_PATH.read_text())
    bad = {"numerator": 5, "denominator": 3, "value": 1.0, "denominator_zero": False}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema["$defs"]["CountRatio"]).validate(bad)


def test_count_ratio_value_zero_when_denominator_zero():
    from app.services.attack_simulator import CountRatio
    r = CountRatio(numerator=0, denominator=0)
    assert r.value == 0.0 and r.denominator_zero is True


def test_validator_recomputes_evaluator_result_from_candidate_rows(tmp_path):
    report = _load_valid_report()
    # Tamper evaluator.result to the opposite of what candidate_rows imply;
    # validator must exit 1.
    report["fixtures"][0]["evaluator"]["result"] = not report["fixtures"][0]["evaluator"]["result"]
    with pytest.raises(SystemExit) as exc:
        _run_validator(report, tmp_path)
    assert exc.value.code == 1


def test_validator_rejects_attack_succeeded_mismatch_for_malicious(tmp_path):
    report = _load_valid_report()
    fix = next(f for f in report["fixtures"] if f["kind"] == "malicious")
    fix["attack_succeeded"] = not fix["attack_succeeded"]
    with pytest.raises(SystemExit):
        _run_validator(report, tmp_path)


def test_validator_rejects_benign_attack_succeeded_true(tmp_path):
    report = _load_valid_report()
    fix = next(f for f in report["fixtures"] if f["kind"] == "benign")
    fix["attack_succeeded"] = True
    with pytest.raises(SystemExit):
        _run_validator(report, tmp_path)


def test_validator_rejects_sql_fixture_mapping_mismatch(tmp_path):
    report = _load_valid_report()
    report["fixtures"][0]["candidate_rows"][0]["sql_document_id"] = 999999
    with pytest.raises(SystemExit):
        _run_validator(report, tmp_path)


def test_validator_requires_one_disabled_and_one_enabled_per_fixture(tmp_path):
    report = _load_valid_report()
    # Drop the enabled row for fixture 0
    report["fixtures"] = [r for r in report["fixtures"]
                          if not (r["id"] == report["fixtures"][0]["id"]
                                  and r["mode"] == "enabled")]
    with pytest.raises(SystemExit):
        _run_validator(report, tmp_path)


def test_validator_rejects_surviving_disposable_id_in_report(tmp_path):
    report = _load_valid_report()
    report["run_id"] = "redteam-deadbeef" * 4  # looks like a surviving disposable ID
    with pytest.raises(SystemExit):
        _run_validator(report, tmp_path)


def test_validator_rejects_production_fingerprint_change(tmp_path):
    report = _load_valid_report()
    report["production_fingerprints"]["post_cleanup"]["sql_sha256"] = "deadbeef"
    with pytest.raises(SystemExit):
        _run_validator(report, tmp_path)


def test_acceptance_thresholds_passed_true_only_when_all_pass(tmp_path):
    report = _load_valid_report()
    assert report["acceptance"]["thresholds_passed"] is True
    report["metrics"]["enabled"]["attack_success_rate"]["value"] = 0.5  # > 0.20
    with pytest.raises(SystemExit):
        _run_validator(report, tmp_path)


def test_acceptance_failure_codes_sorted():
    report = _load_valid_report()
    assert report["acceptance"]["failure_codes"] == sorted(report["acceptance"]["failure_codes"])


def test_answer_sha256_null_iff_generation_blocked():
    report = _load_valid_report()
    for fix in report["fixtures"]:
        if fix["generation_blocked"]:
            assert fix["answer_sha256"] is None
        else:
            assert fix["answer_sha256"] is not None


def test_latency_ns_contains_exactly_ten_positive_samples():
    report = _load_valid_report()
    for fix in report["fixtures"]:
        assert len(fix["latency_ns"]) == 10
        assert all(s > 0 for s in fix["latency_ns"])


def test_validate_redteam_report_argv_exact(monkeypatch, tmp_path):
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_valid_report()))
    schema = "app/tests/fixtures/redteam-report.schema.json"
    argv = [sys.executable, "scripts/validate_redteam_report.py",
            str(report), "--schema", schema]
    result = subprocess.run(argv, cwd=PROJECT_ROOT, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out == {"schema_version": "phase10-redteam-report-v1", "status": "valid"}


def test_validate_redteam_report_no_network_access(monkeypatch):
    # Patch socket.socket to raise; validator must still succeed on a valid
    # report (proves no network fetch).
    import socket
    from scripts.validate_redteam_report import validate_report

    def _refuse(*args, **kwargs):
        raise AssertionError("validator attempted network access")

    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr("urllib.request.urlopen", _refuse)
    schema = json.loads(REPORT_SCHEMA_PATH.read_text())
    status = validate_report(_valid_report(), schema)
    assert status == {"schema_version": "phase10-redteam-report-v1", "status": "valid"}


def test_normalized_reports_byte_equal_for_identical_runs(monkeypatch, tmp_path):
    r1 = tmp_path / "r1.json"
    r2 = tmp_path / "r2.json"
    r1.write_text(json.dumps(_valid_report(run_id="a" * 32)))
    r2.write_text(json.dumps(_valid_report(run_id="b" * 32)))
    n1 = _normalize(r1)
    n2 = _normalize(r2)
    assert n1.read_bytes() == n2.read_bytes()


def test_normalize_strips_latency_overhead_ratios(tmp_path):
    """D-86: wall-clock comparison ratios must not survive normalization —
    two live runs of the same image normalize byte-identically."""
    from scripts.normalize_redteam_report import normalize_report
    base = _valid_report(run_id="a" * 32)
    other = _valid_report(run_id="b" * 32)
    base["metrics"]["comparison"]["p50_latency_overhead"] = {
        "numerator": 1.5, "denominator": 10.0, "value": 0.15,
        "denominator_zero": False}
    other["metrics"]["comparison"]["p50_latency_overhead"] = {
        "numerator": 42.0, "denominator": 8.0, "value": 5.25,
        "denominator_zero": False}
    base["metrics"]["comparison"]["p95_latency_overhead"] = {
        "numerator": 2.0, "denominator": 10.0, "value": 0.2,
        "denominator_zero": False}
    other["metrics"]["comparison"]["p95_latency_overhead"] = {
        "numerator": 7.0, "denominator": 9.0, "value": 0.777778,
        "denominator_zero": False}
    n1 = normalize_report(base)
    n2 = normalize_report(other)
    assert "p50_latency_overhead" not in n1["metrics"]["comparison"]
    assert "p95_latency_overhead" not in n1["metrics"]["comparison"]
    assert json.dumps(n1, sort_keys=True) == json.dumps(
        n2, sort_keys=True)
