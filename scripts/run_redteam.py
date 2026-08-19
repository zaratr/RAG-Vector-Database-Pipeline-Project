"""Task 10D.2 red-team harness CLI (guarded, isolated, disposable).

Runs the pinned attack corpus through the exact production ingestion path
twice — a ``disabled`` mode that observes but does not enforce the
measured content-safety control, and an ``enabled`` mode that enforces
it — each against its own migrated, UUID-named disposable SQL database
and Chroma collection. Production stores are opened read-only for
fingerprints only; every refusal happens before any mutation; one outer
``finally`` removes both disposable stores and proves production
unchanged. Exit codes: 0 harness complete, 1 measured defense failure
(Task 10D.3), 2 isolation/configuration refusal or corpus-invalid input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from itertools import groupby
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services import attack_simulator  # noqa: E402

DEFAULT_FIXTURES = "app/tests/fixtures/attack_payloads.json"


def load_fixtures(fixtures_path) -> list:
    """Validate the corpus and flatten it to ingestion payloads.

    Payload order is the corpus's lexical fixture/document order; each
    payload carries its fixture id so the orchestrator can fingerprint
    after every fixture.
    """
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
    """Execute the two-mode harness; return the in-memory report dict.

    Refusals raise ``ValueError``; corpus-invalid inputs and isolation
    failures raise ``SystemExit(2)``. Cleanup runs in one outer
    ``finally`` regardless of outcome.
    """
    if os.environ.get("RAG_REDTEAM_MODE") != "true":
        raise SystemExit(2)
    if run_id is None:
        run_id = uuid.uuid4().hex

    config = attack_simulator.resolve_redteam_config()
    corpus_path = Path(fixtures_path)
    corpus_bytes = corpus_path.read_bytes()
    corpus = attack_simulator.validate_attack_corpus(corpus_path)
    payloads = load_fixtures(corpus_path)
    attack_simulator.ensure_unique_fixture_documents(payloads)

    manifest = attack_simulator.build_fixture_input_manifest(
        corpus_bytes, corpus)

    # Refuse pre-existing disposable collections before any mutation.
    for collection in config.disposable_collections:
        if attack_simulator.collection_exists(collection):
            raise ValueError(
                f"disposable collection already exists: {collection!r}")

    report = {
        "run_id": run_id,
        "schema_version": "phase10-redteam-report-v1",
        "fixture_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "seed": corpus["seed"],
        "manifests": {"disabled": manifest, "enabled": manifest},
        "disabled": {"bindings": []},
        "enabled": {"bindings": []},
        "production_sql_fingerprints": [
            attack_simulator.production_sql_fingerprint(
                config.production_database_url)],
        "production_chroma_fingerprints": [
            attack_simulator.production_chroma_fingerprint(
                config.production_chroma_collection)],
        "non_acceptance": config.keep_artifacts,
        "cleanup_complete": False,
        "exit_code": 0,
    }

    engines: dict = {}
    try:
        # Per-mode migration through the supported subprocess wrapper.
        for mode, (database_url, _collection) in config.modes.items():
            env = dict(os.environ)
            env["RAG_DATABASE_URL"] = database_url
            subprocess.run(
                [sys.executable, "-m", "app.core.migrations"],
                env=env, check=True, shell=False)
            engines[mode] = attack_simulator.open_mode_engine(database_url)

        fixture_groups = [
            (fixture_id, list(group))
            for fixture_id, group in groupby(
                payloads, key=lambda payload: payload.get("fixture_id"))
        ]

        for mode, (database_url, collection) in config.modes.items():
            with attack_simulator.ModeEnvironment(mode, database_url,
                                                  collection):
                store = attack_simulator.open_mode_store(collection)
                for _fixture_id, group in fixture_groups:
                    for payload in group:
                        binding = attack_simulator.ingest_fixture_document(
                            payload, engines[mode], store)
                        report[mode]["bindings"].append(binding)
                    # Production fingerprints after every fixture.
                    report["production_sql_fingerprints"].append(
                        attack_simulator.production_sql_fingerprint(
                            config.production_database_url))
                    report["production_chroma_fingerprints"].append(
                        attack_simulator.production_chroma_fingerprint(
                            config.production_chroma_collection))
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
                # A cleanup failure can never be masked: force refusal class.
                report["exit_code"] = 2
        report["cleanup_complete"] = not config.keep_artifacts

    sql_fingerprints = report["production_sql_fingerprints"]
    chroma_fingerprints = report["production_chroma_fingerprints"]
    if len(set(sql_fingerprints)) != 1 or len(set(chroma_fingerprints)) != 1:
        report["exit_code"] = 2
        raise SystemExit(2)
    if config.keep_artifacts:
        # Debug retention can never produce an acceptance-grade result.
        report["exit_code"] = 2
    return report


def _write_atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")
    os.replace(tmp, path)


def _render_markdown(report: dict) -> str:
    lines = [
        "# Phase 10D Red-Team Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- fixture_sha256: `{report['fixture_sha256']}`",
        f"- seed: {report['seed']}",
        f"- non_acceptance: {report['non_acceptance']}",
        f"- cleanup_complete: {report['cleanup_complete']}",
        "",
        "## Bindings",
        "",
    ]
    for mode in ("disabled", "enabled"):
        bindings = report[mode]["bindings"]
        blocked = sum(1 for b in bindings if b["status"] == "failed")
        lines.append(f"- {mode}: {len(bindings)} documents, "
                     f"{blocked} failed")
    lines += [
        "",
        "## Production fingerprints",
        "",
        f"- sql unchanged: "
        f"{len(set(report['production_sql_fingerprints'])) == 1}",
        f"- chroma unchanged: "
        f"{len(set(report['production_chroma_fingerprints'])) == 1}",
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
        _write_atomic_json(Path(args.json_path), report)
    if args.markdown_path:
        Path(args.markdown_path).write_text(
            _render_markdown(report), encoding="utf-8")

    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
