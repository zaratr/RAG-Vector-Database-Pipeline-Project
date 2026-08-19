"""Phase 10D gate orchestrator: two recorded red-team runs, containerized
schema validation per report, host normalization, byte comparison.

Validates the closed non-secret source binding against the manifest (and
refuses any production-name propagation inside it) BEFORE any container
child runs, atomically copies the canonical binding bytes to
``<output>/source-binding.json``, then uses typed ``subprocess.run``
argv arrays — never ``shell=True`` — to invoke, in order: the Task 10D.2
``run_redteam`` argv once per run (aliases ``run1``/``run2``); the
containerized ``validate_redteam_report`` for each report; host
``normalize_redteam_report`` with atomic normalized writes; and a
byte-comparison of the normalized outputs plus cleanup/fingerprint
invariant inspection. Returns the first child exit code (1 defense
failure, 2 isolation/schema/configuration failure) or 0.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_PRODUCTION_MARKERS = ("rag-collection", "sqlite:////data/rag.db")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _refuse_binding_names(binding_path: Path) -> None:
    """Refuse production-name propagation inside the source binding.

    The binding itself must never carry the production store identities
    into a child run; detection happens before any child executes.
    """
    if not binding_path.is_file():
        return
    try:
        payload = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for value in _iter_strings(payload):
        if value in _PRODUCTION_MARKERS:
            raise SystemExit(2)


def _iter_strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _iter_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_strings(value)
    elif isinstance(node, str):
        yield node


def _validate_binding_matches_manifest(binding_path: Path,
                                       manifest_path: Path) -> None:
    """Closed non-secret binding must reference the manifest's commit."""
    if not (binding_path.is_file() and manifest_path.is_file()):
        return
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = manifest.get("manifest_sha256") or manifest.get("sha256")
    binding_sha = binding.get("manifest_sha256")
    if manifest_sha and binding_sha and manifest_sha != binding_sha:
        raise SystemExit(2)


def _redteam_argv(output: Path, run_id: str, disabled_id: str,
                  enabled_id: str, alias: str) -> list:
    return [
        "docker", "compose", "run", "--rm",
        "-v", f"{output.resolve().as_posix()}:/reports",
        "-e", "RAG_REDTEAM_MODE=true",
        "-e", f"RAG_REDTEAM_DISABLED_DATABASE_URL=sqlite:////tmp/redteam-{disabled_id}.db",
        "-e", f"RAG_REDTEAM_DISABLED_CHROMA_COLLECTION=redteam-{disabled_id}",
        "-e", f"RAG_REDTEAM_ENABLED_DATABASE_URL=sqlite:////tmp/redteam-{enabled_id}.db",
        "-e", f"RAG_REDTEAM_ENABLED_CHROMA_COLLECTION=redteam-{enabled_id}",
        "-e", "RAG_PRODUCTION_DATABASE_URL=sqlite:////data/rag.db",
        "-e", "RAG_PRODUCTION_CHROMA_COLLECTION=rag-collection",
        "api", "python", "scripts/run_redteam.py",
        "--run-id", run_id,
        "--fixtures", "app/tests/fixtures/attack_payloads.json",
        "--source-binding", "/reports/source-binding.json",
        "--json", f"/reports/{alias}.json",
        "--markdown", f"/reports/{alias}.md",
    ]


def _validate_argv(output: Path, alias: str) -> list:
    return [
        "docker", "compose", "run", "--rm",
        "-v", f"{output.resolve().as_posix()}:/reports",
        "api", "python", "scripts/validate_redteam_report.py",
        f"/reports/{alias}.json",
        "--schema", "app/tests/fixtures/redteam-report.schema.json",
        "--source-binding", "/reports/source-binding.json",
    ]


def _normalize_argv(report_path: Path, normalized_path: Path) -> list:
    # Invoked as a module from the scripts directory so the child argv
    # carries the bare module name; report/output paths stay absolute.
    return [sys.executable, "-m", "normalize_redteam_report",
            str(report_path), "--output", str(normalized_path)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate the Phase 10D recorded red-team gate.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-binding", required=True)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    binding_path = Path(args.source_binding)
    manifest_path = Path(args.source_manifest)

    # Refusals before any container child runs.
    try:
        _refuse_binding_names(binding_path)
        _validate_binding_matches_manifest(binding_path, manifest_path)
    except SystemExit as exc:
        return int(exc.code or 2)
    if binding_path.is_file():
        _atomic_write_bytes(output / "source-binding.json",
                            binding_path.read_bytes())

    first_exit = 0
    validator_outputs: list = []
    schema_valid = 0
    for alias in ("run1", "run2"):
        run_id = uuid.uuid4().hex
        disabled_id = uuid.uuid4().hex
        enabled_id = uuid.uuid4().hex
        result = subprocess.run(_redteam_argv(output, run_id, disabled_id,
                                              enabled_id, alias),
                                check=False)
        if result.returncode != 0 and first_exit == 0:
            first_exit = result.returncode
        report_path = output / f"{alias}.json"

        validation = subprocess.run(_validate_argv(output, alias),
                                    check=False)
        validator_outputs.append(validation.stdout.strip())
        if validation.returncode == 0:
            schema_valid += 1
        elif first_exit == 0:
            first_exit = validation.returncode

        normalized_path = output / f"{alias}.normalized.json"
        normalization = subprocess.run(
            _normalize_argv(report_path, normalized_path), check=False,
            cwd=str(_REPO_ROOT / "scripts"))
        if normalization.returncode != 0 and first_exit == 0:
            first_exit = normalization.returncode

    normalized_equal = True
    normalized_files = [output / f"{alias}.normalized.json"
                        for alias in ("run1", "run2")]
    if all(path.is_file() for path in normalized_files):
        normalized_equal = (
            normalized_files[0].read_bytes()
            == normalized_files[1].read_bytes())
        if not normalized_equal and first_exit == 0:
            first_exit = 2
    elif first_exit == 0:
        first_exit = 2

    cleanup_complete = True
    production_unchanged = True
    for alias in ("run1", "run2"):
        report_path = output / f"{alias}.json"
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            first_exit = first_exit or 2
            continue
        cleanup_complete = cleanup_complete and report.get("cleanup_complete")
        fingerprints = report.get("production_fingerprints", {})
        before = fingerprints.get("before", {})
        post = fingerprints.get("post_cleanup", {})
        production_unchanged = production_unchanged and (
            before.get("sql_sha256") == post.get("sql_sha256")
            and before.get("chroma_sha256") == post.get("chroma_sha256"))
        if "redteam-" + "0" * 32 in json.dumps(report):
            first_exit = first_exit or 2

    for line in validator_outputs:
        if line:
            sys.stdout.write(line + "\n")
    sys.stdout.write(json.dumps({
        "runs": 2,
        "schema_valid": schema_valid,
        "normalized_equal": normalized_equal,
        "cleanup_complete": cleanup_complete,
        "production_unchanged": production_unchanged,
    }, sort_keys=True) + "\n")
    return first_exit


if __name__ == "__main__":
    raise SystemExit(main())
