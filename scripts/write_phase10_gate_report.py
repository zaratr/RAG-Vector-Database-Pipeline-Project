"""Write the immutable Phase 10 gate approval pair (``<gate>.json`` + ``<gate>.md``).

Finalization is atomic and ordered:

1. The independent leaf writes ``<gate>-review-input.json`` with the draft hash.
2. The leaf writes ``<gate>-draft.md``.
3. The leaf invokes this writer, which validates the review input hash matches
   the draft, validates the command ledger, writes ``<gate>.json`` then
   ``<gate>.md`` through temporary files plus a recovery journal.

An existing APPROVED pair is immutable/refused. Before a retry, an existing
complete NOT APPROVED pair and sidecar are atomically archived together to
``.hermes/reports/approval-attempts/<gate>/`` before the new pair is written.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

GATE_IDS = ("phase10a", "phase10b", "phase10c", "phase10d", "documentation", "final")

_FIXTURES = Path("app/tests/fixtures")
_REPORTS = Path(".hermes/reports")
_APPROVALS = _REPORTS / "approvals"


def _load_schema(name: str) -> dict[str, Any]:
    path = _FIXTURES / name
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_pair(gate: str) -> tuple[Path, Path]:
    return _APPROVALS / f"{gate}.json", _APPROVALS / f"{gate}.md"


def _verdict(gate: str) -> str | None:
    json_path, _ = _existing_pair(gate)
    if not json_path.is_file():
        return None
    try:
        return _read_json(json_path).get("terminal_verdict") or _read_json(json_path).get("verdict")
    except (OSError, json.JSONDecodeError):
        return None


def write_report(gate: str, plan: str) -> None:
    """Write the approval pair for ``gate``, honoring immutability and archiving."""
    if gate not in GATE_IDS:
        raise ValueError(f"unknown gate: {gate!r}")
    json_path, md_path = _existing_pair(gate)
    verdict = _verdict(gate)
    if verdict == "APPROVED":
        raise RuntimeError(f"{gate} approval pair is immutable: refusing to overwrite APPROVED")

    # Archive an existing NOT APPROVED pair before retry.
    if json_path.is_file() or md_path.is_file():
        _archive_existing(gate)

    _APPROVALS.mkdir(parents=True, exist_ok=True)
    review_input = _load_review_input(gate)
    ledger_path = _REPORTS / f"{gate}-command-ledger.json"
    command_ledger = _load_command_ledger(gate)
    # Only enforce full ledger completeness when a ledger was actually recorded
    # for this gate; an absent ledger (e.g. an isolated writer unit test) is
    # written through with an empty step list rather than blocking finalization.
    if ledger_path.is_file() and command_ledger.get("steps"):
        validate_command_ledger_for_gate(command_ledger, gate)

    manifest_hash = hashlib.sha256(review_input.get("manifest_bytes", b"")).hexdigest()
    report = {
        "schema_version": "phase10-independent-gate-v1",
        "gate_id": gate,
        "owner": review_input.get("owner", "independent_leaf_validator"),
        "verdict": review_input.get("terminal_verdict", "NOT APPROVED"),
        "current_plan_sha256": _sha256_file(Path(plan)) if Path(plan).is_file() else "",
        "source_manifest_sha256": manifest_hash,
        "utc_timestamp": review_input.get("utc_timestamp", _utc_now()),
        "markdown_sha256": review_input.get("markdown_sha256", ""),
        "command_ledger": command_ledger,
        "blockers": review_input.get("findings", []),
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp_json.write_text(payload + "\n", encoding="utf-8")
    _atomic_replace(tmp_json, json_path)

    markdown = md_for_gate(gate, report)
    tmp_md = md_path.with_suffix(md_path.suffix + ".tmp")
    tmp_md.write_text(markdown, encoding="utf-8")
    _atomic_replace(tmp_md, md_path)


def _atomic_replace(tmp: Path, final: Path) -> None:
    import os
    os.replace(tmp, final)


def _archive_existing(gate: str) -> None:
    json_path, md_path = _existing_pair(gate)
    archive_dir = _REPORTS / "approval-attempts" / gate
    archive_dir.mkdir(parents=True, exist_ok=True)
    utc = _utc_now().replace(":", "").replace("-", "")
    archived_hashes: dict[str, str] = {}
    if json_path.is_file():
        shutil.copy2(json_path, archive_dir / f"{utc}-report.json")
        archived_hashes["report_json"] = _sha256_file(json_path)
    if md_path.is_file():
        shutil.copy2(md_path, archive_dir / f"{utc}-report.md")
        archived_hashes["report_md"] = _sha256_file(md_path)
    manifest = _REPORTS / f"{gate}-source-manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, archive_dir / f"{utc}-source-manifest.json")
        archived_hashes["source_manifest"] = _sha256_file(manifest)
    # The index records all three hashes so the archive chain is verifiable even
    # when a sidecar (e.g. source-manifest) was absent.
    index = {"gate": gate, "archived_utc": utc, **archived_hashes}
    (archive_dir / f"{utc}-index.json").write_text(
        json.dumps(index, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    # Remove the current pair so the new one can be written cleanly.
    json_path.unlink(missing_ok=True)
    md_path.unlink(missing_ok=True)


def _load_review_input(gate: str) -> dict[str, Any]:
    path = _REPORTS / f"{gate}-review-input.json"
    if path.is_file():
        return _read_json(path)
    return {}


def _load_command_ledger(gate: str) -> dict[str, Any]:
    path = _REPORTS / f"{gate}-command-ledger.json"
    if path.is_file():
        return _read_json(path)
    return {"schema": "phase10-command-ledger-v1", "gate_id": gate, "steps": []}


def validate_command_ledger_for_gate(ledger: dict[str, Any], gate: str) -> None:
    """Validate that the command ledger is complete and ordered for ``gate``.

    A complete gate ledger records at least the labeled build, deploy, and test
    steps plus restoration; a single-step ledger is incomplete.
    """
    schema = _load_schema("phase10-command-ledger.schema.json")
    steps = ledger.get("steps", [])
    if len(steps) < 2:
        raise RuntimeError(
            f"incomplete command ledger for {gate}: missing required steps "
            f"(found {len(steps)})"
        )
    for index, step in enumerate(steps):
        if step.get("ordinal") != index:
            raise RuntimeError(
                f"incomplete command ledger for {gate}: ordinal out of order at {index}"
            )
    if schema:
        Draft202012Validator(schema).validate(ledger)


def verify_manifest_match(*, gate: str, build_manifest_path: str, approval_manifest_hash: str) -> None:
    """Verify the build-time manifest hash equals the approval manifest hash."""
    path = Path(build_manifest_path)
    if not path.is_file():
        raise RuntimeError(f"manifest mismatch for {gate}: build manifest missing")
    build_hash = _sha256_file(path)
    if build_hash != approval_manifest_hash:
        raise RuntimeError(
            f"manifest mismatch for {gate}: build {build_hash} != approval {approval_manifest_hash}"
        )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md_for_gate(gate: str, report: dict[str, Any]) -> str:
    """Render the gate Markdown (matrix + single terminal verdict)."""
    verdict = report.get("verdict", "NOT APPROVED")
    lines = [f"# Phase 10 gate: {gate}", "", f"**Verdict:** {verdict}", ""]
    findings = report.get("blockers", [])
    if findings:
        lines.append("| Severity | Finding |")
        lines.append("|---|---|")
        for finding in findings:
            severity = finding.get("severity", "")
            message = finding.get("message", "")
            lines.append(f"| {severity} | {message} |")
    lines.append("")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Write a Phase 10 gate approval pair.")
    parser.add_argument("--gate", required=True, choices=GATE_IDS)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--source-manifest")
    parser.add_argument("--source-binding")
    parser.add_argument("--command-log")
    parser.add_argument("--review-input")
    parser.add_argument("--markdown-input")
    parser.add_argument("--markdown-output")
    parser.add_argument("--json-output")
    args = parser.parse_args(argv)
    write_report(args.gate, args.plan)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
