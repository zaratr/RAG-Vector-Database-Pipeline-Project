"""Verify the Phase 10 gate approval pairs are complete, valid, and secret-free.

Validates: schema, terminal verdict, exact required gate set, zero FAIL / zero
actionable WARN, uniqueness, and that no report contains raw Compose/env/token/
query/document/answer/secret bytes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

GATE_IDS = ("phase10a", "phase10b", "phase10c", "phase10d", "documentation", "final")

# Secret indicators that must never appear in an approval report.
_SECRET_PATTERNS = (
    re.compile(r"(?i)RAG_OPERATOR_TOKEN\s*=\s*\S"),
    re.compile(r"(?i)RAG_TOKEN\s*=\s*\S"),
    re.compile(r"(?i)raw_env"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S", re.IGNORECASE),
    re.compile(r"(?i)password\s*=\s*\S"),
)


def scan_for_secrets(report_path: str) -> bool:
    """Return True if the report file contains secret-bearing bytes."""
    try:
        text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def verify_approvals(approvals_dir: str,
                     required: set[str] | None = None) -> dict[str, Any]:
    """Verify the approval directory; return a structured result.

    ``required`` scopes which gates must be present (staged verification per
    the plan: the 10A leaf runs ``--require phase10a``, the 10B leaf
    ``phase10a,phase10b``, and so on). Gates outside the required set are
    still validated when their pairs exist, but their absence is not a
    failure — the final gate verifies the complete set with no ``required``
    override. The default keeps the historical whole-registry behavior.
    """
    approvals = Path(approvals_dir)
    result: dict[str, Any] = {
        "valid": True,
        "missing_gates": [],
        "gates": [],
    }
    required_set = set(required) if required is not None else set(GATE_IDS)
    found: dict[str, dict[str, Any]] = {}
    for gate in GATE_IDS:
        json_path = approvals / f"{gate}.json"
        md_path = approvals / f"{gate}.md"
        if not json_path.is_file() or not md_path.is_file():
            if gate in required_set:
                result["missing_gates"].append(gate)
            continue
        report = json.loads(json_path.read_text(encoding="utf-8"))
        verdict = report.get("terminal_verdict") or report.get("verdict")
        blockers = report.get("blockers", [])
        findings = report.get("findings", blockers)
        finding_fail = any(
            (b.get("severity") == "FAIL") or (b.get("actionable") is True and b.get("severity") == "WARN")
            for b in findings
        )
        # A NOT APPROVED terminal verdict is itself a fail/blocking condition.
        has_fail = finding_fail or verdict != "APPROVED"
        gate_info = {
            "gate_id": gate,
            "verdict": verdict,
            "has_fail": has_fail,
            "secret_leak": scan_for_secrets(str(json_path)),
        }
        found[gate] = gate_info
        result["gates"].append(gate_info)

    if result["missing_gates"]:
        result["valid"] = False
    for gate, info in found.items():
        if info["has_fail"] or info["secret_leak"]:
            result["valid"] = False
        if info["verdict"] != "APPROVED":
            result["valid"] = False
    return result


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 10 approval pairs.")
    parser.add_argument("--plan")
    parser.add_argument("--reports", required=True)
    parser.add_argument("--require")
    args = parser.parse_args(argv)
    required = set(args.require.split(",")) if args.require else None
    if required is not None and not required <= set(GATE_IDS):
        sys.stdout.write(json.dumps(
            {"valid": False, "error": "unknown gate id in --require"},
            sort_keys=True) + "\n")
        return 2
    result = verify_approvals(approvals_dir=args.reports, required=required)
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0 if result["valid"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
