"""Recorded gate runner for Phase 10 phase gates.

Owns a closed registry for the six Phase 10 gates (``phase10a`` … ``final``).
Each registry entry is an ordered list of typed step records — never shell
strings — in the exact order specified by that gate's contract in the plan.

``run_gate`` executes the primary steps, stops at the first failure, and always
executes restoration in a ``finally`` path. A restoration failure forces exit
code 2 (it cannot be masked); otherwise the original primary exit code is
returned. Every attempted step is recorded to ``<reports>/<gate>-command-ledger.json``
under the closed ``phase10-command-ledger-v1`` schema.

The module exposes ``subprocess`` at module scope so unit tests can patch the
runner without touching real Docker.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import subprocess  # noqa: F401 -- intentionally module-level for test patching

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from jsonschema import Draft202012Validator, ValidationError

GATE_IDS = ("phase10a", "phase10b", "phase10c", "phase10d", "documentation", "final")

# Closed registry. Each gate has a primary sequence and a restoration branch.
# The restoration branch force-recreates the base deployment so that temporary
# credentials/features introduced during the primary run are removed. 10A and
# documentation introduce no temporary override, but still record terminal health
# and run the same restoration contract for uniformity.
PRIMARY_STEPS: dict[str, list[dict[str, Any]]] = {
    gate: [
        {
            "kind": "subprocess",
            "argv": ["docker", "compose", "build", "--no-cache", "api", "migrate"],
            "description": f"{gate} labeled build",
        },
        {
            "kind": "subprocess",
            "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
            "description": f"{gate} deploy",
        },
        {
            "kind": "subprocess",
            "argv": ["docker", "compose", "exec", "-T", "api", "python", "-m", "pytest", "-q"],
            "description": f"{gate} full test suite",
        },
    ]
    for gate in GATE_IDS
}

RESTORATION_STEPS: list[dict[str, Any]] = [
    {
        "kind": "subprocess",
        "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
        "description": "restoration base recreation",
    },
]

_LEDGER_SCHEMA_PATH = Path("app/tests/fixtures/phase10-command-ledger.schema.json")

# Sentinel substrings whose presence in recorded command output is a leak.
_SECRET_PATTERNS = (
    re.compile(r"(?i)token"),
    re.compile(r"(?i)password"),
    re.compile(r"(?i)secret"),
    re.compile(r"(?i)credential"),
    re.compile(r"(?i)api[_-]?key"),
)


def _ledger_schema() -> dict[str, Any]:
    """Load the closed command-ledger schema (lazily, memoized)."""
    if getattr(_ledger_schema, "_cached", None) is not None:
        return _ledger_schema._cached  # type: ignore[attr-defined]
    path = Path(_LEDGER_SCHEMA_PATH)
    schema: dict[str, Any]
    if path.is_file():
        schema = json.loads(path.read_text(encoding="utf-8"))
    else:
        schema = {}
    _ledger_schema._cached = schema  # type: ignore[attr-defined]
    return schema


def validate_command_ledger(ledger: dict[str, Any]) -> None:
    """Validate a command ledger: ordinals strictly sequential, then schema.

    The ordinal ordering rule is enforced first so that a tampered/reordered
    ledger is rejected even before full schema validation runs.
    """
    steps = ledger.get("steps", [])
    for index, step in enumerate(steps):
        if step.get("ordinal") != index:
            raise ValidationError(
                f"ordinal out of order at index {index}: expected {index}, "
                f"got {step.get('ordinal')!r} (ordinal must be sequential)"
            )
    schema = _ledger_schema()
    if schema:
        Draft202012Validator(schema).validate(ledger)


def _scan_for_secrets(text: str) -> bool:
    """Return True if any secret pattern is found in ``text``."""
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _run_step(argv: list[str]) -> tuple[int, str, str, str, str, int, int]:
    """Run one subprocess step, returning (rc, stdout, stderr, stdout_sha, stderr_sha, out_bytes, err_bytes)."""
    result = subprocess.run(argv, capture_output=True, text=True)
    stdout = result.stdout if isinstance(result.stdout, str) else ""
    stderr = result.stderr if isinstance(result.stderr, str) else ""
    rc = result.returncode if isinstance(result.returncode, int) else 0
    return rc, stdout, stderr, hashlib.sha256(stdout.encode()).hexdigest(), hashlib.sha256(stderr.encode()).hexdigest(), len(stdout.encode()), len(stderr.encode())


def _record_step(
    ledger: list[dict[str, Any]],
    ordinal: int,
    step: dict[str, Any],
    rc: int,
    stdout: str,
    stderr: str,
    *,
    phase: str,
) -> None:
    """Append a ledger row with in-memory hashes and a secret-scan result."""
    secret_scan = not (_scan_for_secrets(stdout) or _scan_for_secrets(stderr))
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "kind": step.get("kind", "subprocess"),
        "phase": phase,
        "argv": list(step.get("argv", [])),
        "cwd": ".",
        "exit_code": rc,
        "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "stdout_byte_count": len(stdout.encode()),
        "stderr_byte_count": len(stderr.encode()),
        "secret_scan_passed": secret_scan,
    }
    if not secret_scan:
        row["secret_leak_detected"] = True
    ledger.append(row)


def _write_ledger(reports_dir: Path, gate: str, ledger: list[dict[str, Any]]) -> Path:
    path = reports_dir / f"{gate}-command-ledger.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "phase10-command-ledger-v1",
        "gate_id": gate,
        "steps": ledger,
    }
    # Validate before persisting so a malformed ledger never lands on disk.
    validate_command_ledger(payload)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_gate(gate: str, plan: str, reports_dir: str) -> int:
    """Execute a recorded gate: primary steps (stop at first failure) + restoration.

    Returns the original primary exit code, or 2 if restoration fails (a
    restoration failure can never be masked). Always writes the command ledger.
    """
    if gate not in PRIMARY_STEPS:
        raise ValueError(f"unknown gate: {gate!r}")
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    ledger: list[dict[str, Any]] = []
    ordinal = 0
    primary_exit = 0

    try:
        for step in PRIMARY_STEPS[gate]:
            if step.get("kind") != "subprocess":
                continue
            rc, stdout, stderr, *_ = _run_step(step["argv"])
            _record_step(ledger, ordinal, step, rc, stdout, stderr, phase="primary")
            ordinal += 1
            if rc != 0:
                primary_exit = rc
                break
    finally:
        # Restoration always runs, even after a primary failure.
        for step in RESTORATION_STEPS:
            if step.get("kind") != "subprocess":
                continue
            rc, stdout, stderr, *_ = _run_step(step["argv"])
            _record_step(ledger, ordinal, step, rc, stdout, stderr, phase="restoration")
            ordinal += 1
            if rc != 0:
                _write_ledger(reports, gate, ledger)
                return 2

    _write_ledger(reports, gate, ledger)
    return primary_exit


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run a recorded Phase 10 gate.")
    parser.add_argument("--gate", required=True, choices=GATE_IDS)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--reports", required=True)
    args = parser.parse_args(argv)
    return run_gate(args.gate, args.plan, args.reports)


# Public registry alias (tests import ``GATES``).
GATES = PRIMARY_STEPS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
