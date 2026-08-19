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

# Typed phase10b sequence exactly as the plan's 10B task gate specifies it.
# Scoped environment values are applied in-memory (never argv/ledger strings)
# via the "phase10b_scoped_env" step kind. Steps may carry "stdout_to" to
# redirect captured stdout into a reports file, standing in for `>` shells.
PRIMARY_STEPS["phase10b"] = [
    {"kind": "subprocess", "argv": ["python", "scripts/source_manifest.py",
     "--output", ".hermes/reports/phase10b-source-manifest.json"],
     "description": "phase10b source manifest"},
    {"kind": "subprocess", "argv": ["python", "scripts/snapshot_nonsecret_deployment.py",
     "--deployment-output", ".hermes/reports/phase10b-base-deployment.json",
     "--settings-output", ".hermes/reports/phase10b-base-expected-settings.json"],
     "description": "phase10b base non-secret deployment/settings snapshot"},
    {"kind": "phase10b_scoped_env",
     "description": "phase10b in-memory scoped operator/source env (no values recorded)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "config", "--quiet"],
     "description": "compose config validation"},
    {"kind": "subprocess", "argv": ["docker", "compose", "build", "--no-cache", "api", "migrate"],
     "description": "phase10b labeled build"},
    {"kind": "subprocess", "argv": ["python", "scripts/create_phase10_source_binding.py",
     "--manifest", ".hermes/reports/phase10b-source-manifest.json",
     "--output", ".hermes/reports/phase10b-source-binding.json",
     "--services", "api", "migrate"],
     "description": "phase10b source binding"},
    {"kind": "subprocess", "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
     "description": "phase10b deploy (operator env)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python", "-c",
     "from app.config import get_settings; s=get_settings(); assert s.operator_api_enabled; "
     "print({'operator_api_enabled':True,'source_trust_policy_path':str(s.source_trust_policy_path)})"],
     "description": "operator settings assert in container"},
    {"kind": "subprocess", "argv": ["docker", "compose", "run", "--rm", "migrate"],
     "description": "migrate one-shot"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api",
     "python", "-m", "app.core.migrations"],
     "description": "in-container migrations wrapper"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "alembic", "-c", "alembic.ini", "current"],
     "description": "alembic current"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "alembic", "-c", "alembic.ini", "check"],
     "description": "alembic check"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/fingerprint_production_state.py", "--json"],
     "stdout_to": "phase10b-state-before.json",
     "description": "production state fingerprint (before)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "pytest", "-q"],
     "description": "phase10b full test suite"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10b.py"],
     "description": "phase10b live validator"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10a.py"],
     "description": "phase10a regression validator"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/fingerprint_production_state.py", "--json"],
     "stdout_to": "phase10b-state-after.json",
     "description": "production state fingerprint (after)"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10b-state-before.json",
     ".hermes/reports/phase10b-state-after.json"],
     "description": "state equality"},
    {"kind": "curl_status", "url": "http://127.0.0.1:8000/security/audits/00000000-0000-0000-0000-000000000000",
     "expected_status": 401, "headers_to": "phase10b-missing-auth-headers.txt",
     "description": "missing-auth 401"},
    {"kind": "curl_status", "url": "http://127.0.0.1:8000/security/audits/00000000-0000-0000-0000-000000000000",
     "expected_status": 401, "headers_to": "phase10b-invalid-auth-headers.txt",
     "bearer": "invalid-phase10-token",
     "description": "invalid-auth 401"},
    {"kind": "file_contains_lower", "path": ".hermes/reports/phase10b-missing-auth-headers.txt",
     "substrings": ["www-authenticate: bearer"],
     "description": "missing-auth bearer challenge header"},
    {"kind": "file_contains_lower", "path": ".hermes/reports/phase10b-invalid-auth-headers.txt",
     "substrings": ["www-authenticate: bearer"],
     "description": "invalid-auth bearer challenge header"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "-e", "OPERATOR_BEARER",
     "api", "python", "scripts/probe_operator_auth.py", "--assert-authorized",
     "--expected-status", "404", "--assertion-id", "phase10b-operator-valid-auth"],
     "pass_env": ["OPERATOR_BEARER"],
     "description": "B-12 valid-auth probe (credential only via env, never argv)"},
]

# Typed phase10c sequence exactly as the plan's 10C task gate specifies it.
# Scoped environment values (operator + content-safety) are applied in-memory
# (never argv/ledger strings) via the "phase10c_scoped_env" step kind.
PRIMARY_STEPS["phase10c"] = [
    {"kind": "subprocess", "argv": ["python", "scripts/source_manifest.py",
     "--output", ".hermes/reports/phase10c-source-manifest.json"],
     "description": "phase10c source manifest"},
    {"kind": "subprocess", "argv": ["python", "scripts/snapshot_nonsecret_deployment.py",
     "--deployment-output", ".hermes/reports/phase10c-base-deployment.json",
     "--settings-output", ".hermes/reports/phase10c-base-expected-settings.json"],
     "description": "phase10c base non-secret deployment/settings snapshot"},
    {"kind": "phase10c_scoped_env",
     "description": "phase10c in-memory scoped operator/content-safety/source env (no values recorded)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "config", "--quiet"],
     "description": "compose config validation"},
    {"kind": "subprocess", "argv": ["docker", "compose", "build", "--no-cache", "api", "migrate"],
     "description": "phase10c labeled build"},
    {"kind": "subprocess", "argv": ["python", "scripts/create_phase10_source_binding.py",
     "--manifest", ".hermes/reports/phase10c-source-manifest.json",
     "--output", ".hermes/reports/phase10c-source-binding.json",
     "--services", "api", "migrate"],
     "description": "phase10c source binding"},
    {"kind": "subprocess", "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
     "description": "phase10c deploy (operator + content-safety env)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python", "-c",
     "from app.config import get_settings; s=get_settings(); assert s.operator_api_enabled and s.content_safety_enabled and s.safety_llm_mode == 'rules_only'; "
     "print({'operator_api_enabled':True,'content_safety_enabled':True,'safety_llm_mode':'rules_only'})"],
     "description": "content-safety settings assert in container"},
    {"kind": "subprocess", "argv": ["docker", "compose", "run", "--rm", "migrate"],
     "description": "migrate one-shot"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api",
     "python", "-m", "app.core.migrations"],
     "description": "in-container migrations wrapper"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "alembic", "-c", "alembic.ini", "current"],
     "description": "alembic current"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "alembic", "-c", "alembic.ini", "check"],
     "description": "alembic check"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/fingerprint_production_state.py", "--json"],
     "stdout_to": "phase10c-state-before.json",
     "description": "production state fingerprint (before)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "pytest", "-q"],
     "description": "phase10c full test suite"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10c.py"],
     "description": "phase10c live validator"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10b.py"],
     "description": "phase10b regression validator"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10a.py"],
     "description": "phase10a regression validator"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/fingerprint_production_state.py", "--json"],
     "stdout_to": "phase10c-state-after.json",
     "description": "production state fingerprint (after)"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10c-state-before.json",
     ".hermes/reports/phase10c-state-after.json"],
     "description": "state equality"},
]

# phase10b restoration: destroy the in-memory scoped credential/override, then
# force-recreate under the exact pre-gate environment and assert health.
RESTORATION_STEPS: list[dict[str, Any]] = [
    {
        "kind": "subprocess",
        "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
        "description": "restoration base recreation",
    },
]

PHASE10B_RESTORATION_STEPS: list[dict[str, Any]] = [
    {"kind": "phase10b_unscoped_env",
     "description": "destroy in-memory scoped operator/source env"},
    {
        "kind": "subprocess",
        "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
        "description": "restoration base recreation (default env)",
    },
    # The plan places the restored snapshots and their comparisons OUTSIDE the
    # scoped subshell: they must run after the override is destroyed, or the
    # scoped keys (e.g. RAG_OPERATOR_API_ENABLED) would leak into the
    # "restored" snapshot and the base-vs-restored cmp could never hold (D-50).
    {"kind": "subprocess", "argv": ["python", "scripts/snapshot_nonsecret_deployment.py",
     "--deployment-output", ".hermes/reports/phase10b-restored-deployment.json",
     "--settings-output", ".hermes/reports/phase10b-restored-expected-settings.json"],
     "description": "restored non-secret deployment/settings snapshot (default env)"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10b-base-deployment.json",
     ".hermes/reports/phase10b-restored-deployment.json"],
     "description": "base/restored deployment equality"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10b-base-expected-settings.json",
     ".hermes/reports/phase10b-restored-expected-settings.json"],
     "description": "base/restored expected-settings equality"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api",
     "python", "scripts/snapshot_nonsecret_settings.py"],
     "stdout_to": "phase10b-restored-running-settings.json",
     "description": "restored running settings"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10b-base-expected-settings.json",
     ".hermes/reports/phase10b-restored-running-settings.json"],
     "description": "expected vs running settings equality"},
    {"kind": "heartbeat", "url": "http://127.0.0.1:8000/",
     "attempts": 30, "interval_seconds": 2,
     "description": "api heartbeat (readiness-polled)"},
    {"kind": "heartbeat", "url": "http://127.0.0.1:8001/api/v2/heartbeat",
     "attempts": 30, "interval_seconds": 2,
     "description": "chroma heartbeat (readiness-polled)"},
]

PHASE10C_RESTORATION_STEPS: list[dict[str, Any]] = [
    {"kind": "phase10c_unscoped_env",
     "description": "destroy in-memory scoped operator/content-safety/source env"},
    {"kind": "subprocess", "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
     "description": "restoration base recreation (default env)"},
    # Restored snapshots and comparisons run AFTER the override is destroyed,
    # so scoped keys (e.g. RAG_CONTENT_SAFETY_ENABLED) cannot leak into the
    # "restored" snapshot and the base-vs-restored cmp can hold (D-50).
    {"kind": "subprocess", "argv": ["python", "scripts/snapshot_nonsecret_deployment.py",
     "--deployment-output", ".hermes/reports/phase10c-restored-deployment.json",
     "--settings-output", ".hermes/reports/phase10c-restored-expected-settings.json"],
     "description": "restored non-secret deployment/settings snapshot (default env)"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10c-base-deployment.json",
     ".hermes/reports/phase10c-restored-deployment.json"],
     "description": "base/restored deployment equality"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10c-base-expected-settings.json",
     ".hermes/reports/phase10c-restored-expected-settings.json"],
     "description": "base/restored expected-settings equality"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api",
     "python", "scripts/snapshot_nonsecret_settings.py"],
     "stdout_to": "phase10c-restored-running-settings.json",
     "description": "restored running settings"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10c-base-expected-settings.json",
     ".hermes/reports/phase10c-restored-running-settings.json"],
     "description": "expected vs running settings equality"},
    {"kind": "heartbeat", "url": "http://127.0.0.1:8000/",
     "attempts": 30, "interval_seconds": 2,
     "description": "api heartbeat (readiness-polled)"},
    {"kind": "heartbeat", "url": "http://127.0.0.1:8001/api/v2/heartbeat",
     "attempts": 30, "interval_seconds": 2,
     "description": "chroma heartbeat (readiness-polled)"},
]

# Typed phase10d sequence exactly as the plan's 10D task gate specifies it:
# prior-phase validators in a/b/c order, then the 10D gate orchestrator and
# named-volume durability validator as HOST python steps, then state equality.
PRIMARY_STEPS["phase10d"] = [
    {"kind": "subprocess", "argv": ["python", "scripts/source_manifest.py",
     "--output", ".hermes/reports/phase10d-source-manifest.json"],
     "description": "phase10d source manifest"},
    {"kind": "subprocess", "argv": ["python", "scripts/snapshot_nonsecret_deployment.py",
     "--deployment-output", ".hermes/reports/phase10d-base-deployment.json",
     "--settings-output", ".hermes/reports/phase10d-base-expected-settings.json"],
     "description": "phase10d base non-secret deployment/settings snapshot"},
    {"kind": "phase10d_scoped_env",
     "description": "phase10d in-memory scoped operator/content-safety/source env (no values recorded)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "config", "--quiet"],
     "description": "compose config validation"},
    {"kind": "subprocess", "argv": ["docker", "compose", "build", "--no-cache", "api", "migrate"],
     "description": "phase10d labeled build"},
    {"kind": "subprocess", "argv": ["python", "scripts/create_phase10_source_binding.py",
     "--manifest", ".hermes/reports/phase10d-source-manifest.json",
     "--output", ".hermes/reports/phase10d-source-binding.json",
     "--services", "api", "migrate"],
     "description": "phase10d source binding"},
    {"kind": "subprocess", "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
     "description": "phase10d deploy (operator + content-safety env)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python", "-c",
     "from app.config import get_settings; s=get_settings(); assert s.operator_api_enabled and s.content_safety_enabled and s.safety_llm_mode == 'rules_only'; "
     "print({'operator_api_enabled':True,'content_safety_enabled':True,'safety_llm_mode':'rules_only'})"],
     "description": "content-safety settings assert in container"},
    {"kind": "subprocess", "argv": ["docker", "compose", "run", "--rm", "migrate"],
     "description": "migrate one-shot"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api",
     "python", "-m", "app.core.migrations"],
     "description": "in-container migrations wrapper"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "alembic", "-c", "alembic.ini", "current"],
     "description": "alembic current"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "alembic", "-c", "alembic.ini", "check"],
     "description": "alembic check"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/fingerprint_production_state.py", "--json"],
     "stdout_to": "phase10d-state-before.json",
     "description": "production state fingerprint (before)"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "pytest", "-q"],
     "description": "phase10d full test suite"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10a.py"],
     "description": "phase10a regression validator"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10b.py"],
     "description": "phase10b regression validator"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/validate_phase10c.py"],
     "description": "phase10c regression validator"},
    {"kind": "subprocess", "argv": ["python", "scripts/run_phase10d_gate.py",
     "--output", ".hermes/reports",
     "--source-manifest", ".hermes/reports/phase10d-source-manifest.json",
     "--source-binding", ".hermes/reports/phase10d-source-binding.json"],
     "description": "phase10d recorded red-team gate (two runs + validation + normalization)"},
    {"kind": "subprocess", "argv": ["python", "scripts/validate_named_volume_durability.py",
     "--output", ".hermes/reports/phase10d-durability.json"],
     "description": "named-volume durability validation"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "scripts/fingerprint_production_state.py", "--json"],
     "stdout_to": "phase10d-state-after.json",
     "description": "production state fingerprint (after)"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10d-state-before.json",
     ".hermes/reports/phase10d-state-after.json"],
     "description": "state equality"},
]

PHASE10D_RESTORATION_STEPS: list[dict[str, Any]] = [
    {"kind": "phase10d_unscoped_env",
     "description": "destroy in-memory scoped operator/content-safety/source env"},
    {"kind": "subprocess", "argv": ["docker", "compose", "up", "-d", "--force-recreate"],
     "description": "restoration base recreation (default env)"},
    {"kind": "subprocess", "argv": ["python", "scripts/snapshot_nonsecret_deployment.py",
     "--deployment-output", ".hermes/reports/phase10d-restored-deployment.json",
     "--settings-output", ".hermes/reports/phase10d-restored-expected-settings.json"],
     "description": "restored non-secret deployment/settings snapshot (default env)"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10d-base-deployment.json",
     ".hermes/reports/phase10d-restored-deployment.json"],
     "description": "base/restored deployment equality"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10d-base-expected-settings.json",
     ".hermes/reports/phase10d-restored-expected-settings.json"],
     "description": "base/restored expected-settings equality"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api",
     "python", "scripts/snapshot_nonsecret_settings.py"],
     "stdout_to": "phase10d-restored-running-settings.json",
     "description": "restored running settings"},
    {"kind": "subprocess", "argv": ["cmp", ".hermes/reports/phase10d-base-expected-settings.json",
     ".hermes/reports/phase10d-restored-running-settings.json"],
     "description": "expected vs running settings equality"},
    {"kind": "subprocess", "argv": ["docker", "compose", "exec", "-T", "api", "python",
     "-m", "alembic", "-c", "alembic.ini", "current"],
     "description": "alembic current (restored)"},
    {"kind": "heartbeat", "url": "http://127.0.0.1:8000/",
     "attempts": 30, "interval_seconds": 2,
     "description": "api heartbeat (readiness-polled)"},
    {"kind": "heartbeat", "url": "http://127.0.0.1:8001/api/v2/heartbeat",
     "attempts": 30, "interval_seconds": 2,
     "description": "chroma heartbeat (readiness-polled)"},
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
    """Run one subprocess step, returning (rc, stdout, stderr, stdout_sha, stderr_sha, out_bytes, err_bytes).

    Host-side python steps always use ``sys.executable``: a bare ``python``
    resolves through PATH/CreateProcess and may land on a dependency-less
    interpreter when the runner itself executes inside a virtualenv (D-46).
    """
    resolved = list(argv)
    if resolved and resolved[0] == "python":
        resolved[0] = sys.executable
    result = subprocess.run(resolved, capture_output=True, text=True)
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
    """Append a ledger row with in-memory hashes and a secret-scan result.

    Rich step kinds (scoped-env, curl probes, file assertions) record under
    the closed schema's ``internal`` kind with a symbolic argv; only real
    subprocesses are recorded as ``subprocess`` (D-45).
    """
    secret_scan = not (_scan_for_secrets(stdout) or _scan_for_secrets(stderr))
    step_kind = step.get("kind", "subprocess")
    ledger_kind = step_kind if step_kind == "subprocess" else "internal"
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "kind": ledger_kind,
        "phase": phase,
        "argv": _step_argv_for_ledger(step),
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


_PHASE10B_SCOPED_KEYS = (
    "RAG_OPERATOR_API_ENABLED", "RAG_OPERATOR_TOKEN", "OPERATOR_BEARER",
    "SOURCE_REVISION", "SOURCE_CONTEXT_SHA256", "SOURCE_DIRTY",
)

_PHASE10C_SCOPED_KEYS = (
    "RAG_OPERATOR_API_ENABLED", "RAG_OPERATOR_TOKEN", "OPERATOR_BEARER",
    "RAG_CONTENT_SAFETY_ENABLED", "RAG_SAFETY_LLM_MODE",
    "SOURCE_REVISION", "SOURCE_CONTEXT_SHA256", "SOURCE_DIRTY",
)

# The phase10d registry's scoped env is the same operator/content-safety/
# source set as phase10c (plan lines 2013-2022); the manifest source differs.
_PHASE10D_SCOPED_KEYS = _PHASE10C_SCOPED_KEYS


def _phase10b_scoped_env(reports: Path) -> tuple[int, str, str]:
    """Apply the typed scoped env in-memory only (never argv/ledger strings).

    The operator bearer is a fresh token_urlsafe(32); the SOURCE_* values come
    from the phase10b source manifest produced earlier in the sequence.
    """
    import os
    import secrets

    manifest_path = reports / "phase10b-source-manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bearer = secrets.token_urlsafe(32)
    os.environ["RAG_OPERATOR_API_ENABLED"] = "true"
    os.environ["RAG_OPERATOR_TOKEN"] = bearer
    os.environ["OPERATOR_BEARER"] = bearer
    if manifest.get("commit_sha"):
        os.environ["SOURCE_REVISION"] = str(manifest["commit_sha"])
    if manifest.get("image_context_sha256"):
        os.environ["SOURCE_CONTEXT_SHA256"] = str(manifest["image_context_sha256"])
    if manifest.get("dirty") is not None:
        os.environ["SOURCE_DIRTY"] = str(manifest["dirty"]).lower()
    return 0, "", ""


def _phase10b_unscoped_env() -> tuple[int, str, str]:
    """Destroy the in-memory scoped credential/override (restoration)."""
    import os

    for key in _PHASE10B_SCOPED_KEYS:
        os.environ.pop(key, None)
    return 0, "", ""


def _phase10c_scoped_env(reports: Path) -> tuple[int, str, str]:
    """Apply the typed scoped env in-memory only (never argv/ledger strings).

    The operator bearer is a fresh token_urlsafe(32); the SOURCE_* values come
    from the phase10c source manifest produced earlier in the sequence. 10C
    additionally sets RAG_CONTENT_SAFETY_ENABLED and RAG_SAFETY_LLM_MODE for
    the content-safety feature gate.
    """
    import os
    import secrets

    manifest_path = reports / "phase10c-source-manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bearer = secrets.token_urlsafe(32)
    os.environ["RAG_OPERATOR_API_ENABLED"] = "true"
    os.environ["RAG_OPERATOR_TOKEN"] = bearer
    os.environ["OPERATOR_BEARER"] = bearer
    os.environ["RAG_CONTENT_SAFETY_ENABLED"] = "true"
    os.environ["RAG_SAFETY_LLM_MODE"] = "rules_only"
    if manifest.get("commit_sha"):
        os.environ["SOURCE_REVISION"] = str(manifest["commit_sha"])
    if manifest.get("image_context_sha256"):
        os.environ["SOURCE_CONTEXT_SHA256"] = str(manifest["image_context_sha256"])
    if manifest.get("dirty") is not None:
        os.environ["SOURCE_DIRTY"] = str(manifest["dirty"]).lower()
    return 0, "", ""


def _phase10c_unscoped_env() -> tuple[int, str, str]:
    """Destroy the in-memory scoped operator/content-safety/source override."""
    import os

    for key in _PHASE10C_SCOPED_KEYS:
        os.environ.pop(key, None)
    return 0, "", ""


def _phase10d_scoped_env(reports: Path) -> tuple[int, str, str]:
    """Apply the phase10d scoped env in-memory (10C key set, 10D manifest)."""
    import os
    import secrets

    manifest_path = reports / "phase10d-source-manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bearer = secrets.token_urlsafe(32)
    os.environ["RAG_OPERATOR_API_ENABLED"] = "true"
    os.environ["RAG_OPERATOR_TOKEN"] = bearer
    os.environ["OPERATOR_BEARER"] = bearer
    os.environ["RAG_CONTENT_SAFETY_ENABLED"] = "true"
    os.environ["RAG_SAFETY_LLM_MODE"] = "rules_only"
    if manifest.get("commit_sha"):
        os.environ["SOURCE_REVISION"] = str(manifest["commit_sha"])
    if manifest.get("image_context_sha256"):
        os.environ["SOURCE_CONTEXT_SHA256"] = str(manifest["image_context_sha256"])
    if manifest.get("dirty") is not None:
        os.environ["SOURCE_DIRTY"] = str(manifest["dirty"]).lower()
    return 0, "", ""


def _phase10d_unscoped_env() -> tuple[int, str, str]:
    """Destroy the in-memory phase10d scoped env (restoration)."""
    import os

    for key in _PHASE10D_SCOPED_KEYS:
        os.environ.pop(key, None)
    return 0, "", ""


def _curl_status_step(step: dict[str, Any], reports: Path) -> tuple[int, str, str]:
    """Fetch a URL, capture headers, and compare the HTTP status code."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(step["url"], method="GET")
    bearer = step.get("bearer")
    if bearer:
        req.add_header("Authorization", f"Bearer {bearer}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.getcode()
            headers = "".join(f"{k}: {v}" + chr(10) for k, v in resp.headers.items())
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = "".join(f"{k}: {v}" + chr(10) for k, v in exc.headers.items())
    headers_to = step.get("headers_to")
    if headers_to:
        (reports / headers_to).write_text(headers, encoding="utf-8")
    if status != step["expected_status"]:
        return 1, "", f"status {status} != expected {step['expected_status']}"
    return 0, "", ""


def _heartbeat_step(step: dict[str, Any]) -> tuple[int, str, str]:
    """Poll a heartbeat URL until it answers 2xx, with bounded retries.

    `docker compose up` returns when containers start, not when the HTTP
    listener is ready; a zero-tolerance curl can race uvicorn's startup and
    fail the whole restoration (D-48).
    """
    import time
    import urllib.error
    import urllib.request

    attempts = int(step.get("attempts", 30))
    interval = float(step.get("interval_seconds", 2.0))
    last_error = ""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(step["url"], timeout=5) as resp:
                if 200 <= resp.getcode() < 300:
                    return 0, "", ""
                last_error = f"status {resp.getcode()}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            time.sleep(interval)
    return 1, "", f"heartbeat failed after {attempts} attempts: {last_error}"


def _file_contains_lower_step(step: dict[str, Any]) -> tuple[int, str, str]:
    """Assert each substring is present in the file (case-insensitive)."""
    text = Path(step["path"]).read_text(encoding="utf-8").lower()
    missing = [sub for sub in step["substrings"] if sub.lower() not in text]
    if missing:
        return 1, "", f"missing substrings: {missing}"
    return 0, "", ""


def _run_recorded_step(
    step: dict[str, Any], reports: Path
) -> tuple[int, str, str, str, str, int, int]:
    """Dispatch one typed step; same return shape as ``_run_step``."""
    kind = step.get("kind", "subprocess")
    if kind == "phase10b_scoped_env":
        rc, out, err = _phase10b_scoped_env(reports)
    elif kind == "phase10b_unscoped_env":
        rc, out, err = _phase10b_unscoped_env()
    elif kind == "phase10c_scoped_env":
        rc, out, err = _phase10c_scoped_env(reports)
    elif kind == "phase10c_unscoped_env":
        rc, out, err = _phase10c_unscoped_env()
    elif kind == "phase10d_scoped_env":
        rc, out, err = _phase10d_scoped_env(reports)
    elif kind == "phase10d_unscoped_env":
        rc, out, err = _phase10d_unscoped_env()
    elif kind == "curl_status":
        rc, out, err = _curl_status_step(step, reports)
    elif kind == "file_contains_lower":
        rc, out, err = _file_contains_lower_step(step)
    elif kind == "heartbeat":
        rc, out, err = _heartbeat_step(step)
    else:
        rc, out, err, *_ = _run_step(step["argv"])
        stdout_to = step.get("stdout_to")
        if stdout_to and rc == 0:
            (reports / stdout_to).write_text(out, encoding="utf-8", newline="")
    return (
        rc, out, err,
        hashlib.sha256(out.encode()).hexdigest(),
        hashlib.sha256(err.encode()).hexdigest(),
        len(out.encode()), len(err.encode()),
    )


def _step_argv_for_ledger(step: dict[str, Any]) -> list[str]:
    """Symbolic argv for non-subprocess kinds (never values/env contents)."""
    if "argv" in step:
        return list(step["argv"])
    return [f"<{step.get('kind', 'subprocess')}>"]


def run_gate(gate: str, plan: str, reports_dir: str) -> int:
    """Execute a recorded gate: primary steps (stop at first failure) + restoration.

    Returns the original primary exit code, or 2 if restoration fails (a
    restoration failure can never be masked). Always writes the command ledger.
    """
    if gate not in PRIMARY_STEPS:
        raise ValueError(f"unknown gate: {gate!r}")
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    restoration_steps = (
        PHASE10B_RESTORATION_STEPS if gate == "phase10b"
        else PHASE10C_RESTORATION_STEPS if gate == "phase10c"
        else PHASE10D_RESTORATION_STEPS if gate == "phase10d"
        else RESTORATION_STEPS
    )
    ledger: list[dict[str, Any]] = []
    ordinal = 0
    primary_exit = 0

    try:
        for step in PRIMARY_STEPS[gate]:
            rc, stdout, stderr, *_ = _run_recorded_step(step, reports)
            _record_step(ledger, ordinal, step, rc, stdout, stderr, phase="primary")
            ordinal += 1
            if rc != 0:
                primary_exit = rc
                break
    finally:
        # Restoration always runs, even after a primary failure.
        for step in restoration_steps:
            rc, stdout, stderr, *_ = _run_recorded_step(step, reports)
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
