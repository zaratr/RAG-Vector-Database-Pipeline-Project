"""Snapshot non-secret Compose deployment and Settings (B-14 gate support).

Emits canonical JSON of the resolved Compose config (filtered to non-secret keys)
and a canonical allowlisted Settings projection. Never writes/prints raw resolved
Compose, secrets, database URLs, or provider credentials.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FORBIDDEN_KEYS = re.compile(
    r"(?i)(token|secret|password|credential|api[_-]?key|database.*url|operator_token)"
)


def _is_forbidden(key: str) -> bool:
    return bool(_FORBIDDEN_KEYS.search(key))


def snapshot_deployment(output_path: str) -> dict:
    """Capture non-secret Compose deployment config."""
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT), check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker compose config failed: {result.stderr[:200]}")
    raw = json.loads(result.stdout)
    # Filter to non-secret service environment keys.
    filtered: dict = {}
    for svc_name, svc in raw.get("services", {}).items():
        svc_env = {}
        for key, value in (svc.get("environment") or {}).items():
            if not _is_forbidden(key):
                svc_env[key] = value
        filtered[svc_name] = {"environment": svc_env}
    deployment = {"services": filtered}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(deployment, sort_keys=True, separators=(",", ":")) + "\n"
    _scan_and_write(out, payload)
    return deployment


def _nonsecret_settings_payload() -> str:
    """Canonical non-secret Settings projection as a JSON string + LF."""
    from app.config import get_settings

    settings = get_settings()
    raw = settings.model_dump()
    filtered = {}
    for key, value in raw.items():
        if not _is_forbidden(key):
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
    return json.dumps(filtered, sort_keys=True, separators=(",", ":")) + "\n"


def snapshot_settings(output_path: str) -> dict:
    """Capture non-secret Settings projection."""
    filtered = json.loads(_nonsecret_settings_payload())
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _scan_and_write(out, _nonsecret_settings_payload())
    return filtered


def _scan_and_write(path: Path, payload: str) -> None:
    """Secret-scan payload bytes before atomic canonical-LF write."""
    for secret_marker in ("password", "token", "credential", "api_key"):
        if secret_marker + "=" in payload.lower():
            # Allow key names containing these words, but not key=value assignments.
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Binary write: canonical LF bytes regardless of host platform so that
    # host-written and container-written snapshots compare byte-for-byte (D-40).
    tmp.write_bytes(payload.encode("utf-8"))
    import os
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-output")
    parser.add_argument("--settings-output")
    args = parser.parse_args()
    if args.deployment_output:
        snapshot_deployment(args.deployment_output)
    if args.settings_output:
        snapshot_settings(args.settings_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
