"""Snapshot non-secret Settings from running API (B-14 gate support).

Wraps snapshot_nonsecret_deployment.snapshot_settings for standalone use.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.snapshot_nonsecret_deployment import snapshot_settings


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    # Absent or "-" emits the canonical JSON payload on stdout so the typed
    # gate command (`... > report.json`) works as written (D-39).
    parser.add_argument("--settings-output", default=None)
    args = parser.parse_args()
    if not args.settings_output or args.settings_output == "-":
        from scripts.snapshot_nonsecret_deployment import _nonsecret_settings_payload

        # Bytes on stdout: text-mode stdout on Windows would translate the
        # canonical LF to CRLF and break byte-for-byte gate comparisons (D-40).
        sys.stdout.buffer.write(_nonsecret_settings_payload().encode("utf-8"))
        sys.stdout.buffer.flush()
        return 0
    snapshot_settings(args.settings_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
