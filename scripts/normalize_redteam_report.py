"""Deterministic normalizer for Phase 10D red-team reports.

Produces a canonical, run-independent view of a report JSON: run ID,
timestamps, durations/latency metrics, and host paths are removed while
the fixture hash, versions, decisions, security metrics, and
fingerprints are retained. Task 10D.4 extends this view for the full
cross-phase gate (two runs must normalize byte-identically).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_VOLATILE_KEYS = {
    "run_id",
    "timestamps",
    "started_at",
    "completed_at",
    "duration_ns",
    "latency",
    "latency_ns",
    "host",
    "host_path",
    "output_path",
    # Wall-clock-derived comparison ratios: two live runs of the same
    # immutable image can never byte-equal while these remain.
    "p50_latency_overhead",
    "p95_latency_overhead",
}


def normalize_report(report: dict) -> dict:
    """Return the deterministic view of ``report`` (does not mutate)."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    normalized = {}
    for key, value in report.items():
        if key in _VOLATILE_KEYS:
            continue
        if isinstance(value, dict):
            normalized[key] = normalize_report(value)
        elif isinstance(value, list):
            normalized[key] = [
                normalize_report(item) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str) and (
                value.startswith("/") or "\\" in value):
            # Host paths are environment-specific; keep only basenames.
            normalized[key] = Path(value).name
        else:
            normalized[key] = value
    return normalized


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize a red-team report deterministically.")
    parser.add_argument("report", help="path to the report JSON file")
    parser.add_argument("--output", default=None,
                        help="optional output path (default: stdout)")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    normalized = normalize_report(payload)
    canonical = json.dumps(normalized, sort_keys=True,
                           separators=(",", ":")) + "\n"
    if args.output:
        Path(args.output).write_text(canonical, encoding="utf-8")
    else:
        sys.stdout.write(canonical)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
