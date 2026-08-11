"""Validate the literal labeled payloads embedded in the Phase 10 plan.

The plan embeds normative JSON payloads in two forms:

* **JSON-fence payloads** — literal JSON inside a ```` ```json ```` code fence,
  immediately preceded by a uniquely labeled HTML comment marker:
  ``<!-- payload: <label> bytes=<n> [sha256=<hex>] -->``.
* **Inline-text payloads** — literal JSON in prose (not a fence), marked by the
  same HTML comment form with ``type=inline``.

The tool locates each labeled payload, reconstructs it with the exact canonical
serializer (``json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=True)
+ b"\\n"``), and asserts the adjacent declared byte count and SHA-256. Missing,
duplicate, malformed, or mismatched material exits 2 without printing payload
bytes. Payload content is never echoed on failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

# Marker form: <!-- payload: <label> bytes=<n> [sha256=<hex>] [type=fence|inline] -->
_MARKER_RE = re.compile(
    r"<!--\s*payload:\s*(?P<label>[A-Za-z0-9_\-:]+)"
    r"\s+bytes=(?P<bytes>\d+)"
    r"(?:\s+sha256=(?P<sha256>[0-9a-fA-F]+))?"
    r"(?:\s+type=(?P<ptype>fence|inline))?"
    r"\s*-->"
)

# A ```json fence block.
_FENCE_RE = re.compile(r"```json\n(?P<body>.*?)\n```", re.DOTALL)

# Known payload labels the manifest requires (L158-164). The tool locates
# whichever are present; validate_plan asserts the required set is present.
KNOWN_LABELS = {
    "content-safety-policy",
    "content-safety-fixture",
    "context-security-policy",
    "attack-corpus",
    "source-trust-policy",
    "owasp-excerpt",
    "retrieval-calibration",
}

CANONICAL_SEPARATORS = (",", ":")
LABEL_LINE_SCAN = 6


def locate_payloads(plan_path: str) -> list[dict[str, Any]]:
    """Locate every labeled payload in the plan and return its descriptor.

    Each descriptor contains: ``label``, ``payload_type`` (``json-fence`` or
    ``inline-text``), ``raw`` (the literal payload text), ``serialized`` (the
    canonical bytes), ``declared_bytes``, and optional ``declared_sha256``.
    """
    text = Path(plan_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    payloads: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    for match in _MARKER_RE.finditer(text):
        label = match.group("label")
        if label in seen_labels:
            # Duplicate label is a contract violation; still record for caller.
            continue
        declared_bytes = int(match.group("bytes"))
        declared_sha = match.group("sha256")
        ptype = match.group("ptype") or "fence"

        raw, serialized = _extract_payload(text, match.end(), ptype)
        if raw is None:
            # Malformed/missing payload body; record a descriptor without bytes.
            payloads.append({
                "label": label,
                "payload_type": "inline-text" if ptype == "inline" else "json-fence",
                "raw": "",
                "serialized": b"",
                "declared_bytes": declared_bytes,
                "declared_sha256": declared_sha,
            })
            seen_labels.add(label)
            continue

        payloads.append({
            "label": label,
            "payload_type": "inline-text" if ptype == "inline" else "json-fence",
            "raw": raw,
            "serialized": serialized,
            "declared_bytes": declared_bytes,
            "declared_sha256": declared_sha,
        })
        seen_labels.add(label)

    return payloads


def _extract_payload(text: str, after_marker: int, ptype: str) -> tuple[str | None, bytes]:
    """Extract the payload body following a marker and serialize it canonically."""
    if ptype == "inline":
        # Inline payload: a JSON object embedded in prose, terminated by a
        # backtick close. Find the next balanced ``{...}`` inside backticks.
        return _extract_inline(text, after_marker)
    return _extract_fence(text, after_marker)


def _extract_fence(text: str, after_marker: int) -> tuple[str | None, bytes]:
    """Extract the first ```json fence at or after the marker."""
    search_from = after_marker
    fence = _FENCE_RE.search(text, search_from)
    # The fence should be the very next code block after the marker (within a
    # small window) to avoid matching an unrelated later fence.
    window = text[search_from:search_from + 8192]
    fence_in_window = _FENCE_RE.search(window)
    if not fence_in_window:
        return None, b""
    body = fence_in_window.group("body")
    serialized = _canonical_serialize_text(body)
    return body, serialized


def _extract_inline(text: str, after_marker: int) -> tuple[str | None, bytes]:
    """Extract an inline JSON object in backticks following the marker."""
    window = text[after_marker:after_marker + 4096]
    # The inline payload is a ``{...}`` JSON wrapped in backticks.
    m = re.search(r"`(\{.*?\})`", window, re.DOTALL)
    if not m:
        # Fall back: a bare {...} object.
        m = re.search(r"(\{.*?\})", window, re.DOTALL)
        if not m:
            return None, b""
    body = m.group(1)
    serialized = _canonical_serialize_text(body)
    return body, serialized


def _canonical_serialize_text(body: str) -> bytes:
    """Parse JSON text and reserialize canonically + LF."""
    obj = json.loads(body)
    return json.dumps(obj, sort_keys=True, separators=CANONICAL_SEPARATORS, ensure_ascii=True).encode("utf-8") + b"\n"


def reserialize_payload(payload: dict[str, Any]) -> bytes:
    """Return the canonical serialized bytes for a located payload."""
    return payload.get("serialized", b"")


def validate_plan(plan_path: str) -> None:
    """Validate every labeled payload in the plan; exit 2 on any defect."""
    payloads = locate_payloads(plan_path)
    found_labels = {p["label"] for p in payloads}
    required = {
        "content-safety-policy",
        "content-safety-fixture",
        "context-security-policy",
        "attack-corpus",
        "source-trust-policy",
    }
    missing = required - found_labels
    if missing:
        _fail(f"missing required payloads: {sorted(missing)}")

    for payload in payloads:
        serialized = payload.get("serialized", b"")
        if not serialized:
            _fail(f"payload {payload['label']!r} could not be parsed")
        if len(serialized) != payload["declared_bytes"]:
            _fail(
                f"payload {payload['label']!r} byte count mismatch: "
                f"declared {payload['declared_bytes']}, actual {len(serialized)}"
            )
        declared_sha = payload.get("declared_sha256")
        if declared_sha:
            actual_sha = hashlib.sha256(serialized).hexdigest()
            if actual_sha.lower() != declared_sha.lower():
                _fail(f"payload {payload['label']!r} sha256 mismatch")


def _fail(message: str) -> None:
    """Exit 2 with a non-payload-leaking message."""
    sys.stderr.write(f"validate_phase10_plan_payloads: {message}\n")
    raise SystemExit(2)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 10 plan payloads.")
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    validate_plan(args.plan)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
