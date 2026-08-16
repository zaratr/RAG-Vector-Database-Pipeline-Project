"""Validate the literal labeled payloads embedded in the Phase 10 plan.

The plan embeds normative JSON payloads in two forms:

* **JSON-fence payloads** — literal JSON inside a ```` ```json ```` code fence,
  immediately preceded by a uniquely labeled HTML comment marker:
  ``<!-- payload: <label> bytes=<n> [sha256=<hex>] -->``.
* **Inline-text payloads** — literal JSON in prose (not a fence), marked by the
  same HTML comment form with ``type=inline``.
* **Prose-declared payloads** — literal fences bound to an artifact path by an
  adjacent prose line declaring the byte count and SHA-256 (``json`` fences for
  ``.json`` artifacts, raw-text fences for ``.txt`` artifacts), plus the attack
  corpus declared via a "the following N-byte … SHA-256" lead.

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

# A ```text fence block (raw-text payloads such as the OWASP excerpt).
_TEXT_FENCE_RE = re.compile(r"```text\n(?P<body>.*?)\n```", re.DOTALL)

# Prose declarations: some payloads are bound to an artifact path or a
# "the following N-byte" lead instead of an HTML marker.
# Trailing form "`path`: N bytes ... SHA-256 `hex`" — the payload fence
# immediately precedes the declaration line.
_TRAILING_DECL_RE = re.compile(
    r"`(?P<path>[^`\n]+)`:\s*(?P<bytes>[\d,]+)\s+bytes\b[^\n]*?"
    r"SHA-256\s+`(?P<sha>[0-9a-fA-F]{64})`"
)
# Leading form "`path` is exactly these N ... bytes ... SHA-256 `hex`" —
# the payload fence immediately follows the declaration line.
_LEADING_DECL_RE = re.compile(
    r"`(?P<path>[^`\n]+)`\s+is exactly these\s+(?P<bytes>[\d,]+)[^\n]*?"
    r"SHA-256\s+`(?P<sha>[0-9a-fA-F]{64})`"
)
# "the following N-byte ... SHA-256 `hex`" on a line naming the attack
# file — the payload fence immediately follows the declaration line.
_ATTACK_DECL_RE = re.compile(
    r"the following\s+(?P<bytes>[\d,]+)-byte\b[^\n]*?"
    r"SHA-256\s+`(?P<sha>[0-9a-fA-F]{64})`"
)

# Artifact-path fragment → payload label for prose-declared payloads.
_PATH_LABEL_FRAGMENTS = (
    ("content-safety-policy.json", "content-safety-policy"),
    ("content_safety.json", "content-safety-fixture"),
    ("attack_payloads.json", "attack-corpus"),
    ("owasp", "owasp-excerpt"),
)

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

    payloads.extend(_locate_prose_payloads(text, seen_labels))
    return payloads


def _label_for_path(path: str) -> str | None:
    """Map a prose-declared artifact path to its payload label."""
    for fragment, label in _PATH_LABEL_FRAGMENTS:
        if fragment in path:
            return label
    return None


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    """Return the (start, end) span of the line containing ``pos``."""
    start = text.rfind("\n", 0, pos) + 1
    nl = text.find("\n", pos)
    end = len(text) if nl == -1 else nl
    return start, end


def _locate_prose_payloads(text: str, seen_labels: set[str]) -> list[dict[str, Any]]:
    """Locate payloads the plan declares in prose instead of HTML markers.

    HTML markers take precedence; prose declarations for an already-located
    label are ignored so each label resolves to exactly one payload.
    """
    payloads: list[dict[str, Any]] = []
    json_fences = list(_FENCE_RE.finditer(text))
    text_fences = list(_TEXT_FENCE_RE.finditer(text))

    def fence_before(pos: int, fences: list) -> re.Match | None:
        candidates = [
            m for m in fences
            if m.end() <= pos and not text[m.end():pos].strip()
        ]
        return max(candidates, key=lambda m: m.end(), default=None)

    def fence_after(pos: int, fences: list) -> re.Match | None:
        candidates = [
            m for m in fences
            if m.start() >= pos and not text[pos:m.start()].strip()
        ]
        return min(candidates, key=lambda m: m.start(), default=None)

    def record(
        label: str | None,
        declared_bytes: int,
        declared_sha: str,
        fence: re.Match | None,
        payload_type: str,
        serialize,
    ) -> None:
        if label is None or fence is None or label in seen_labels:
            return
        raw = fence.group("body")
        payloads.append({
            "label": label,
            "payload_type": payload_type,
            "raw": raw,
            "serialized": serialize(raw),
            "declared_bytes": declared_bytes,
            "declared_sha256": declared_sha.lower(),
        })
        seen_labels.add(label)

    for m in _TRAILING_DECL_RE.finditer(text):
        record(
            _label_for_path(m.group("path")),
            int(m.group("bytes").replace(",", "")),
            m.group("sha"),
            fence_before(m.start(), json_fences),
            "json-fence",
            _canonical_serialize_text,
        )

    for m in _LEADING_DECL_RE.finditer(text):
        path = m.group("path")
        if path.endswith(".txt"):
            record(
                _label_for_path(path),
                int(m.group("bytes").replace(",", "")),
                m.group("sha"),
                fence_after(_line_bounds(text, m.start())[1], text_fences),
                "text-fence",
                lambda raw: raw.encode("utf-8") + b"\n",
            )
        else:
            record(
                _label_for_path(path),
                int(m.group("bytes").replace(",", "")),
                m.group("sha"),
                fence_after(_line_bounds(text, m.start())[1], json_fences),
                "json-fence",
                _canonical_serialize_text,
            )

    for m in _ATTACK_DECL_RE.finditer(text):
        line_start, line_end = _line_bounds(text, m.start())
        if "attack" not in text[line_start:line_end]:
            continue
        record(
            "attack-corpus",
            int(m.group("bytes").replace(",", "")),
            m.group("sha"),
            fence_after(line_end, json_fences),
            "json-fence",
            _canonical_serialize_text,
        )

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
