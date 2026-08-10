"""Canonical source manifest for Phase 10 gate provenance.

Produces two content-addressed domains that are intentionally different and never
compared:

* ``delivery_tree_sha256`` — over tracked files plus non-ignored untracked
  *delivery* files (including this Phase 10 plan under ``.hermes/plans/``).
* ``image_context_sha256`` — over exactly the non-ignored Docker build-context
  files after applying ``.dockerignore``.

Both domains are built from sorted POSIX ``path/status/sha256`` tuples so that
the manifest is deterministic for a given tree. Git status porcelain entries
(deletions, modifications, untracked) are included so that a dirty tree is
distinguished from a clean one. Ignored files and the Phase 10 reports
directory are excluded from both domains.

The manifest JSON is written canonically (sorted keys, ``",:"`` separators,
ASCII) followed by a trailing LF, and is never printed to stdout by the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Schema version pinned by Task 10.0. Bumping requires a plan amendment.
MANIFEST_VERSION = "phase10-source-manifest-v1"

# Fields exposed by ``--input PATH --field NAME``. Each maps to a manifest key
# and a validation regex that the printed scalar must satisfy.
FIELDS = {
    "commit_sha": r"^[0-9a-f]{40}$",
    "dirty": r"^(True|False)$",
    "image_context_sha256": r"^[0-9a-f]{64}$",
}

# The Phase 10 plan is delivery evidence even though it lives beneath .hermes
# (which .dockerignore excludes from the image context). It must be part of the
# delivery tree fingerprint.
PLAN_PATH = Path(".hermes/plans/2026-08-01_094008-phase-10-contract-reassessment-and-implementation.md")

# Reports produced by gates are never delivery evidence and never enter the
# image; they are excluded from both domains.
REPORTS_DIR = Path(".hermes/reports")

REPO_ROOT_HINT_FILES = (".git", "Dockerfile", "requirements.txt", "app")


def _run_git(argv: list[str], cwd: Path) -> str:
    """Run a git command and return decoded stdout (empty string on failure)."""
    try:
        result = subprocess.run(
            ["git", *argv],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _git_porcelain(cwd: Path) -> list[str]:
    """Return ``git status --porcelain`` lines (includes untracked, deleted)."""
    return [
        line
        for line in _run_git(["status", "--porcelain"], cwd).splitlines()
        if line.strip()
    ]


def _git_tracked_files(cwd: Path) -> list[str]:
    """Return the sorted list of tracked file paths (POSIX)."""
    raw = _run_git(["ls-files"], cwd)
    return sorted(line for line in raw.splitlines() if line.strip())


def _porcelain_signature(porcelain_lines: list[str]) -> str:
    """Stable hash over the porcelain status (XY path) tuples."""
    payload = "\n".join(porcelain_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_ignored(path_str: str, cwd: Path, *, respect_dockerignore: bool) -> bool:
    """Return True if ``path_str`` is git-ignored (and, optionally, docker-ignored)."""
    parts = Path(path_str).parts
    if str(REPORTS_DIR) in parts:
        return True
    if respect_dockerignore and path_str.startswith(".hermes/"):
        # The image context excludes the entire .hermes tree per .dockerignore.
        return True
    if respect_dockerignore and _docker_ignores(cwd, path_str):
        return True
    # git check-ignore decides the rest (covers secrets like *.secret).
    return _git_is_ignored(path_str, cwd)


def _git_is_ignored(path_str: str, cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", path_str],
        cwd=str(cwd),
        check=False,
        capture_output=True,
    )
    # exit 0 -> ignored; 1 -> not ignored
    return result.returncode == 0


_DOCKERIGNORE_CACHE: dict[str, list[tuple[re.Pattern[str], bool, bool]]] | None = None


def _load_dockerignore(cwd: Path) -> list[tuple[re.Pattern[str], bool, bool]]:
    """Load .dockerignore patterns as (regex, is_dir, negate) tuples."""
    global _DOCKERIGNORE_CACHE
    key = str(cwd)
    if _DOCKERIGNORE_CACHE is not None and key in _DOCKERIGNORE_CACHE:
        return _DOCKERIGNORE_CACHE[key]
    patterns: list[tuple[re.Pattern[str], bool, bool]] = []
    path = cwd / ".dockerignore"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(_dockerignore_to_regex(line))
    if _DOCKERIGNORE_CACHE is None:
        _DOCKERIGNORE_CACHE = {}
    _DOCKERIGNORE_CACHE[key] = patterns
    return patterns


def _dockerignore_to_regex(pattern: str) -> tuple[re.Pattern[str], bool, bool]:
    """Translate a .dockerignore glob pattern to (anchored regex, is_dir, negate)."""
    negate = pattern.startswith("!")
    p = pattern[1:] if negate else pattern
    is_dir = p.endswith("/")
    if is_dir:
        p = p[:-1]
    if p.startswith("/"):
        p = p[1:]
    out = ["^"]
    for ch in p:
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
    regex = "".join(out)
    if is_dir:
        regex = f"{regex}(/.*)?$"
    else:
        regex = f"{regex}$"
    return re.compile(regex), is_dir, negate


def _docker_ignores(cwd: Path, path_str: str) -> bool:
    """Return True if a .dockerignore pattern excludes ``path_str``.

    Implements the ``.dockerignore`` last-match-wins / ``!`` re-include rule
    used by the Docker build context.
    """
    patterns = _load_dockerignore(cwd)
    ignored = False
    for regex, _is_dir, negate in patterns:
        if regex.match(path_str):
            ignored = not negate
    return ignored


def _porcelain_entries(porcelain_lines: list[str]) -> list[tuple[str, str]]:
    """Return (status, path) tuples from ``git status --porcelain``."""
    entries: list[tuple[str, str]] = []
    for line in porcelain_lines:
        path = line[3:].strip()
        if "->" in path:
            path = path.split("->", 1)[1].strip()
        entries.append((line[:2], path.replace("\\", "/")))
    return entries


def _is_deletion(status: str) -> bool:
    return status.startswith("D") or status[1:2] == "D"


def _delivery_files(cwd: Path) -> list[str]:
    """Tracked files + non-ignored untracked delivery files, including the plan.

    Deletions are recorded so the manifest reflects the working-tree delta; a
    deleted delivery file remains part of the delivery fingerprint.
    """
    porcelain_lines = _git_porcelain(cwd)
    tracked = set(_git_tracked_files(cwd))
    extra: set[str] = set()
    for status, path in _porcelain_entries(porcelain_lines):
        if not path:
            continue
        if _is_deletion(status):
            extra.add(path)  # deletion recorded as a delivery delta
            continue
        if not _is_ignored(path, cwd, respect_dockerignore=False):
            extra.add(path)
    files = tracked | extra
    if PLAN_PATH.is_file():
        files.add(str(PLAN_PATH).replace("\\", "/"))
    return sorted(p.replace("\\", "/") for p in files)


def _image_context_files(cwd: Path) -> list[str]:
    """Committed Docker build-context files (tracked, non-dockerignored).

    The image provenance is the content-addressed committed context; untracked
    work-in-progress is intentionally excluded so two builds at the same commit
    resolve to the same image context (and so the delivery and image domains
    differ whenever the tree is dirty). Deletions are recorded as deltas.
    """
    porcelain_lines = _git_porcelain(cwd)
    tracked = {
        p for p in _git_tracked_files(cwd)
        if not _is_ignored(p, cwd, respect_dockerignore=True)
    }
    deltas: set[str] = set()
    for status, path in _porcelain_entries(porcelain_lines):
        if path and _is_deletion(status):
            deltas.add(path)
    return sorted(p.replace("\\", "/") for p in tracked | deltas)


def _path_status_hash(cwd: Path, path_str: str, status: str) -> str:
    """SHA-256 over ``status\0path\0<file bytes or empty>``."""
    h = hashlib.sha256()
    h.update(status.encode("utf-8"))
    h.update(b"\x00")
    h.update(path_str.encode("utf-8"))
    h.update(b"\x00")
    full = cwd / path_str
    if full.is_file():
        try:
            h.update(full.read_bytes())
        except OSError:
            pass
    return h.hexdigest()


def _fingerprint(cwd: Path, files: Iterable[str], *, porcelain_lines: list[str]) -> str:
    """Aggregate hash over (path, status, per-file-hash) tuples, sorted."""
    status_by_path = _porcelain_status_map(porcelain_lines)
    tuples: list[str] = []
    for path_str in sorted(set(files)):
        status = status_by_path.get(path_str, " ")
        tuples.append(f"{status}\x00{path_str}\x00{_path_status_hash(cwd, path_str, status)}")
    payload = "\n".join(tuples).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _porcelain_status_map(porcelain_lines: list[str]) -> dict[str, str]:
    """Map path -> porcelain status code from ``git status --porcelain``."""
    mapping: dict[str, str] = {}
    for line in porcelain_lines:
        path = line[3:].strip()
        if "->" in path:
            path = path.split("->", 1)[1].strip()
        mapping[path.replace("\\", "/")] = line[:2]
    return mapping


def _branch(cwd: Path) -> str:
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd).strip()
    return branch or "HEAD"


def _commit(cwd: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd).strip()


def _is_dirty(porcelain_lines: list[str]) -> bool:
    return any(line.strip() for line in porcelain_lines)


def build_manifest(output_path: str | None = None) -> dict:
    """Build (and optionally write) the canonical source manifest.

    Args:
        output_path: If provided, write canonical JSON + LF to this path and
            create parent directories. The file is overwritten atomically.

    Returns:
        The manifest as a Python dict.
    """
    cwd = _find_repo_root()
    porcelain_lines = _git_porcelain(cwd)
    delivery = _delivery_files(cwd)
    image_context = _image_context_files(cwd)
    delivery_hash = _fingerprint(cwd, delivery, porcelain_lines=porcelain_lines)
    image_hash = _fingerprint(cwd, image_context, porcelain_lines=porcelain_lines)
    porcelain_hash = _porcelain_signature(porcelain_lines)
    dirty = _is_dirty(porcelain_lines)
    manifest = {
        "schema": MANIFEST_VERSION,
        "branch": _branch(cwd),
        "commit_sha": _commit(cwd),
        "dirty": dirty,
        "porcelain_hash": porcelain_hash,
        "delivery_tree_sha256": delivery_hash,
        "image_context_sha256": image_hash,
        "files": [
            {"path": p}
            for p in sorted({p.replace("\\", "/") for p in set(delivery) | set(image_context)})
        ],
    }
    if output_path:
        _write_canonical(output_path, manifest)
    return manifest


def _find_repo_root() -> Path:
    """Find the repository root by searching upward for a ``.git`` directory."""
    here = Path(os.getcwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
        if any((candidate / hint).exists() for hint in REPO_ROOT_HINT_FILES):
            return candidate
    return here


def _write_canonical(path: str, obj: dict) -> None:
    """Write canonical JSON (sorted keys, ASCII, comma-colon) + LF, atomically."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp, out)


def read_field(input_path: str, field: str) -> str:
    """Read and validate a single scalar field from a manifest file."""
    if field not in FIELDS:
        raise ValueError(f"unknown field: {field!r}")
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    value = data.get(field)
    if value is None:
        raise ValueError(f"missing field: {field!r}")
    scalar = value if isinstance(value, str) else str(value)
    pattern = FIELDS[field]
    if not re.match(pattern, scalar):
        raise ValueError(f"field {field!r} failed validation: {scalar!r}")
    return scalar


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build/read Phase 10 source manifest.")
    parser.add_argument("--output", help="Write canonical manifest JSON to this path.")
    parser.add_argument("--input", help="Read a manifest file (use with --field).")
    parser.add_argument(
        "--field",
        help="Print a single validated scalar (commit_sha|dirty|image_context_sha256).",
    )
    args = parser.parse_args(argv)

    if args.input:
        if not args.field:
            parser.error("--field is required with --input")
        sys.stdout.write(read_field(args.input, args.field) + "\n")
        return 0

    if not args.output:
        parser.error("--output is required when not reading a field")
    build_manifest(output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
