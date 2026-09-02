"""Host-side Docker image hygiene scanner (Phase 10 DOC.1).

Verifies that the built ``api``/``migrate`` images carry the expected OCI
source-provenance labels and that their final merged root filesystems (the
result of ``docker create --entrypoint /bin/true`` + ``docker export``)
contain no forbidden artifacts: ``.git``, ``.env*``, ``.hermes``, caches or
bytecode, local SQLite databases (``rag.db``/-wal/-shm), report files, host
absolute paths, escaping symlinks, path traversal members, credential
sentinels, attack fixtures outside the pinned allowlist, or content that
does not match the pinned policy hash inventory.

The scanner is strictly host-side: it refuses to run inside a container, it
never mounts the Docker socket, and every subprocess call uses an argv list
with ``shell=False`` semantics.  Matched credential bytes are never echoed;
only the offending member path is reported.

Output: one sorted-key single-line JSON object on stdout.  Exit status is 0
when every check passes and 2 for any refusal or failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

LABEL_REVISION = "org.opencontainers.image.revision"
LABEL_CONTEXT = "org.opencontainers.image.source-context-sha256"
LABEL_DIRTY = "org.opencontainers.image.source-dirty"

# Zero/unknown build-time placeholder defaults (Dockerfile ARG defaults) are
# never acceptable label values.
_DEFAULT_SENTINELS = {"", "none", "unknown", "null", "n/a", "dirty-unknown"}

_HEX_RE = re.compile(r"^[0-9a-f]+$")

# Pinned policy inventory (SHA-256 over the committed file bytes).  A member
# present in the image at one of these paths must hash to exactly the pinned
# value; a member that hashes to the pin is trusted and exempt from the
# credential sentinel byte scan.
PINNED_POLICY_HASHES = {
    "/app/config/source-trust-policy.json":
        "b61f58f519f0c67c7ac7820417c055329fed7974d3ea94bee4d361299b6a979a",
    "/app/config/content-safety-policy.json":
        "2a9c9c5d4d44cce8ecb02bbf2b8586f6dd86dc410e474b93552e22180637d4f1",
    "/app/config/context-security-policy.json":
        "baac1ee5c0c0c2e8a60a004910166f7fcb631188168550f183d777f4f31b2bc1",
    "/app/config/retrieval-security-policy.json":
        "1cc5310fffaf28bbefcf2debf6aa5fbf31ff88d81d7997d77d5c96f0c2acf1bf",
}

# Attack fixtures are shipped with the image's test suite; exactly these
# three members are allowed, at exactly these paths.
ATTACK_FIXTURE_ALLOWLIST = {
    "/app/app/tests/fixtures/attack_payloads.json",
    "/app/app/tests/fixtures/attack_payloads.schema.json",
    "/app/app/tests/fixtures/redteam-report.schema.json",
}

_FORBIDDEN_DB_BASENAMES = {"rag.db", "rag.db-wal", "rag.db-shm"}
_REPORT_FILE_RE = re.compile(r"^(redteam|red-team|phase10).*\.(json|md)$", re.IGNORECASE)
# Host-absolute path forms: POSIX single-letter roots ("/c/Users/...") and
# Windows drive+colon forms ("C:/Users/...", "C:\\Users\\...").
_HOST_ABSOLUTE_COMPONENT_RE = re.compile(r"^[a-z](?::|$)", re.IGNORECASE)
_HOST_ROOT_COMPONENTS = {"users", "host", "host_mnt"}

# Credential key/value sentinels: an uppercase env-style KEY=VALUE line with
# a non-empty value, or a well-known bearer-token prefix.  Only member paths
# with these suffixes are byte-scanned.
_SENTINEL_TEXT_SUFFIXES = {
    ".env", ".ini", ".cfg", ".conf", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".sh", ".md", ".properties", ".xml",
}
_SENTINEL_KEY_VALUE_RE = re.compile(
    rb"(?m)^[A-Z0-9_.-]*(?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|PRIVATE_KEY"
    rb"|PASSPHRASE|CREDENTIAL)[A-Z0-9_.-]*\s*=\s*\S+"
)
_SENTINEL_BEARER_RE = re.compile(
    rb"(?:ghp|gho|github_pat|glpat|sk-ant|sk-proj|sk-|xoxb|xoxp|AKIA)"
    rb"[A-Za-z0-9_\-]{16,}"
)
_SENTINEL_SCAN_MAX_BYTES = 1024 * 1024

_CONTAINER_MARKERS = (
    "/.dockerenv",
    "/app/.dockerenv",
    "/.containerenv",
    "/run/.containerenv",
)


def _is_inside_container() -> bool:
    """Host guard: the scanner must never execute inside a container."""
    import os

    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    return any(Path(marker).exists() for marker in _CONTAINER_MARKERS)


def _run(argv: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
    """Every subprocess call in this scanner goes through here: argv list,
    never a shell string, never ``shell=True``, never a Docker socket mount.

    A missing required binary (e.g. no ``docker`` on PATH) is a clean,
    bounded failure — never an unhandled traceback."""
    try:
        return subprocess.run(argv, capture_output=True, text=text)
    except FileNotFoundError:
        empty = "" if text else b""
        return subprocess.CompletedProcess(
            argv, 127, stdout=empty,
            stderr=f"required executable not found: {Path(argv[0]).name}")


def _normalize_member(name: str) -> str:
    """Normalize a tar member name to an absolute POSIX image path."""
    cleaned = name.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("/")
    return "/" + cleaned


def _load_expected_labels(args: argparse.Namespace) -> dict[str, str | None]:
    """Resolve expected OCI label values from flags and/or the manifest.

    Exact values are returned when known (manifest field or flag); ``None``
    means "structurally verified only" (present, non-default, well-formed).
    """
    expected: dict[str, str | None] = {
        LABEL_REVISION: getattr(args, "expected_revision", None),
        LABEL_CONTEXT: getattr(args, "expected_source_context_sha256", None),
        LABEL_DIRTY: getattr(args, "expected_dirty", None),
    }
    manifest = getattr(args, "manifest", None)
    if manifest:
        try:
            obj = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            obj = {}
        if isinstance(obj, dict):
            revision = obj.get("commit_sha") or obj.get("revision") \
                or obj.get("source_revision")
            context = obj.get("image_context_sha256") \
                or obj.get("source_context_sha256") or obj.get("context_sha256")
            dirty = obj.get("dirty", obj.get("source_dirty"))
            if isinstance(revision, str) and revision:
                expected[LABEL_REVISION] = revision
            if isinstance(context, str) and context:
                expected[LABEL_CONTEXT] = context
            if isinstance(dirty, bool):
                expected[LABEL_DIRTY] = "true" if dirty else "false"
            elif isinstance(dirty, str) and dirty:
                expected[LABEL_DIRTY] = dirty.lower()
    return expected


def _verify_labels(labels: dict | None,
                   expected: dict[str, str | None]) -> list[str]:
    """Verify OCI labels; returns a list of human-readable refusals."""
    problems: list[str] = []
    labels = labels if isinstance(labels, dict) else {}
    for key, want in expected.items():
        got = labels.get(key)
        if not isinstance(got, str) or got.strip().lower() in _DEFAULT_SENTINELS:
            problems.append(
                f"label absent or default: {key} (value suppressed)")
            continue
        if want is not None:
            if got != want:
                problems.append(
                    f"label mismatch: {key} (expected value suppressed)")
            continue
        # Structural verification when no expected value is known.
        if key == LABEL_REVISION and not (_HEX_RE.match(got) and 7 <= len(got) <= 64):
            problems.append(f"label malformed: {key}")
        elif key == LABEL_CONTEXT and not (_HEX_RE.match(got) and len(got) == 64):
            problems.append(f"label malformed: {key}")
        elif key == LABEL_DIRTY and got.strip().lower() not in ("true", "false"):
            problems.append(f"label malformed: {key}")
    return problems


def _whiteouts(members: list[tarfile.TarInfo]) -> tuple[set[str], set[str]]:
    """Paths (and directory subtrees) hidden by OCI layer whiteout markers.

    Returns ``(exact_hidden_paths, hidden_directory_prefixes)``.
    """
    hidden: set[str] = set()
    prefixes: set[str] = set()
    for info in members:
        path = _normalize_member(info.name)
        parent = str(PurePosixPath(path).parent)
        base = PurePosixPath(path).name
        if base == ".wh..wh..opq":
            # Opaque directory: everything under the parent is hidden.
            prefixes.add(parent + "/")
        elif base.startswith(".wh."):
            hidden.add(str(PurePosixPath(parent) / base[len(".wh."):]))
    return hidden, prefixes


def _is_hidden(path: str, hidden: set[str], prefixes: set[str]) -> bool:
    if path in hidden:
        return True
    if any(path.startswith(prefix) for prefix in prefixes):
        return True
    return any(path.startswith(target + "/") for target in hidden)


def _scan_tarball(tar_path: Path, forbidden: list[str]) -> int:
    """Scan one exported rootfs tarball; append violations; return the
    number of regular-file members scanned (after whiteout filtering)."""
    scanned = 0
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        hidden, prefixes = _whiteouts(members)
        # The image namespace: every member path visible in the final merged
        # rootfs (after whiteout filtering).  A symlink whose lexical target
        # resolves to a path that is not part of the image points outside the
        # image namespace.
        namespace = set()
        for info in members:
            if PurePosixPath(info.name.replace("\\", "/")).name.startswith(".wh."):
                continue
            member_path = _normalize_member(info.name)
            if not _is_hidden(member_path, hidden, prefixes):
                namespace.add(member_path)
        for info in members:
            path = _normalize_member(info.name)
            base = PurePosixPath(path).name
            if base.startswith(".wh."):
                continue  # whiteout marker itself, not image content
            if _is_hidden(path, hidden, prefixes):
                continue
            parts = PurePosixPath(path).parts[1:]
            # Path traversal: never honored, always refused.
            if ".." in PurePosixPath(info.name.replace("\\", "/")).parts or \
                    ".." in parts:
                forbidden.append(f"path traversal member: {info.name}")
                continue
            # Escaping symlink.  A symlink is "escaping" only when it
            # (i) resolves via a ".." chain above the image's virtual root,
            # (ii) targets a host-absolute path form (Windows drive+colon
            # or POSIX single-letter root), or (iii) resolves to a path
            # outside the image namespace.  A symlink that merely points
            # elsewhere WITHIN the image filesystem (e.g. a Debian OS
            # symlink "/bin -> usr/bin") is legitimate and not flagged.
            if info.issym():
                target = info.linkname.replace("\\", "/")
                stripped = target.lstrip("/")
                host_absolute = bool(
                    stripped
                    and _HOST_ABSOLUTE_COMPONENT_RE.match(
                        stripped.split("/")[0])
                )
                if target.startswith("/"):
                    base_path = ""
                else:
                    base_path = str(PurePosixPath(path).parent)
                stack: list[str] = [
                    p for p in base_path.split("/") if p not in ("", ".")
                ]
                escaped_root = False
                for part in target.split("/"):
                    if part in ("", "."):
                        continue
                    if part == "..":
                        if stack:
                            stack.pop()
                        else:
                            escaped_root = True
                    else:
                        stack.append(part)
                norm = "/" + "/".join(stack)
                if host_absolute:
                    forbidden.append(
                        f"host absolute path member: {path} "
                        f"(symlink target)")
                elif escaped_root or norm not in namespace:
                    forbidden.append(f"escaping symlink member: {info.name}")
                continue
            if info.islnk():
                continue
            if info.isdir():
                continue
            if not info.isfile():
                continue
            scanned += 1
            lowered_parts = [p.lower() for p in parts]
            first = lowered_parts[0] if lowered_parts else ""
            # Host absolute path members (Windows drive-style roots, /Users,
            # host bind mounts) never belong in a Linux image rooted at /app.
            if _HOST_ABSOLUTE_COMPONENT_RE.match(first) or \
                    first in _HOST_ROOT_COMPONENTS:
                forbidden.append(f"host absolute path member: {path}")
                continue
            if ".git" in lowered_parts:
                forbidden.append(f"git directory member: {path}")
                continue
            if ".hermes" in lowered_parts:
                forbidden.append(f"hermes directory member: {path}")
                continue
            if "__pycache__" in lowered_parts or base.endswith(".pyc"):
                forbidden.append(f"cache/bytecode member: {path}")
                continue
            if base == ".env" or base.startswith(".env"):
                forbidden.append(f"environment file member: {path}")
                continue
            if base.lower() in _FORBIDDEN_DB_BASENAMES:
                forbidden.append(f"local database member: {path}")
                continue
            if _REPORT_FILE_RE.match(base) and path not in ATTACK_FIXTURE_ALLOWLIST:
                forbidden.append(f"report file member: {path}")
                continue
            # Attack fixture allowlist: exact three paths only.
            under_fixtures = path.startswith("/app/app/tests/fixtures/")
            name_allowlisted = base in {
                PurePosixPath(p).name for p in ATTACK_FIXTURE_ALLOWLIST
            }
            if (under_fixtures and path not in ATTACK_FIXTURE_ALLOWLIST) or \
                    (name_allowlisted and path not in ATTACK_FIXTURE_ALLOWLIST):
                forbidden.append(f"attack fixture outside allowlist: {path}")
                continue
            if path in ATTACK_FIXTURE_ALLOWLIST:
                # Pinned test-corpus fixture at its exact allowed path.
                continue
            # Pinned hash inventory.
            pinned = PINNED_POLICY_HASHES.get(path)
            if pinned is not None:
                digest = hashlib.sha256(tf.extractfile(info).read()).hexdigest()
                if digest != pinned:
                    forbidden.append(f"pinned hash mismatch: {path}")
                    continue
                continue  # hash-verified content is trusted
            # Credential sentinel byte scan (value never echoed).
            suffix = Path(base).suffix.lower()
            if suffix in _SENTINEL_TEXT_SUFFIXES and info.size <= _SENTINEL_SCAN_MAX_BYTES:
                data = tf.extractfile(info).read(info.size)
                if _SENTINEL_KEY_VALUE_RE.search(data) or \
                        _SENTINEL_BEARER_RE.search(data):
                    forbidden.append(
                        f"credential sentinel (value suppressed) in: {path}")
                continue
    return scanned


def main(argv: list[str] | None = None) -> int:
    if _is_inside_container():
        print(
            "refusing to run: validate_image_hygiene is a host-side scanner "
            "and must never execute inside a container",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=None,
                        help="source manifest JSON with expected label values")
    parser.add_argument("--services", nargs="+", default=["api", "migrate"],
                        help="compose services to scan")
    parser.add_argument("--expected-revision", default=None)
    parser.add_argument("--expected-source-context-sha256", default=None)
    parser.add_argument("--expected-dirty", default=None)
    args = parser.parse_args(argv)

    expected_labels = _load_expected_labels(args)
    forbidden: list[str] = []
    per_service: dict[str, dict] = {}
    cleanup_ok = True
    failure = False

    for service in args.services:
        images = _run(["docker", "compose", "images", "-q", service])
        if images.returncode != 0:
            forbidden.append(f"image id resolution failed: service {service}")
            failure = True
            break
        ids = [line.strip() for line in (images.stdout or "").splitlines()
               if line.strip()]
        if not ids:
            forbidden.append(f"missing image id: service {service}")
            failure = True
            break
        if len(ids) > 1:
            forbidden.append(f"ambiguous image id: service {service}")
            failure = True
            break
        image_id = ids[0]

        inspect = _run(["docker", "inspect", image_id])
        labels: dict | None = None
        if inspect.returncode == 0:
            try:
                payload = inspect.stdout
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8", "replace")
                labels = json.loads(payload)[0].get("Config", {}).get("Labels")
            except (ValueError, IndexError, AttributeError):
                labels = None
        label_problems = _verify_labels(labels, expected_labels)
        if label_problems:
            forbidden.extend(label_problems)
            failure = True
            break

        create = _run(["docker", "create", "--entrypoint", "/bin/true", image_id])
        if create.returncode != 0 or not (create.stdout or "").strip():
            forbidden.append(f"container creation failed: service {service}")
            failure = True
            break
        container_id = create.stdout.strip()

        import os
        fd, tar_name = tempfile.mkstemp(prefix="hygiene-", suffix=".tar")
        os.close(fd)
        tar_path = Path(tar_name)
        try:
            export = _run(["docker", "export", container_id], text=False)
            if export.returncode != 0 or not export.stdout:
                forbidden.append(f"rootfs export failed: service {service}")
                failure = True
            else:
                tar_path.write_bytes(export.stdout)
                try:
                    scanned = _scan_tarball(tar_path, forbidden)
                except tarfile.TarError as exc:
                    forbidden.append(
                        f"unreadable export tarball: service {service} "
                        f"({exc.__class__.__name__})")
                    failure = True
                    scanned = 0
                per_service[service] = {
                    "image_id": image_id,
                    "scanned_files": scanned,
                }
        finally:
            try:
                if tar_path.exists():
                    tar_path.unlink()
            except OSError:
                cleanup_ok = False
            rm = _run(["docker", "rm", container_id])
            if rm.returncode != 0:
                cleanup_ok = False
                failure = True

    if failure or forbidden:
        status = "failed"
    else:
        status = "passed"

    report = {service: info for service, info in per_service.items()}
    report["forbidden_artifacts"] = forbidden
    report["cleanup_complete"] = cleanup_ok
    report["status"] = status
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 2 if (failure or forbidden or not cleanup_ok) else 0


if __name__ == "__main__":
    sys.exit(main())
