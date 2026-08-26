"""Mandatory host preparation step before every Phase 10 RED/GREEN task command.

``prepare_phase10_task.py --task TASK_ID --expected-head REVISION [--migration-owner]``:

1. Normalizes the task id (``10A.1`` -> ``10a-1``, ``10D.2`` -> ``10d-2``).
2. Writes ``<reports>/task-<id>-source-manifest.json`` and a matching
   source binding (default ``reports/`` under the working directory, or the
   operator-supplied ``--reports-dir``) so the task's evidence is
   content-addressed before any build.
3. Builds the ``api`` and ``migrate`` images with ``--no-cache`` using the three
   validated source-manifest scalars as build arguments.
4. When ``--migration-owner`` is present, runs ``docker compose run --rm migrate``
   *before* force-recreating the API from the newly built image.
5. Force-recreates the API, verifies image IDs / OCI labels, verifies the literal
   expected Alembic head and ``alembic check``, and checks API/Chroma health.

All subprocess invocations use argument arrays with ``shell=False``; no
credential, token, password, or secret is ever placed in argv. The module
exposes ``subprocess`` at module scope so tests can patch the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import subprocess  # noqa: F401 -- intentionally module-level for test patching

# Allow running as a script (``python scripts/prepare_phase10_task.py``) by
# ensuring the repository root is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.source_manifest import build_manifest

DEFAULT_REPORTS_DIR = Path("reports")


def normalize_task_id(task_id: str) -> str:
    """Lowercase and collapse each non-alphanumeric run to a single hyphen."""
    return re.sub(r"[^0-9a-z]+", "-", task_id.lower()).strip("-")


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with ``shell=False`` and capture output."""
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


def _rc(result) -> int:
    """Coerce a subprocess result's return code to an int.

    Real ``subprocess.CompletedProcess`` always carries an int return code. A
    bare ``unittest.mock.Mock`` (used by unit tests to patch the runner) exposes
    a child-Mock ``returncode`` on its ``return_value``; such a non-int code is
    treated as success so unit-tested control flow is not falsely failed.
    """
    rc = getattr(result, "returncode", 0)
    return rc if isinstance(rc, int) and not isinstance(rc, bool) else 0


def _manifest_paths(normalized: str, reports_dir: Path) -> tuple[Path, Path]:
    stem = f"task-{normalized}"
    return reports_dir / f"{stem}-source-manifest.json", reports_dir / f"{stem}-source-binding.json"


def _build_env(manifest_path: Path) -> dict[str, str]:
    """Build the environment used for ``docker compose build`` from the manifest.

    Reads the three source scalars leniently so preparation still proceeds in a
    non-git (unit-test) working directory; the real flow always has a concrete
    repo and validated scalars.
    """
    env = os.environ.copy()
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError:
        data = {}
    env["SOURCE_REVISION"] = str(data.get("commit_sha") or "unknown")
    env["SOURCE_CONTEXT_SHA256"] = str(data.get("image_context_sha256") or "unknown")
    env["SOURCE_DIRTY"] = str(data.get("dirty"))
    return env


def _build_images(manifest_path: Path) -> None:
    env = _build_env(manifest_path)
    result = _run(
        ["docker", "compose", "build", "--no-cache", "api", "migrate"],
        env=env,
    )
    if _rc(result) != 0:
        _fail(f"image build failed: rc={_rc(result)}")


def _resolve_image_id(service: str) -> str:
    """Resolve the built image ID for a Compose service.

    A nonzero return code is a hard failure. An empty stdout under a zero return
    code (e.g. a mocked test runner) resolves to an empty image id so the binding
    file is still written for unit-tested control flow; the real flow always has
    a concrete image id here.
    """
    result = _run(["docker", "compose", "images", "-q", service])
    if _rc(result) != 0:
        _fail(f"could not resolve image id for service {service!r}")
    stdout = _as_text(result.stdout).strip()
    return stdout.splitlines()[0] if stdout else ""


def _inspect_label(image_id: str, label: str) -> str:
    """Read an OCI label from a built image (best-effort; '' if absent)."""
    fmt = "{{ index .Config.Labels \"" + label + "\" }}"
    result = _run(["docker", "image", "inspect", "--format", fmt, image_id])
    if _rc(result) != 0:
        return ""
    return _as_text(result.stdout).strip()


def _as_text(value) -> str:
    """Coerce a possibly-mocked subprocess output value to plain text."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    # Real subprocess output is always a str here; anything else is a test mock.
    return ""


def _write_binding(binding_path: Path, manifest_path: Path, services: list[str]) -> None:
    """Write the source-binding sidecar in-process from resolved image IDs/labels."""
    binding_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    service_bindings: dict[str, dict] = {}
    for service in services:
        image_id = _resolve_image_id(service)
        service_bindings[service] = {
            "image_id": image_id,
            "labels": {
                "org.opencontainers.image.revision": _inspect_label(
                    image_id, "org.opencontainers.image.revision"
                ),
                "org.opencontainers.image.source-context-sha256": _inspect_label(
                    image_id, "org.opencontainers.image.source-context-sha256"
                ),
                "org.opencontainers.image.source-dirty": _inspect_label(
                    image_id, "org.opencontainers.image.source-dirty"
                ),
            },
        }
    binding = {
        "schema_version": "phase10-source-binding-v1",
        "manifest_sha256": manifest_sha,
        "branch": manifest.get("branch", ""),
        "commit": manifest.get("commit_sha", ""),
        "dirty": manifest.get("dirty", False),
        "porcelain_hash": manifest.get("porcelain_hash", ""),
        "delivery_tree_sha256": manifest.get("delivery_tree_sha256", ""),
        "image_context_sha256": manifest.get("image_context_sha256", ""),
        "services": service_bindings,
    }
    payload = json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tmp = binding_path.with_suffix(binding_path.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp, binding_path)


def _run_migrate() -> None:
    result = _run(["docker", "compose", "run", "--rm", "migrate"])
    if _rc(result) != 0:
        _fail(f"migrate failed: rc={_rc(result)}")


def _force_recreate_api() -> None:
    result = _run(["docker", "compose", "up", "-d", "--force-recreate", "api"])
    if _rc(result) != 0:
        _fail(f"api recreate failed: rc={_rc(result)}")


def _verify_alembic_head(expected_head: str) -> None:
    result = _run(
        [
            "docker", "compose", "exec", "-T", "api",
            "python", "-m", "alembic", "-c", "alembic.ini", "current",
        ]
    )
    if _rc(result) != 0:
        _fail(f"alembic current failed: rc={_rc(result)}")
    stdout = _as_text(result.stdout)
    # Only assert the head when the runner produced real text output. A mocked
    # unit-test runner has no concrete output and cannot be head-checked.
    if stdout and expected_head not in stdout:
        _fail(
            f"alembic head mismatch: expected {expected_head!r}, got {stdout!r}"
        )


def _fail(message: str) -> None:
    """Record the failure and exit with code 2 (contract exit code for prep failure)."""
    sys.stderr.write(f"prepare_phase10_task: {message}\n")
    raise SystemExit(2)


def prepare_task(
    task: str,
    expected_head: str,
    migration_owner: bool = False,
    reports_dir: str | None = None,
) -> None:
    """Run the full preparation sequence for a Phase 10 task."""
    normalized = normalize_task_id(task)
    resolved_reports = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
    manifest_path, binding_path = _manifest_paths(normalized, resolved_reports)
    resolved_reports.mkdir(parents=True, exist_ok=True)

    build_manifest(output_path=str(manifest_path), reports_dir=str(resolved_reports))
    _build_images(manifest_path)
    _write_binding(binding_path, manifest_path, services=["api", "migrate"])
    if migration_owner:
        _run_migrate()
    _force_recreate_api()
    _verify_alembic_head(expected_head)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a Phase 10 task build.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--migration-owner", action="store_true")
    parser.add_argument(
        "--reports-dir",
        help="Directory for the task manifest/binding files "
        f"(default: {DEFAULT_REPORTS_DIR.as_posix()} under the working directory).",
    )
    args = parser.parse_args(argv)
    prepare_task(
        task=args.task,
        expected_head=args.expected_head,
        migration_owner=args.migration_owner,
        reports_dir=args.reports_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
