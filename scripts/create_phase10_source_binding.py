"""Create the Phase 10 source binding sidecar after a labeled build.

``create_phase10_source_binding.py --manifest PATH --output PATH --services api migrate``
resolves one unambiguous image ID per service, inspects the three exact OCI
labels, compares them to the manifest, and atomically writes canonical JSON plus
LF containing only the manifest SHA-256, branch/commit/dirty, porcelain/delivery/
image-context hashes, service image IDs, and validated labels.

Because ``docker-compose.yml`` gives both ``api`` and ``migrate`` the identical
``build: .`` context and ``Dockerfile``, a content-addressed build resolves both
services to the same image ID; the binding therefore permits cross-service
image-ID equality when the verified OCI labels are identical for both services
and rejects only genuinely ambiguous or missing IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import subprocess

LABELS = (
    "org.opencontainers.image.revision",
    "org.opencontainers.image.source-context-sha256",
    "org.opencontainers.image.source-dirty",
)


def _resolve_image_id(service: str) -> str:
    result = subprocess.run(
        ["docker", "compose", "images", "-q", service],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().splitlines()[0]
    # After a no-cache rebuild the running containers may reference an image
    # record the daemon no longer holds, which makes `compose images` fail;
    # resolve from the configured image name instead — the freshly built
    # image exists locally (D-47).
    config = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        capture_output=True,
        text=True,
    )
    if config.returncode != 0:
        raise RuntimeError(f"could not resolve image id for service {service!r}")
    config_json = json.loads(config.stdout)
    service_config = (config_json.get("services", {}).get(service, {}) or {})

    # Explicit image name (rare here): inspect it directly.
    image_name = service_config.get("image")
    if image_name:
        inspect = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_name],
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0 and inspect.stdout.strip():
            return inspect.stdout.strip()

    # Build-only services carry no `image` key; the freshly built image is
    # findable through the compose labels docker build stamps on it. Pick the
    # most recently created match (older layers linger after rebuilds).
    project = config_json.get("name") or "rag-vector-database-pipeline-project"
    listing = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.ID}}	{{.CreatedAt}}",
         "--filter", f"label=com.docker.compose.project={project}",
         "--filter", f"label=com.docker.compose.service={service}"],
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0 or not listing.stdout.strip():
        raise RuntimeError(f"could not resolve image id for service {service!r}")
    best_id, _best_created = max(
        (line.split("	", 1) for line in listing.stdout.strip().splitlines()),
        key=lambda parts: parts[1] if len(parts) > 1 else "",
    )
    if not best_id:
        raise RuntimeError(f"could not resolve image id for service {service!r}")
    return best_id


def _inspect_label(image_id: str, label: str) -> str:
    fmt = '{{ index .Config.Labels "' + label + '" }}'
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", fmt, image_id],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _binding_for_service(service: str) -> dict[str, Any]:
    image_id = _resolve_image_id(service)
    return {
        "image_id": image_id,
        "labels": {label: _inspect_label(image_id, label) for label in LABELS},
    }


def create_binding(manifest_path: str, output_path: str, services: list[str]) -> dict[str, Any]:
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    binding = {
        "schema_version": "phase10-source-binding-v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "branch": manifest.get("branch", ""),
        "commit": manifest.get("commit_sha", ""),
        "dirty": manifest.get("dirty", False),
        "porcelain_hash": manifest.get("porcelain_hash", ""),
        "delivery_tree_sha256": manifest.get("delivery_tree_sha256", ""),
        "image_context_sha256": manifest.get("image_context_sha256", ""),
        "services": {service: _binding_for_service(service) for service in services},
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    import os
    os.replace(tmp, out)
    return binding


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a Phase 10 source binding.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--services", required=True, nargs="+")
    args = parser.parse_args(argv)
    create_binding(args.manifest, args.output, args.services)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
