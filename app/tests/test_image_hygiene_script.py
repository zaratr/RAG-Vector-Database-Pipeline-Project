"""Phase 10 DOC.1 — host-side image hygiene scanner tests (appendix spec).

Covers: ``scripts/validate_image_hygiene.py`` — exact argv observation,
synthetic export-tarball member scanning, forbidden-artifact detection,
OCI label verification, pinned-hash inventory, cleanup in ``finally``.

D1(a)/Q3 adaptation: every lane runs anywhere with a mocked ``subprocess.run``
and synthetic tarballs under ``tmp_path`` — zero Docker dependency.  The
dispatcher executes the scanner in-process (``runpy``) beneath the active
monkeypatch, so every ``subprocess.run`` the scanner issues — including all
``docker`` commands — is intercepted and simulated; no Docker socket, no
network, no container.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import runpy
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "validate_image_hygiene.py"

# The pristine subprocess.run, captured before any monkeypatching.
_REAL_RUN = subprocess.run

# Structurally valid OCI source-provenance labels (the expected values are
# unknown when no manifest exists, so the scanner verifies structure).
GOOD_LABELS = {
    "org.opencontainers.image.revision": "f" * 40,
    "org.opencontainers.image.source-context-sha256": "a" * 64,
    "org.opencontainers.image.source-dirty": "false",
}


def _argv(manifest: str) -> list[str]:
    return [sys.executable, str(SCRIPT),
            "--manifest", manifest,
            "--services", "api", "migrate"]


# ---------------------------------------------------------------------------
# Shared helpers (appendix-referenced: _build_synthetic_tarball,
# _fake_run_creating_container, _run_valid_scan)
# ---------------------------------------------------------------------------

def _build_synthetic_tarball(directory, members) -> Path:
    """Build a synthetic ``docker export`` tarball under *directory*."""
    tarball = Path(directory) / "synthetic-export.tar"
    with tarfile.open(tarball, "w") as tf:
        for member in members:
            if isinstance(member, tuple):
                name, data = member
            else:
                name, data = member, b""
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tarball


def _run_scanner_in_process(argv, **kwargs):
    """Execute the scanner script in-process so that its own
    ``subprocess.run`` calls flow through the already-active monkeypatch."""
    old_argv = sys.argv
    out, err = io.StringIO(), io.StringIO()
    sys.argv = [str(a) for a in argv[1:]]
    code = 0
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exc:
        if isinstance(exc.code, int):
            code = exc.code
        else:
            code = 0 if exc.code is None else 2
    finally:
        sys.argv = old_argv
    as_text = bool(kwargs.get("text") or kwargs.get("universal_newlines"))
    stdout = out.getvalue() if as_text else out.getvalue().encode()
    stderr = err.getvalue() if as_text else err.getvalue().encode()
    return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr=stderr)


def _simulate_docker(argv, tarball: Path, **kwargs):
    """Simulate the exact docker commands the scanner issues."""
    as_text = bool(kwargs.get("text", True))

    def done(stdout) -> subprocess.CompletedProcess:
        if as_text and isinstance(stdout, bytes):
            stdout = stdout.decode()
        if not as_text and isinstance(stdout, str):
            stdout = stdout.encode()
        stderr = "" if as_text else b""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=stderr)

    if "images" in argv and "-q" in argv:
        return done("ab" * 32 + "\n")
    if argv[:2] == ["docker", "inspect"]:
        return done(json.dumps([{"Config": {"Labels": dict(GOOD_LABELS)}}]))
    if argv[:2] == ["docker", "create"]:
        return done("c" * 64 + "\n")
    if argv[:2] == ["docker", "export"]:
        return done(Path(tarball).read_bytes())
    if argv[:2] == ["docker", "rm"]:
        return done("")
    return subprocess.CompletedProcess(
        argv, 1, stdout="", stderr="unsupported simulated docker command")


def _fake_run_creating_container(tarball):
    """Return a ``subprocess.run`` dispatcher that executes the scanner
    in-process and simulates every docker argv from the synthetic tarball."""
    def base(argv, **kwargs):
        argv = [str(a) for a in argv]
        if len(argv) >= 2 and Path(argv[1]).resolve() == SCRIPT.resolve():
            return _run_scanner_in_process(argv, **kwargs)
        if argv and argv[0] == "docker":
            return _simulate_docker(argv, tarball, **kwargs)
        return _REAL_RUN(argv, **kwargs)
    return base


def _run_valid_scan(monkeypatch, tmp_path) -> subprocess.CompletedProcess:
    members = [
        ("/app/app/main.py", b"print('ok')\n"),
        ("/app/requirements.txt", b"fastapi\n"),
    ]
    tarball = _build_synthetic_tarball(tmp_path, members=members)
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    # The outer invocation must resolve the (patched) module attribute so the
    # scanner executes in-process beneath the monkeypatch.
    return subprocess.run(_argv(str(tmp_path / "manifest.json")),
                          capture_output=True, check=False)


def _load_scanner_module():
    spec = importlib.util.spec_from_file_location(
        "validate_image_hygiene_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Host-side guard (H01 — ADAPT-REQUIRED per owner ruling D1(a)/Q3: the
# appendix lane ran inside the API container, which contradicts the
# host-side scanner contract; the adaptation asserts the same host-guard
# substance, runnable anywhere with zero Docker dependency.)
# ---------------------------------------------------------------------------

def test_script_is_host_side_never_runs_inside_api(monkeypatch, capsys):
    scanner = _load_scanner_module()
    # Positive: when container markers are present the scanner refuses.
    monkeypatch.setattr(scanner, "_is_inside_container", lambda: True)
    assert scanner.main(["--manifest", "manifest.json",
                         "--services", "api"]) == 2
    err = capsys.readouterr().err.lower()
    assert "host" in err or "container" in err
    # Negative: on the host the guard passes and a clean scan proceeds.
    monkeypatch.setattr(scanner, "_is_inside_container", lambda: False)
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[("/app/app/main.py", b"")])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    assert scanner.main(["--manifest", str(scratch / "manifest.json"),
                         "--services", "api"]) == 0


def test_no_docker_socket_mounted(monkeypatch):
    # Inspect every subprocess.run argv the script issues; assert no -v /var/run/docker.sock.
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)
    recorded = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    for argv in recorded:
        joined = " ".join(str(a) for a in argv)
        assert "/var/run/docker.sock" not in joined, \
            f"scanner must not mount the Docker socket: {argv}"


def test_missing_image_returns_exit_two(monkeypatch):
    # Mock `docker compose images -q api` to return empty; assert exit 2.
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)

    def fake_run(argv, **kwargs):
        if "images" in argv and "-q" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2


def test_ambiguous_image_id_returns_exit_two(monkeypatch):
    # Mock two image IDs returned; assert exit 2.
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)

    def fake_run(argv, **kwargs):
        if "images" in argv and "-q" in argv:
            return subprocess.CompletedProcess(
                argv, 0,
                stdout="111111111111111111111111111111111111111111111111\n"
                       "222222222222222222222222222222222222222222222222\n",
                stderr="")
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2


def test_label_mismatch_returns_exit_two(monkeypatch):
    # Mock docker inspect to return labels that don't match the manifest;
    # assert exit 2.
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)

    def fake_run(argv, **kwargs):
        if "inspect" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps([{
                "Config": {"Labels": {
                    "org.opencontainers.image.title": "DEFINITELY-WRONG-TITLE-xyz",
                    "org.opencontainers.image.source": "https://mismatched.invalid/rag",
                }},
            }]), stderr="")
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2


def test_default_label_rejected(monkeypatch):
    # Labels containing the zero/unknown defaults -> exit 2.
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)

    def fake_run(argv, **kwargs):
        if "inspect" in argv:
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps([{
                "Config": {"Labels": {
                    "org.opencontainers.image.title": "",
                    "org.opencontainers.image.source": "UNKNOWN",
                    "org.opencontainers.image.revision": "none",
                }},
            }]), stderr="")
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2


def test_forbidden_git_directory_detected(monkeypatch, tmp_path):
    tarball = _build_synthetic_tarball(tmp_path, members=["/app/.git/config"])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    out = json.loads(result.stdout)
    assert any(".git" in a for a in out["forbidden_artifacts"])


def test_forbidden_env_file_detected(monkeypatch, tmp_path):
    tarball = _build_synthetic_tarball(tmp_path, members=["/app/.env"])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    out = json.loads(result.stdout)
    assert any(".env" in a for a in out["forbidden_artifacts"])


def test_forbidden_hermes_directory_detected(monkeypatch, tmp_path):
    tarball = _build_synthetic_tarball(tmp_path, members=["/app/.hermes/reports/x.json"])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    out = json.loads(result.stdout)
    assert any(".hermes" in a for a in out["forbidden_artifacts"])


def test_forbidden_local_db_wal_shm_detected(monkeypatch, tmp_path):
    for fname in ["rag.db", "rag.db-wal", "rag.db-shm"]:
        tarball = _build_synthetic_tarball(tmp_path, members=[f"/app/data/{fname}"])
        monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
        result = subprocess.run(_argv("manifest.json"),
                                capture_output=True, check=False)
        assert result.returncode != 0, f"{fname} must be rejected"
        out = json.loads(result.stdout)
        assert any(fname in a for a in out["forbidden_artifacts"]), fname


def test_forbidden_report_files_detected(monkeypatch, tmp_path):
    tarball = _build_synthetic_tarball(tmp_path, members=["/app/.hermes/reports/redteam.json"])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    out = json.loads(result.stdout)
    assert any("redteam" in a or ".hermes" in a for a in out["forbidden_artifacts"])


def test_forbidden_host_absolute_path_detected(monkeypatch, tmp_path):
    tarball = _build_synthetic_tarball(tmp_path, members=["/c/Users/zarat/secret.txt"])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    out = json.loads(result.stdout)
    # A member escaping the /app image root (host absolute path) is forbidden.
    assert any("secret.txt" in a or "Users" in a for a in out["forbidden_artifacts"])


def test_forbidden_credential_sentinel_in_file_bytes(monkeypatch, tmp_path):
    tarball = _build_synthetic_tarball(tmp_path,
        members=[("/app/config/x.txt", b"RAG_OPERATOR_TOKEN=secret")])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    out = json.loads(result.stdout)
    assert any("x.txt" in a for a in out["forbidden_artifacts"])
    # The secret value itself must never be echoed in the report.
    assert "secret" not in json.dumps(out["forbidden_artifacts"])


def test_attack_fixtures_outside_allowlist_rejected(monkeypatch, tmp_path):
    # Only the three pinned files under /app/app/tests/fixtures/ are allowed.
    tarball = _build_synthetic_tarball(tmp_path,
        members=["/app/app/tests/fixtures/extra_attack.json"])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    out = json.loads(result.stdout)
    assert any("extra_attack.json" in a or "fixtures" in a
               for a in out["forbidden_artifacts"])


def test_allowed_attack_fixtures_pass(monkeypatch, tmp_path):
    allowed = [
        "/app/app/tests/fixtures/attack_payloads.json",
        "/app/app/tests/fixtures/attack_payloads.schema.json",
        "/app/app/tests/fixtures/redteam-report.schema.json",
    ]
    members = [
        # Image path /app/<rel> maps to the repository <rel>.
        (tar_path, (PROJECT_ROOT / tar_path.removeprefix("/app/")).read_bytes())
        for tar_path in allowed
    ]
    tarball = _build_synthetic_tarball(tmp_path, members=members)
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    out = json.loads(result.stdout)
    assert out["forbidden_artifacts"] == []
    assert out["status"] == "passed"


def test_pinned_policy_hash_inventory_verified(monkeypatch, tmp_path):
    # The scanner inventories required policy/schema files at exact paths and
    # verifies their pinned hashes; a tampered content -> exit 2.
    # H16 adaptation: pins recomputed against the current config/*.json at
    # the DOC.1 branch point (the appendix values for source-trust,
    # content-safety, and context-security still match; retrieval-security
    # was added to the inventory).
    PINNED = {
        "/app/config/source-trust-policy.json":
            "b61f58f519f0c67c7ac7820417c055329fed7974d3ea94bee4d361299b6a979a",
        "/app/config/content-safety-policy.json":
            "2a9c9c5d4d44cce8ecb02bbf2b8586f6dd86dc410e474b93552e22180637d4f1",
        "/app/config/context-security-policy.json":
            "baac1ee5c0c0c2e8a60a004910166f7fcb631188168550f183d777f4f31b2bc1",
        "/app/config/retrieval-security-policy.json":
            "1cc5310fffaf28bbefcf2debf6aa5fbf31ff88d81d7997d77d5c96f0c2acf1bf",
    }
    members = [(path, b"tampered-policy-content-does-not-match-pinned-sha256")
               for path in PINNED]
    tarball = _build_synthetic_tarball(tmp_path, members=members)
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2
    out = json.loads(result.stdout)
    assert any(path in artifact or "hash" in artifact.lower()
               for path in PINNED for artifact in out["forbidden_artifacts"])


def test_output_is_sorted_key_json_line(monkeypatch, tmp_path):
    result = _run_valid_scan(monkeypatch, tmp_path)
    out = json.loads(result.stdout)
    assert set(out) >= {"api", "migrate", "forbidden_artifacts",
                        "cleanup_complete", "status"}
    assert out["forbidden_artifacts"] == []
    assert out["status"] == "passed"
    # Sorted-key single-line JSON: byte-identical re-serialization.
    assert result.stdout == json.dumps(
        out, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def test_output_never_emits_matched_secret_bytes(monkeypatch, tmp_path):
    # Seed a credential-looking sentinel; assert the output JSON does not
    # contain the sentinel substring, only a forbidden_artifacts entry naming
    # the file path.
    sentinel = "glpat-9xmXFAKEsecretDOnotEMIT1234567890"
    payload = ("RAG_OPERATOR_TOKEN=" + sentinel).encode()
    tarball = _build_synthetic_tarball(tmp_path,
        members=[("/app/config/leaked.ini", payload)])
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode != 0
    # The matched secret bytes must never appear anywhere in scanner output.
    assert sentinel not in result.stdout.decode()
    assert sentinel not in result.stderr.decode()
    out = json.loads(result.stdout)
    # Only the offending file PATH is named; the secret value is not.
    assert any("leaked.ini" in a for a in out["forbidden_artifacts"])
    assert all(sentinel not in a for a in out["forbidden_artifacts"])


def test_cleanup_removes_container_and_tarball_in_finally(monkeypatch, tmp_path):
    # Even on failure, the temporary container and tarball must be removed.
    tarball = _build_synthetic_tarball(tmp_path, members=["/app/.git/config"])
    base = _fake_run_creating_container(tarball)
    recorded = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    # Scan failed on the forbidden .git member, but cleanup still ran.
    assert result.returncode != 0
    rm_calls = [a for a in recorded if a[:2] == ["docker", "rm"]]
    assert rm_calls, "docker rm must be issued in finally even on scan failure"
    out = json.loads(result.stdout)
    assert out["cleanup_complete"] is True


def test_synthetic_tarball_path_traversal_member_rejected(monkeypatch, tmp_path):
    # A tar member named "../../etc/passwd" -> exit 2, no file extracted outside.
    tarball = tmp_path / "traversal.tar"
    with tarfile.open(tarball, "w") as tf:
        bad = tarfile.TarInfo("../../etc/passwd")
        bad.size = 0
        tf.addfile(bad)
        normal = tarfile.TarInfo("/app/app/main.py")
        normal.size = 0
        tf.addfile(normal)
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2
    out = json.loads(result.stdout)
    assert any("etc/passwd" in a or ".." in a for a in out["forbidden_artifacts"])
    # No member is ever normalized into the image root as a result of the escape.
    assert not (tmp_path / "etc" / "passwd").exists()


def test_synthetic_tarball_symlink_member_rejected(monkeypatch, tmp_path):
    # A symlink member pointing outside /app -> exit 2.
    tarball = tmp_path / "symlink.tar"
    with tarfile.open(tarball, "w") as tf:
        link = tarfile.TarInfo("/app/evil-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
        normal = tarfile.TarInfo("/app/app/main.py")
        normal.size = 0
        tf.addfile(normal)
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2
    out = json.loads(result.stdout)
    assert any("evil-link" in a or "symlink" in a.lower()
               for a in out["forbidden_artifacts"])


def test_synthetic_tarball_whiteout_honored(monkeypatch, tmp_path):
    # A whiteout marker (.wh.foo) correctly hides the foo member in the
    # final merged filesystem.
    tarball = tmp_path / "whiteout.tar"
    with tarfile.open(tarball, "w") as tf:
        for name in ["/app/.wh.leaked", "/app/leaked/.env", "/app/app/main.py"]:
            info = tarfile.TarInfo(name)
            info.size = 0
            tf.addfile(info)
    monkeypatch.setattr(subprocess, "run", _fake_run_creating_container(tarball))
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    # Honoring .wh.leaked hides /app/leaked/.env, so the scan passes.
    assert result.returncode == 0, result.stdout + result.stderr
    out = json.loads(result.stdout)
    assert out["forbidden_artifacts"] == []
    assert out["status"] == "passed"
    assert not any("leaked" in a or ".env" in a
                   for a in out["forbidden_artifacts"])


def test_create_uses_entrypoint_bin_true(monkeypatch):
    # Inspect the docker create argv; assert --entrypoint /bin/true.
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)
    recorded = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    create_calls = [a for a in recorded if "create" in a and a[:1] == ["docker"]]
    assert create_calls, "expected at least one docker create call"
    for call in create_calls:
        assert "--entrypoint" in call
        idx = call.index("--entrypoint")
        assert call[idx + 1] == "/bin/true"


def test_export_uses_docker_export_to_host_tar(monkeypatch):
    # Inspect the export argv; assert docker export <id> > tar.
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)
    recorded = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    export_calls = [a for a in recorded if "export" in a and a[:1] == ["docker"]]
    assert export_calls, "expected at least one docker export call"
    for call in export_calls:
        # Form is exactly ["docker", "export", <container_id>]; no -o flag and
        # no shell redirection (stdout is captured to the host tar by the script).
        assert call[:2] == ["docker", "export"]
        assert "-o" not in call
        assert len(call) == 3


def test_no_shell_true_in_any_subprocess_call(monkeypatch):
    # Every subprocess.run issued by the scanner must use shell=False (default).
    import tempfile
    scratch = Path(tempfile.mkdtemp())
    tarball = _build_synthetic_tarball(scratch, members=[])
    base = _fake_run_creating_container(tarball)
    seen_kwargs = []

    def fake_run(argv, **kwargs):
        seen_kwargs.append(kwargs)
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert seen_kwargs, "expected subprocess.run calls"
    for kwargs in seen_kwargs:
        msg = "scanner must never invoke subprocess.run with shell=True"
        assert kwargs.get("shell", False) is False, f"{msg}: {kwargs}"


def test_exit_two_on_cleanup_failure(monkeypatch, tmp_path):
    # Mock `docker rm` to fail; assert exit 2 even if scan otherwise passed.
    tarball = _build_synthetic_tarball(tmp_path, members=[])  # clean image
    base = _fake_run_creating_container(tarball)

    def fake_run(argv, **kwargs):
        if argv[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="rm failed")
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2


def test_exit_two_on_export_failure(monkeypatch, tmp_path):
    # Mock `docker export` to fail; assert exit 2 and partial tarball removed.
    tarball = _build_synthetic_tarball(tmp_path, members=[])
    base = _fake_run_creating_container(tarball)
    recorded = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        if argv[:2] == ["docker", "export"]:
            return subprocess.CompletedProcess(argv, 1, stdout=b"",
                                               stderr="export failed")
        return base(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = subprocess.run(_argv("manifest.json"), capture_output=True, check=False)
    assert result.returncode == 2
    # The finally clause must still remove the temporary container.
    assert any(a[:2] == ["docker", "rm"] for a in recorded)
