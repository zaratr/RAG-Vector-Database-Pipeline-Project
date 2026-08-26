import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest


def test_prepare_normalizes_task_id_by_lowercasing_and_hyphenating_non_alnum_runs(tmp_path, monkeypatch):
    from scripts.prepare_phase10_task import normalize_task_id

    assert normalize_task_id("10A.1") == "10a-1"
    assert normalize_task_id("10D.2") == "10d-2"
    assert normalize_task_id("DOC.1") == "doc-1"


def test_prepare_writes_source_manifest_and_binding_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scripts.prepare_phase10_task.subprocess.run", mock.Mock(returncode=0))

    from scripts.prepare_phase10_task import prepare_task

    prepare_task(task="10A.1", expected_head="a6e2c4f8b1d9")

    assert (tmp_path / "reports" / "task-10a-1-source-manifest.json").exists()
    assert (tmp_path / "reports" / "task-10a-1-source-binding.json").exists()


def test_prepare_invokes_docker_compose_build_with_argv_array_not_shell(monkeypatch, tmp_path):
    captured = []
    monkeypatch.chdir(tmp_path)

    def fake_run(argv, **kwargs):
        captured.append(argv)
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.prepare_phase10_task.subprocess.run", fake_run)

    from scripts.prepare_phase10_task import prepare_task

    prepare_task(task="10A.1", expected_head="a6e2c4f8b1d9")

    build_calls = [c for c in captured if "build" in c]
    assert build_calls
    for call in build_calls:
        assert isinstance(call, list)  # never a string
        assert "--no-cache" in call


def test_prepare_exits_2_on_build_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_run(argv, **kwargs):
        return mock.Mock(returncode=1, stdout="", stderr="build failed")

    monkeypatch.setattr("scripts.prepare_phase10_task.subprocess.run", fake_run)

    from scripts.prepare_phase10_task import prepare_task

    with pytest.raises(SystemExit) as exc_info:
        prepare_task(task="10A.1", expected_head="a6e2c4f8b1d9")

    assert exc_info.value.code == 2


def test_prepare_exits_2_on_alembic_head_mismatch(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    call_count = [0]

    def fake_run(argv, **kwargs):
        call_count[0] += 1
        if "current" in argv:
            return mock.Mock(returncode=0, stdout="wrong_head (head)\n", stderr="")
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.prepare_phase10_task.subprocess.run", fake_run)

    from scripts.prepare_phase10_task import prepare_task

    with pytest.raises(SystemExit) as exc_info:
        prepare_task(task="10A.1", expected_head="a6e2c4f8b1d9")

    assert exc_info.value.code == 2


def test_prepare_with_migration_owner_runs_migrate_before_api_recreation(monkeypatch, tmp_path):
    argv_sequence = []
    monkeypatch.chdir(tmp_path)

    def fake_run(argv, **kwargs):
        argv_sequence.append(list(argv))
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.prepare_phase10_task.subprocess.run", fake_run)

    from scripts.prepare_phase10_task import prepare_task

    prepare_task(task="10A.3", expected_head="b7f3d5a9c2e1", migration_owner=True)

    migrate_indices = [i for i, argv in enumerate(argv_sequence) if "run" in argv and "migrate" in argv]
    up_indices = [i for i, argv in enumerate(argv_sequence) if "up" in argv and "-d" in argv]
    assert migrate_indices
    assert up_indices
    assert migrate_indices[0] < up_indices[0]  # migrate runs before force-recreate


def test_prepare_never_passes_credentials_in_argv(monkeypatch, tmp_path):
    captured_argv = []
    monkeypatch.chdir(tmp_path)

    def fake_run(argv, **kwargs):
        captured_argv.append(list(argv))
        return mock.Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("scripts.prepare_phase10_task.subprocess.run", fake_run)

    from scripts.prepare_phase10_task import prepare_task

    prepare_task(task="10A.1", expected_head="a6e2c4f8b1d9")

    for argv in captured_argv:
        joined = " ".join(argv)
        assert "token" not in joined.lower()
        assert "password" not in joined.lower()
        assert "secret" not in joined.lower()


def test_prepare_re_runs_keep_delivery_tree_hash_stable(monkeypatch, tmp_path):
    """Re-running preparation into a non-ignored reports dir must not change
    ``delivery_tree_sha256``: the manifest/binding outputs are excluded from
    the fingerprint via the reports dir (regression: without passing
    ``reports_dir`` the second run's fingerprint absorbed the first run's
    output files and drifted)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], check=True)
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], check=True)
    monkeypatch.setattr(
        "scripts.prepare_phase10_task.subprocess.run", mock.Mock(returncode=0)
    )

    from scripts.prepare_phase10_task import prepare_task

    def delivery_hash() -> str:
        manifest = json.loads(
            (repo / "reports" / "task-10a-1-source-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        return manifest["delivery_tree_sha256"]

    prepare_task(task="10A.1", expected_head="a6e2c4f8b1d9")
    first = delivery_hash()
    prepare_task(task="10A.1", expected_head="a6e2c4f8b1d9")

    assert delivery_hash() == first
