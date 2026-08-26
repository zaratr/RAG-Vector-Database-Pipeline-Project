import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.source_manifest import (
    build_manifest,
    read_field,
    MANIFEST_VERSION,
)


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    """Create a minimal git repo with tracked, untracked, deleted, and ignored files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    subprocess.run(["git", "init", "-q"], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], check=True)
    (repo / ".gitignore").write_text("*.secret\n.venv/\n")
    (repo / "app").mkdir()
    (repo / "app/main.py").write_text("print('hi')\n", encoding="utf-8")
    (repo / "README.md").write_text("# Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], check=True)
    # Untracked delivery file
    (repo / "app/new_module.py").write_text("# new\n", encoding="utf-8")
    # Ignored file
    (repo / "config.secret").write_text("password=123\n", encoding="utf-8")
    return repo


def test_source_manifest_includes_branch_commit_dirty_and_two_distinct_hashes(isolated_repo, tmp_path):
    output = tmp_path / "manifest.json"
    manifest = build_manifest(output_path=str(output))

    assert output.exists()
    assert manifest["schema"] == MANIFEST_VERSION
    assert manifest["branch"] == "master"
    assert len(manifest["commit_sha"]) == 40
    assert manifest["dirty"] is True  # untracked delivery file
    assert "porcelain_hash" in manifest
    assert "delivery_tree_sha256" in manifest
    assert "image_context_sha256" in manifest
    assert manifest["delivery_tree_sha256"] != manifest["image_context_sha256"]


def test_source_manifest_is_deterministic_across_runs_with_same_tree(isolated_repo, tmp_path):
    out1 = tmp_path / "m1.json"
    out2 = tmp_path / "m2.json"
    m1 = build_manifest(output_path=str(out1))
    m2 = build_manifest(output_path=str(out2))

    assert m1["delivery_tree_sha256"] == m2["delivery_tree_sha256"]
    assert m1["image_context_sha256"] == m2["image_context_sha256"]
    assert m1["porcelain_hash"] == m2["porcelain_hash"]


def test_source_manifest_excludes_ignored_secret_files(isolated_repo, tmp_path):
    output = tmp_path / "manifest.json"
    manifest = build_manifest(output_path=str(output))

    all_paths = json.dumps(manifest)
    assert "config.secret" not in all_paths
    assert "password" not in all_paths


def test_source_manifest_includes_deleted_files(isolated_repo, tmp_path):
    (isolated_repo / "app" / "old_module.py").write_text("# old\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add-old"], check=True)
    (isolated_repo / "app" / "old_module.py").unlink()
    subprocess.run(["git", "rm", "--cached", "-q", "app/old_module.py"], check=True)

    output = tmp_path / "manifest.json"
    manifest = build_manifest(output_path=str(output))

    all_paths = json.dumps(manifest)
    assert "old_module.py" in all_paths  # deletion recorded with status 'D'


def test_source_manifest_field_reader_returns_validated_scalar(isolated_repo, tmp_path):
    output = tmp_path / "manifest.json"
    build_manifest(output_path=str(output))

    assert len(read_field(str(output), "commit_sha")) == 40
    assert read_field(str(output), "dirty") == "True"
    assert len(read_field(str(output), "image_context_sha256")) == 64


def test_source_manifest_field_reader_rejects_unknown_field_name(isolated_repo, tmp_path):
    output = tmp_path / "manifest.json"
    build_manifest(output_path=str(output))

    with pytest.raises(ValueError, match="unknown field"):
        read_field(str(output), "nonexistent_field")


def test_source_manifest_excludes_reports_directory(isolated_repo, tmp_path):
    reports = isolated_repo / "reports"
    reports.mkdir(parents=True)
    (reports / "sensitive-output.json").write_text('{"token":"abc"}', encoding="utf-8")

    output = tmp_path / "manifest.json"
    manifest = build_manifest(output_path=str(output), reports_dir="reports")

    assert "sensitive-output" not in json.dumps(manifest)


def test_source_manifest_normalizes_plan_path_without_absolute_entries(isolated_repo, tmp_path):
    """The operator-supplied plan path must never embed machine-specific
    absolute paths, and an in-repo plan must not be double-counted."""
    # In-tree plan: tracked, so the tree walk already lists it exactly once.
    (isolated_repo / "docs").mkdir()
    (isolated_repo / "docs" / "plan.md").write_text("# Plan\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "plan"], check=True)

    in_tree = build_manifest(
        output_path=str(tmp_path / "m-in.json"),
        plan_path="docs/plan.md",
    )
    in_tree_paths = [f["path"] for f in in_tree["files"]]
    assert in_tree_paths.count("docs/plan.md") == 1
    assert str(isolated_repo) not in json.dumps(in_tree)

    # Out-of-tree plan: exactly one stable basename-label entry, no absolute
    # path string anywhere in the manifest, and stable across runs.
    external = tmp_path / "external-plan.md"
    external.write_text("# External\n", encoding="utf-8")
    out1 = build_manifest(output_path=str(tmp_path / "m-out1.json"), plan_path=str(external))
    out2 = build_manifest(output_path=str(tmp_path / "m-out2.json"), plan_path=str(external))

    out_paths = [f["path"] for f in out1["files"]]
    assert out_paths.count("external-plan.md") == 1
    assert str(tmp_path) not in json.dumps(out1)
    assert out1["delivery_tree_sha256"] == out2["delivery_tree_sha256"]


def test_source_manifest_normalizes_paths_to_posix(isolated_repo, tmp_path):
    output = tmp_path / "manifest.json"
    manifest = build_manifest(output_path=str(output))

    for entry in manifest.get("files", []):
        path = entry["path"] if isinstance(entry, dict) else entry
        assert "\\" not in path
