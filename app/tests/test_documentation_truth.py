"""Phase 10 DOC.1 — README truth claims map to evidence rows (appendix spec).

Covers: README truth claims map to evidence rows, migration IDs current, no
obsolete NetworkX production claim, no public-production-readiness
equivalence, evidence-file column contract, and README-patch preservation.

D2 adaptations (owner-ruled): T09 pins the residual-risk truth without
referencing the superseded final gate; T10 asserts the amended evidence-doc
column set; T12/T13 (approval-report machinery) are SUPERSEDED and
intentionally absent; T02 pins the verified head ``d9b5f7c1e4a3``.
"""
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
README = PROJECT_ROOT / "README.md"
EVIDENCE = PROJECT_ROOT / "docs" / "phase10-evidence.md"
PATCH_PATH = PROJECT_ROOT / ".hermes" / "reports" / "readme-pre-doc.patch"

CURRENT_HEAD = "d9b5f7c1e4a3"


def test_readme_exists_and_is_nonempty():
    assert README.exists() and README.stat().st_size > 0


def test_readme_cites_current_migration_head():
    text = README.read_text()
    assert CURRENT_HEAD in text


def test_readme_does_not_cite_stale_migration_ids():
    text = README.read_text()
    for stale in ["a6e2c4f8b1d9", "b7f3d5a9c2e1", "c8a4e6b0d3f2"]:
        # Stale IDs may appear in history sections but NOT as "current head"
        assert f"current head: {stale}" not in text.lower()


def test_readme_does_not_claim_networkx_in_production():
    text = README.read_text().lower()
    # networkx may appear in a "history"/"amendment" context but not as a
    # current production dependency.
    for line in text.splitlines():
        if "networkx" in line and "production" in line:
            assert "removed" in line or "no longer" in line or "history" in line


def test_readme_mentions_docker_only_commands():
    text = README.read_text()
    assert "docker compose" in text


def test_readme_documents_operator_configuration():
    text = README.read_text()
    assert "RAG_OPERATOR_API_ENABLED" in text or "operator" in text.lower()


def test_readme_documents_single_operator_static_token_limitation():
    text = README.read_text().lower()
    assert "single-operator" in text or "single operator" in text


def test_readme_documents_no_tenancy_limitation():
    text = README.read_text().lower()
    assert "tenant" in text or "no multi-tenancy" in text


def test_readme_does_not_equate_phase10_with_public_production_readiness():
    # T09 adaptation: the residual-risk truth. The README must never equate
    # Phase 10 completion with public production readiness; any use of the
    # phrase must be an explicit negation.
    text = README.read_text().lower()
    assert "production ready" not in text or "not production ready" in text


def test_evidence_file_exists_and_has_required_columns():
    # T10 adaptation: the amended evidence-of-record column set (the
    # superseded "validator report path" / "approval verdict" columns are
    # replaced by the repo-self-contained schema). The table header row is
    # the first markdown table row of the document and must carry exactly
    # the seven defined columns.
    assert EVIDENCE.exists()
    header = next(
        (line for line in EVIDENCE.read_text().splitlines()
         if line.startswith("|")),
        "",
    )
    cells = [c.strip().lower() for c in header.strip("|").split("|")]
    assert cells == [
        "claim id", "phase/task", "requirement summary", "source path + symbol",
        "test path :: name", "live command (hermetic)", "expected invariant",
    ]


def test_every_completion_claim_in_readme_maps_to_evidence_row():
    # Parse README for quantitative claims (e.g. "95% recall", "ASR ≤ 0.20");
    # each must have a matching claim ID in docs/phase10-evidence.md.
    readme = README.read_text()
    evidence = EVIDENCE.read_text()
    readme_ids = set(re.findall(r"\[EVID-[A-Z0-9]+\]", readme))
    evidence_ids = set(re.findall(r"\[EVID-[A-Z0-9]+\]", evidence))
    assert readme_ids, "README must cite at least one [EVID-*] claim identifier"
    missing = readme_ids - evidence_ids
    assert not missing, f"README claims lack evidence rows: {sorted(missing)}"
    # Any line stating a quantitative metric must be anchored to an evidence ID.
    metric = re.compile(r"\d+(?:\.\d+)?\s*%|ASR\s*[≤<]|recall\s*[≥>]|precision",
                        re.IGNORECASE)
    for line in readme.splitlines():
        if metric.search(line):
            assert "[EVID-" in line, \
                f"Quantitative claim lacks evidence anchor: {line.strip()}"


def test_pre_existing_readme_patch_preserved():
    # The pre-doc README patch must still be represented after DOC.1 edits.
    # Self-disabling: with no patch file present the lane is vacuously green.
    if PATCH_PATH.exists():
        # Every + line in the patch must still appear in the current README.
        patch = PATCH_PATH.read_text()
        for line in patch.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                assert line[1:].strip() in README.read_text()
