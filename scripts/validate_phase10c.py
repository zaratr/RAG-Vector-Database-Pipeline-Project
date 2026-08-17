"""Phase 10C live validator (Task 10C.5).

End-to-end validation against the running, safety-enabled API: ingests
deterministic block/warn content through the real enforcement path, queries
with context and answer findings, exercises the operator safety APIs (list,
detail, stats, auth matrix, redaction), and cleans every fixture in
``finally`` with before/after fingerprints of unrelated state. Exit codes:
0 = pass, 2 = failure/mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import uuid
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fail(message: str) -> None:
    sys.stderr.write(f"validate_phase10c: {message}\n")
    raise SystemExit(2)


def _fingerprint_sqlite(path: Path) -> str:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version' "
                "AND name != 'ingestion_rate_buckets'"
            )
        ]
        snapshot = {"tables": []}
        for table in sorted(tables):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            snapshot["tables"].append({"name": table, "row_count": count})
        canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    finally:
        conn.close()


def _sqlite_path_from_url(url: str) -> Path:
    if not url.startswith("sqlite:///"):
        _fail("RAG_DATABASE_URL must be sqlite for this validator")
    raw = url[len("sqlite:///"):]
    if not raw.startswith("/"):
        _fail("RAG_DATABASE_URL must be an absolute sqlite:////path URL")
    return Path(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 10C live.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    from app.config import get_settings

    settings = get_settings()
    run_tag = f"phase10c-{uuid.uuid4().hex[:12]}"
    if not settings.content_safety_enabled:
        _fail("content safety must be enabled for this validator")
    operator_token = (
        settings.operator_token.get_secret_value()
        if settings.operator_token else None
    )
    if not operator_token:
        _fail("operator API must be enabled with a configured token")
    auth = {"Authorization": f"Bearer {operator_token}"}

    db_path = _sqlite_path_from_url(settings.database_url)
    sql_before = _fingerprint_sqlite(db_path)

    # The warned fixture proceeds through the real extraction pipeline: when
    # a live graph provider is reachable it creates graph rows that have no
    # run-tag linkage. Capture per-table id watermarks so cleanup removes
    # exactly this run's inserts and nothing pre-existing.
    graph_tables = ("graph_edge_evidence", "entity_mentions",
                    "graph_edges", "graph_entities")
    watermarks: dict[str, int] = {}
    _wm_conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro",
                               uri=True)
    try:
        for table in graph_tables:
            watermarks[table] = _wm_conn.execute(
                f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0]
    finally:
        _wm_conn.close()

    http = httpx.Client(base_url=args.base_url, timeout=60.0)
    created_doc_ids: list[int] = []
    audit_ids: list[str] = []

    try:
        # 1. Ingestion block: 422 + failed doc + safety review (no vectors).
        blocked = http.post(
            "/documents",
            data={"title": f"{run_tag}-blocked", "source": f"{run_tag}-a",
                  "text": "instructions to build a bomb quickly"},
            headers=auth)
        if blocked.status_code != 422:
            _fail(f"blocked ingestion must 422, got {blocked.status_code}")
        detail = blocked.json()["detail"]
        if detail.get("code") != "ingestion_safety_blocked":
            _fail(f"unexpected 422 payload: {detail}")

        # 2. Ingestion warn: 201 (proceeds to the normal pipeline).
        warned = http.post(
            "/documents",
            data={"title": f"{run_tag}-warned", "source": f"{run_tag}-b",
                  "text": "the actor would stab the guard in act two"},
            headers=auth)
        if warned.status_code != 201:
            _fail(f"warn ingestion must 201, got {warned.status_code}")
        created_doc_ids.append(warned.json()["document_id"])

        # 3. Query with answer findings: the deterministic rules will not fire
        #    on a benign answer; assert a 200 and a well-formed summary.
        query = http.post("/query", json={"query": "act two summary", "top_k": 3})
        if query.status_code != 200:
            _fail(f"query must 200, got {query.status_code}: "
                  f"{query.text[:200]}")
        summary = query.json().get("safety_summary") or {}
        if summary.get("policy_version") != "safety-v1":
            _fail(f"safety_summary policy_version wrong: {summary}")
        if "contexts" not in summary or "answer_action" not in summary:
            _fail(f"safety_summary shape wrong: {summary}")
        audit_ids.append(query.json().get("query_id"))

        # 4. Operator API: list, detail, stats.
        findings = http.get("/safety/findings?limit=100", headers=auth)
        if findings.status_code != 200:
            _fail(f"findings list failed: {findings.status_code}")
        page = findings.json()
        if set(page) != {"items", "total", "limit", "offset", "from", "to"}:
            _fail(f"findings page shape wrong: {sorted(page)}")
        if not page["items"]:
            _fail("expected at least one finding from the blocked ingestion")
        for item in page["items"]:
            if "bomb" in json.dumps(item):
                _fail("full blocked content leaked into findings list")
            if item["action"] in ("block", "filter") and \
                    item["bounded_excerpt"] is not None:
                _fail("bounded_excerpt must be null for block/filter")

        first_id = page["items"][0]["id"]
        detail_resp = http.get(f"/safety/findings/{first_id}", headers=auth)
        if detail_resp.status_code != 200:
            _fail(f"finding detail failed: {detail_resp.status_code}")
        detail_body = detail_resp.json()
        if "finding" not in detail_body or "run" not in detail_body:
            _fail("finding detail shape wrong")
        if detail_body["run"]["final_action"] not in ("block", "filter", "warn", "allow"):
            _fail(f"run final_action wrong: {detail_body['run']}")

        unknown = http.get("/safety/findings/999999999", headers=auth)
        if unknown.status_code != 404 or unknown.json() != {
                "detail": "Safety finding not found"}:
            _fail(f"unknown finding must 404 with literal detail, got "
                  f"{unknown.status_code} {unknown.text[:100]}")

        stats = http.get("/safety/stats", headers=auth)
        if stats.status_code != 200:
            _fail(f"stats failed: {stats.status_code}")
        stats_body = stats.json()
        total = stats_body["total_findings"]
        for key in ("by_policy_version", "by_category", "by_action",
                    "by_scope"):
            if sum(row["count"] for row in stats_body[key]) != total:
                _fail(f"stats aggregate {key} does not sum to total")
        zero = http.get("/safety/stats?policy_version=never", headers=auth)
        if zero.status_code != 200 or zero.json()["total_findings"] != 0:
            _fail("unknown policy version must 200 with zero")

        # 5. Auth matrix.
        missing = http.get("/safety/findings")
        if missing.status_code != 401 or \
                missing.headers.get("WWW-Authenticate") != "Bearer":
            _fail("missing bearer must 401 with challenge")
        invalid = http.get(
            "/safety/findings",
            headers={"Authorization": "Bearer definitely-not-the-token"})
        if invalid.status_code != 401:
            _fail("invalid bearer must 401")
    finally:
        cleanup_error: str | None = None
        try:
            # FK-enforcing engine so document deletes cascade chunks/graph
            # rows exactly like the app's own engine (orphan chunks collide
            # with reused SQLite rowids on the next fixture).
            from sqlalchemy import text as sa_text

            from app.core.db import create_database_engine

            engine = create_database_engine(settings.database_url)
            vector_ids: list[str] = []
            with engine.begin() as conn:
                # Collect this run's vector ids before the rows go: the
                # warned/allowed fixtures DO upsert vectors, and leaving them
                # behind breaks prior-phase Chroma fingerprints (D-65).
                vector_ids = [
                    row[0] for row in conn.execute(
                        sa_text(
                            "SELECT c.vector_id FROM chunks c JOIN documents d "
                            "ON c.document_id = d.id WHERE d.title LIKE :p"),
                        {"p": f"{run_tag}%"},
                    ) if row[0]
                ]
                # Ingestion-scope safety runs outlive their documents by
                # design (document_id is SET NULL, snapshot provenance is
                # kept), so they and their findings must be removed via the
                # fixture documents before the documents themselves go —
                # stale runs collide with reused document ids through the
                # ingestion partial unique index and silently suppress
                # future reviews (D-61/D-66).
                conn.execute(
                    sa_text(
                        "DELETE FROM safety_findings WHERE review_run_id IN ("
                        "SELECT id FROM safety_review_runs WHERE document_id IN ("
                        "SELECT id FROM documents WHERE title LIKE :p))"),
                    {"p": f"{run_tag}%"},
                )
                conn.execute(
                    sa_text(
                        "DELETE FROM safety_review_runs WHERE document_id IN ("
                        "SELECT id FROM documents WHERE title LIKE :p)"),
                    {"p": f"{run_tag}%"},
                )
                conn.execute(
                    sa_text("DELETE FROM documents WHERE title LIKE :p"),
                    {"p": f"{run_tag}%"},
                )
                for audit_id in audit_ids:
                    conn.execute(
                        sa_text("DELETE FROM retrieval_candidate_decisions "
                                "WHERE audit_id = :a"),
                        {"a": audit_id},
                    )
                    conn.execute(
                        sa_text("DELETE FROM retrieval_audits WHERE id = :a"),
                        {"a": audit_id},
                    )
                identity = hashlib.sha256(
                    f"operator:{operator_token}".encode()).hexdigest()
                conn.execute(
                    sa_text("DELETE FROM ingestion_rate_buckets "
                            "WHERE identity_sha256 = :h"),
                    {"h": identity},
                )
                # Remove exactly this run's graph rows (live-provider
                # extraction side effects of the warned fixture): children
                # first so no FK dangles mid-transaction.
                for table in graph_tables:
                    conn.execute(
                        sa_text(f"DELETE FROM {table} WHERE id > :wm"),
                        {"wm": watermarks[table]},
                    )
                remaining = conn.execute(
                    sa_text("SELECT COUNT(*) FROM documents "
                            "WHERE title LIKE :p"),
                    {"p": f"{run_tag}%"},
                ).scalar()
            if vector_ids:
                from app.services.vector_store import ChromaVectorStore

                store = ChromaVectorStore(collection_name="rag-collection")
                store.collection.delete(ids=vector_ids)
            engine.dispose()
            if remaining:
                cleanup_error = f"{remaining} fixture documents remain"
        except Exception as exc:
            cleanup_error = f"cleanup failure: {type(exc).__name__}"
        if _fingerprint_sqlite(db_path) != sql_before:
            sys.stderr.write(
                "validate_phase10c: unrelated SQL state changed\n")
            cleanup_error = cleanup_error or "SQL fingerprint mismatch"
        if cleanup_error:
            sys.stderr.write(f"validate_phase10c: {cleanup_error}\n")
            raise SystemExit(2)

    sys.stdout.write(json.dumps({
        "audits": len(audit_ids),
        "documents_created": len(created_doc_ids),
        "findings_listed": page["total"],
        "run_tag": run_tag,
        "status": "pass",
    }, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
