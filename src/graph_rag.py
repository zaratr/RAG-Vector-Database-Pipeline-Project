"""Operator CLI for querying the persisted GraphRAG relationship graph.

Phase 10A.5: migrated to ``retrieve_graph_paths`` returning complete
``GraphPath`` objects, with ``--hops`` capped at 1–3, ``--direction`` support,
and ``--filters`` accepting repeated ``key=value`` document filters from the
scalar filter matrix (document_id, title, source, tags).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.db import session_scope
from app.services.graph_retrieval import UnsupportedGraphFilter, retrieve_graph_paths


def _parse_filters(parser: argparse.ArgumentParser, raw: list[str] | None) -> dict:
    """Parse repeated ``key=value`` occurrences into the scalar filter matrix.

    Only ``document_id`` is converted to ``int``; a value that is not an
    integer literal is passed through unchanged so the service raises the
    plan-pinned ``UnsupportedGraphFilter`` for invalid integer forms (the
    service owns the filter contract, not this CLI).
    """
    filters: dict = {}
    for item in raw or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(
                "--filters must be KEY=VALUE pairs using the scalar filter "
                "matrix keys: document_id, title, source, tags"
            )
        if key == "document_id":
            try:
                value = int(value)
            except ValueError:
                pass
        filters[key] = value
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Traverse persisted GraphRAG relationships with provenance."
    )
    parser.add_argument("query", help="Question or entity name used to seed traversal")
    parser.add_argument("--hops", type=int, default=2, choices=range(1, 4))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--direction", choices=("outbound", "inbound", "both"), default="outbound"
    )
    parser.add_argument(
        "--filters",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="document filter as KEY=VALUE; repeatable "
        "(document_id, title, source, tags)",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    filters = _parse_filters(parser, args.filters)

    try:
        with session_scope() as session:
            paths = retrieve_graph_paths(
                session,
                query=args.query,
                max_hops=args.hops,
                direction=args.direction,
                limit=args.limit,
                filters=filters or None,
            )
    except UnsupportedGraphFilter as exc:
        parser.error(str(exc))
    # Serialize complete GraphPathStep objects (pydantic .model_dump()).
    payload = [path.model_dump() for path in paths]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()