"""Operator CLI for querying the persisted GraphRAG relationship graph.

Phase 10A.5: migrated to ``retrieve_graph_paths`` returning complete
``GraphPath`` objects, with ``--hops`` capped at 1–3 and ``--direction`` support.
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
from app.services.graph_retrieval import retrieve_graph_paths


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
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    with session_scope() as session:
        paths = retrieve_graph_paths(
            session,
            query=args.query,
            max_hops=args.hops,
            direction=args.direction,
            limit=args.limit,
        )
    # Serialize complete GraphPathStep objects (pydantic .model_dump()).
    payload = [path.model_dump() for path in paths]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()