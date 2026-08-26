"""Live acceptance script for real Ollama/Gemma graph extraction.

Runs the configured provider against a fixture text and prints a bounded JSON
summary to stdout. Exit codes:

* ``0`` — at least one grounded, schema-valid relation.
* ``1`` — valid successful-empty output for a fixture that requires a relation.
* ``2`` — configuration/provider/output failure (sanitized JSON error to stderr).

No source text or provider payload is ever written to stdout/stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings
from app.services.graph_extraction import (
    GraphExtractionError,
    GraphProviderOutputError,
    GraphProviderUnavailable,
    get_graph_extractor,
)

FIXTURE_TEXT = "Aria manages Project Helios. Project Helios uses Vector Engine."


def _run(text: str) -> int:
    provider = "unknown"
    try:
        settings = get_settings()
        provider = (
            "ollama" if settings.llm_provider == "ollama" else settings.llm_provider
        )
        model = settings.graph_extraction_model or settings.llm_model
        extractor = get_graph_extractor()
        relations = asyncio.run(extractor.extract(text))
    except GraphProviderUnavailable as exc:
        _emit_error(provider, "provider_unavailable", exc)
        return 2
    except GraphProviderOutputError as exc:
        _emit_error(provider, "provider_output_error", exc)
        return 2
    except GraphExtractionError as exc:
        _emit_error(provider, "extraction_error", exc)
        return 2
    except ValidationError as exc:
        _emit_error(provider, "configuration_error", exc)
        return 2

    if not relations:
        summary = {
            "provider": provider,
            "model": model,
            "relations": [],
            "grounding_valid": True,
        }
        sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
        return 1

    payload = {
        "provider": provider,
        "model": model,
        "relations": [
            {
                "source": r.source.name,
                "predicate": r.predicate,
                "target": r.target.name,
                "evidence_start": r.evidence_start,
                "evidence_end": r.evidence_end,
            }
            for r in relations
        ],
        "grounding_valid": True,
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    return 0


def _emit_error(provider: str, code: str, exc: Exception) -> None:
    """Emit a sanitized error JSON to stderr; never include raw text/payload."""
    sys.stderr.write(
        json.dumps(
            {"provider": provider, "error": code, "detail": str(exc)[:200]},
            sort_keys=True,
        )
        + "\n"
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate real graph extraction.")
    parser.add_argument("--text", default=FIXTURE_TEXT)
    args = parser.parse_args(argv)
    return _run(args.text)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
