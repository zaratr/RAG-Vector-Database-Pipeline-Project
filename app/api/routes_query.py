"""Query endpoint."""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import QueryRequest, QueryResponse
from app.services.embeddings import get_embedding_provider
from app.services.llm import LLMProviderOutputError, get_llm_client
from app.services.rag import answer_query
from app.services.graph_retrieval import GraphTraversalLimitError, UnsupportedGraphFilter
from app.services.vector_store import get_vector_store
from app.config import get_settings

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(payload: QueryRequest, session: Session = Depends(get_db)):
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    llm_client = get_llm_client(
            provider = settings.llm_provider,
            base_url = settings.llm_base_url,
            model = settings.llm_model,
            api_key = settings.openai_api_key,
    )
    try:
        result = await answer_query(
            query=payload.query,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            llm_client=llm_client,
            top_k=payload.top_k,
            filters=payload.filters,
            session=session,
            retrieval_mode=payload.retrieval_mode,
            graph_max_hops=payload.graph_max_hops,
        )
    except UnsupportedGraphFilter as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GraphTraversalLimitError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderOutputError as exc:
        # The answer provider answered but its envelope was unusable: a typed
        # 502, mirroring the graph-extraction provider-output convention.
        raise HTTPException(status_code=502, detail="LLM provider failed") from exc
    except httpx.HTTPError as exc:
        # The answer LLM is an external provider: transport failures
        # (connection refused, timeout, 5xx via raise_for_status) surface as
        # a stable, typed 503 instead of an unhandled 500 with a traceback.
        raise HTTPException(
            status_code=503, detail="LLM provider unavailable"
        ) from exc
    return QueryResponse(**result)
