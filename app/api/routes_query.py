"""Query endpoint."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.models import QueryRequest, QueryResponse
from app.services.embeddings import get_embedding_provider
from app.services.llm import get_llm_client
from app.services.rag import answer_query
from app.services.vector_store import get_vector_store
from app.config import get_settings

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(payload: QueryRequest, session: Session = Depends(get_db)):
    settings = get_settings()
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    llm_client = get_llm_client(api_key=settings.openai_api_key, provider="dummy")
    result = await answer_query(
        query=payload.query,
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_client=llm_client,
        top_k=payload.top_k,
        filters=payload.filters,
    )
    return QueryResponse(**result)
