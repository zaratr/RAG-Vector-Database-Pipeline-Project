"""Pydantic schemas for API requests and responses."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    text: Optional[str] = None


class DocumentSummary(BaseModel):
    id: int
    title: str
    source: Optional[str] = None
    tags: List[str] = []
    chunk_count: int


class ChunkSchema(BaseModel):
    id: int
    index: int
    text: str
    start_offset: int
    end_offset: int


class DocumentDetail(BaseModel):
    id: int
    title: str
    source: Optional[str] = None
    tags: List[str] = []
    chunks: List[ChunkSchema]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: Optional[dict] = None
    retrieval_mode: Literal["vector", "graph", "hybrid"] = "vector"
    graph_max_hops: int = Field(default=2, ge=1, le=3)


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: dict
    vector_id: Optional[str] = Field(default=None, exclude=True)


class QueryResponse(BaseModel):
    answer: str
    context: List[RetrievedChunk]
