"""Pydantic schemas for API requests and responses."""
from typing import List, Optional

from pydantic import BaseModel


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
    query: str
    top_k: int = 5
    filters: Optional[dict] = None


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    context: List[RetrievedChunk]
