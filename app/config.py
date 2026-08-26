"""Application configuration using Pydantic settings."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    app_name: str = Field(default="RAG Pipeline API")
    debug: bool = Field(default=True)
    database_url: str = Field(default="sqlite:///./rag.db")
    # "local" (deterministic hash embeddings, no model downloads) matches the
    # Dockerfile ENV default and keeps a bare environment runnable; production
    # deployments opt into "fastembed" via environment or .env.
    embedding_provider: Literal["local", "fastembed", "openai"] = Field(default="local")
    embedding_model: str = Field(default="jinaai/jina-clip-v1")
    vector_store: Literal["chroma"] = Field(default="chroma")
    llm_provider: Literal["dummy", "ollama", "openai"] = Field(default="ollama")
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_model: str = Field(default="gemma4:latest")
    graph_extraction_enabled: bool = Field(default=True)
    graph_extraction_model: Optional[str] = Field(default=None)
    graph_max_hops: int = Field(default=2, ge=1, le=3)
    # 10A.3 extraction lease duration (seconds). Default 600 (10 min); 60–3600.
    extraction_lease_seconds: int = Field(default=600, ge=60, le=3600)

    openai_api_key: Optional[str] = Field(default=None)
    chroma_host: Optional[str] = Field(default=None)
    chroma_port: int = Field(default=8000)
    chroma_persist_directory: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""

    return Settings()
