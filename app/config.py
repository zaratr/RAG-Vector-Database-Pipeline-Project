"""Application configuration using Pydantic settings."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    app_name: str = Field(default="RAG Pipeline API")
    debug: bool = Field(default=True)
    database_url: str = Field(default="sqlite:///./rag.db")
    embedding_provider: Literal["local", "openai"] = Field(default="local")
    vector_store: Literal["chroma"] = Field(default="chroma")

    openai_api_key: Optional[str] = Field(default=None)
    chroma_persist_directory: Optional[str] = Field(default=None)

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""

    return Settings()
