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
    embedding_provider: Literal["local", "fastembed", "openai"] = Field(default="fastembed")
    embedding_model: str = Field(default="jinaai/jina-clip-v1")
    vector_store: Literal["chroma"] = Field(default="chroma")
    llm_provider: Literal["dummy", "ollama", "openai"] = Field(default="ollama")
    llm_base_url: str = Field(default="http://localhost:11434/v1")
    llm_model: str = Field(default="gemma4:latest")

    openai_api_key: Optional[str] = Field(default=None)
    chroma_host: Optional[str] = Field(default=None)
    chroma_port: int = Field(default=8000)
    chroma_persist_directory: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""

    return Settings()
