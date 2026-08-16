"""Application configuration using Pydantic settings."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    app_name: str = Field(default="RAG Pipeline API")
    debug: bool = Field(default=False)  # never ship tracebacks in 500 bodies
    database_url: str = Field(default="sqlite:///./rag.db")
    embedding_provider: Literal["local", "fastembed", "openai"] = Field(default="fastembed")
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

    # 10B.2 provenance/security settings.
    source_trust_policy_path: str = Field(default="/app/config/source-trust-policy.json")
    operator_api_enabled: bool = Field(default=False)
    operator_token: SecretStr = Field(default=SecretStr(""))
    security_audit_retention_days: int = Field(default=30)

    # 10B.3 ingestion limits + retrieval security.
    retrieval_security_policy_path: str = Field(default="/app/config/retrieval-security-policy.json")
    context_security_policy_path: str = Field(default="/app/config/context-security-policy.json")
    ingestion_request_max_bytes: int = Field(default=11534336, ge=2048, le=53477376)
    ingestion_file_max_bytes: int = Field(default=10485760, ge=1024, le=52428800)
    ingestion_extracted_max_bytes: int = Field(default=5242880, ge=1024, le=26214400)
    ingest_rate_limit_requests: int = Field(default=30, ge=1, le=1000)
    ingest_rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    # 10C.1 content safety.
    content_safety_enabled: bool = Field(default=False)
    content_safety_policy_path: str = Field(
        default="/app/config/content-safety-policy.json"
    )
    safety_llm_mode: str = Field(default="disabled")

    openai_api_key: Optional[str] = Field(default=None)
    chroma_host: Optional[str] = Field(default=None)
    chroma_port: int = Field(default=8000)
    chroma_persist_directory: Optional[str] = Field(default=None)

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _validate_operator_token(self):
        """When operator API is enabled, token must be at least 32 characters."""
        if self.operator_api_enabled:
            token = self.operator_token.get_secret_value()
            if not token or len(token) < 32:
                raise ValueError(
                    "RAG_OPERATOR_TOKEN must contain at least 32 characters "
                    "when RAG_OPERATOR_API_ENABLED=true"
                )
        return self

    @model_validator(mode="after")
    def _validate_envelope_greater_than_file(self):
        """Startup requires the request-envelope limit to be greater than the file limit."""
        if self.ingestion_request_max_bytes <= self.ingestion_file_max_bytes:
            raise ValueError(
                "RAG_INGESTION_REQUEST_MAX_BYTES must be greater than "
                "RAG_INGESTION_FILE_MAX_BYTES"
            )
        return self

    @model_validator(mode="after")
    def _validate_safety_llm_mode(self):
        """10C.1: the LLM safety mode is a closed set."""
        if self.safety_llm_mode not in ("disabled", "rules_only", "fail_closed"):
            raise ValueError(
                "RAG_SAFETY_LLM_MODE must be one of disabled|rules_only|fail_closed"
            )
        return self


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""

    return Settings()
