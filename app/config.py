"""
app/config.py
Centralised configuration using pydantic-settings.
All values come from environment variables — nothing hardcoded.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM / OpenRouter ─────────────────────────────
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    llm_model: str = Field(
        default="openrouter/openai/gpt-4o-mini",
        description="LiteLLM model string for OpenRouter",
    )
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=4096, ge=256, le=32000)

    # ── Clerk Auth ───────────────────────────────────
    clerk_publishable_key: str = Field(default="", description="Clerk publishable key")
    clerk_secret_key: str = Field(default="", description="Clerk secret key")
    clerk_jwt_audience: str = Field(default="", description="Clerk JWT audience")
    auth_bypass: bool = Field(
        default=False,
        description="Set True to bypass auth in dev mode",
    )

    # ── LangSmith ────────────────────────────────────
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(default="intelliresearch")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    # ── External APIs ────────────────────────────────
    serpapi_key: str = Field(default="", description="SerpAPI key for web search")

    # ── FastAPI ──────────────────────────────────────
    backend_host: str = Field(default="0.0.0.0")
    backend_port: int = Field(default=8000)
    cors_origins: str = Field(default="http://localhost:8501")
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # ── RAG / Vector Store ───────────────────────────
    faiss_index_path: str = Field(default="./data/faiss_index")
    chroma_persist_dir: str = Field(default="./data/chroma")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k_retrieval: int = Field(default=5, ge=1, le=20)

    # ── Cache ────────────────────────────────────────
    cache_ttl_seconds: int = Field(default=3600)
    cache_max_size: int = Field(default=128)

    # ── Security ─────────────────────────────────────
    max_query_length: int = Field(default=5000)
    rate_limit_requests: int = Field(default=10)
    rate_limit_window: int = Field(default=60)

    # ── Agent Control ────────────────────────────────
    judge_quality_threshold: float = Field(default=6.0, ge=0.0, le=10.0)
    max_self_correction_loops: int = Field(default=2, ge=0, le=5)

    # ── Circuit Breaker ──────────────────────────────
    circuit_breaker_failure_threshold: int = Field(default=3)
    circuit_breaker_timeout: int = Field(default=30)

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def is_production(self) -> bool:
        """True if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def tracing_enabled(self) -> bool:
        """True if LangSmith tracing is configured."""
        return bool(self.langsmith_api_key and self.langchain_tracing_v2)

    @property
    def auth_enabled(self) -> bool:
        """True if Clerk auth is configured and bypass is disabled."""
        return bool(self.clerk_secret_key) and not self.auth_bypass


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance (singleton pattern)."""
    return Settings()
