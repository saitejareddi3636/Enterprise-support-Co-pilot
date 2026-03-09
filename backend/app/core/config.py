from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    app_env: str
    app_port: int
    database_url: str
    chunk_size: int
    chunk_overlap: int
    openai_api_key: str | None
    openai_embedding_model: str
    openai_chat_model: str
    semantic_candidates: int
    keyword_candidates: int
    rerank_enabled: bool
    rerank_model: str
    langfuse_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        app_env = os.getenv("APP_ENV", "local")
        app_port = int(os.getenv("APP_PORT", "8000"))
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://postgres:postgres@localhost:5432/enterprise_support",
        )
        chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        openai_chat_model = os.getenv(
            "OPENAI_CHAT_MODEL",
            "gpt-4o-mini",
        )
        semantic_candidates = int(os.getenv("SEMANTIC_CANDIDATES", "32"))
        keyword_candidates = int(os.getenv("KEYWORD_CANDIDATES", "32"))
        rerank_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
        rerank_model = os.getenv("RERANK_MODEL", "gpt-4o-mini")
        langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
        return cls(
            app_env=app_env,
            app_port=app_port,
            database_url=database_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            openai_api_key=openai_api_key,
            openai_embedding_model=openai_embedding_model,
            openai_chat_model=openai_chat_model,
            semantic_candidates=semantic_candidates,
            keyword_candidates=keyword_candidates,
            rerank_enabled=rerank_enabled,
            rerank_model=rerank_model,
            langfuse_enabled=langfuse_enabled,
        )


settings = Settings.from_env()

