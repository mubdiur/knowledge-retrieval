"""Application configuration — loaded from environment with sensible defaults."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Knowledge Retrieval System"
    debug: bool = False
    log_level: str = "INFO"

    # PostgreSQL
    postgres_user: str = "knowrob"
    postgres_password: str = "knowrob_secret_2026"
    postgres_db: str = "knowrob"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def database_url_sync(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking
    max_chunk_size: int = 1024
    chunk_overlap: int = 128

    # Redis (optional)
    redis_url: str | None = None
    cache_ttl: int = 3600

    # Qdrant collection names
    vector_collection: str = "knowledge_docs"
    log_collection: str = "knowledge_logs"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
