from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "VideoRAG"
    groq_api_key: str = Field(default="")
    groq_model: str = "openai/gpt-oss-120b"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    streamlit_port: int = 8501

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    qdrant_path: Path = Path("data/qdrant")
    video_data_path: Path = Path("data/videos")
    trace_path: Path = Path("data/logs/traces.jsonl")
    redis_url: str = ""

    target_chunk_tokens: int = 650
    min_chunk_tokens: int = 350
    max_chunk_tokens: int = 800
    overlap_tokens: int = 100
    semantic_boundary_quantile: float = 0.25

    dense_candidates: int = 20
    sparse_candidates: int = 20
    rerank_top_k: int = 6
    max_retrieval_retries: int = 1
    max_generation_retries: int = 1

    groq_timeout_seconds: int = 120
    request_timeout_seconds: int = 600

    def ensure_directories(self) -> None:
        self.qdrant_path.mkdir(parents=True, exist_ok=True)
        self.video_data_path.mkdir(parents=True, exist_ok=True)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
