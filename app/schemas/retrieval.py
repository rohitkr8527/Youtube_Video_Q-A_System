from __future__ import annotations

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: int
    video_id: str
    title: str
    text: str
    start_time: float
    end_time: float
    topic: str = ""
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
