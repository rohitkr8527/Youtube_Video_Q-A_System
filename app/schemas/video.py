from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class ProcessVideoRequest(BaseModel):
    url: str = Field(min_length=10, max_length=500)


class VideoInfo(BaseModel):
    video_id: str
    title: str
    url: str
    thumbnail_url: str
    chunk_count: int
    duration_seconds: float | None = None


class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float


class VideoChunk(BaseModel):
    video_id: str
    title: str
    chunk_id: int
    start_time: float
    end_time: float
    topic: str
    text: str
