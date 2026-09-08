from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    video_id: str
    question: str = Field(min_length=1, max_length=4000)
    history: list[ChatTurn] = Field(default_factory=list)


class SourceCitation(BaseModel):
    label: str
    start_time: float
    end_time: float
    timestamp: str
    url: str
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    insufficient_evidence: bool = False


class ContentRequest(BaseModel):
    video_id: str


class ContentResponse(BaseModel):
    content: str
    sources: list[SourceCitation] = Field(default_factory=list)


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source: SourceCitation | None = None


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
