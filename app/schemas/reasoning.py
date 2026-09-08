from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryPlan(StrictModel):
    route: Literal[
        "FACTUAL_QA",
        "SUMMARY",
        "NOTES",
        "QUIZ",
        "TIMESTAMP_QUERY",
        "COMPARISON",
        "EXPLANATION",
    ]
    standalone_query: str
    needs_decomposition: bool
    subqueries: list[str]
    use_timestamp_filter: bool
    start_time: float | None
    end_time: float | None


class EvidenceGrade(StrictModel):
    label: Literal["RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT"]
    reason: str
    improved_query: str


class GroundingResult(StrictModel):
    grounded: bool
    unsupported_claims: list[str]
    reason: str


class QuizItemLLM(StrictModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source_chunk_id: int


class QuizLLMResponse(StrictModel):
    questions: list[QuizItemLLM]
