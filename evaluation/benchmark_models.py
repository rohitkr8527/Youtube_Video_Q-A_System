from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimeRange(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class BenchmarkCase(StrictModel):
    case_id: str
    video_id: str
    video_url: str
    video_title: str
    question: str
    category: Literal[
        "factual",
        "conceptual",
        "exact_term",
        "comparison",
        "multi_part",
        "explanation",
        "unanswerable",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    answerable: bool
    reference_answer: str
    relevant_timestamps: list[TimeRange] = Field(default_factory=list)
    reviewed: bool = False


from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class GeneratedAnswerableCase(BaseModel):
    # Ignore accidental extra fields returned by the LLM.
    model_config = ConfigDict(extra="ignore")

    question: str

    reference_answer: str = Field(
        validation_alias=AliasChoices(
            "reference_answer",
            "answer",
        )
    )

    relevant_start: float = Field(
        validation_alias=AliasChoices(
            "relevant_start",
            "start",
            "start_time",
        )
    )

    relevant_end: float = Field(
        validation_alias=AliasChoices(
            "relevant_end",
            "end",
            "end_time",
        )
    )


class GeneratedUnanswerableQuestions(StrictModel):
    questions: list[str]


class JudgeScores(StrictModel):
    faithfulness: float = Field(ge=0, le=1)
    answer_relevance: float = Field(ge=0, le=1)
    reason: str
