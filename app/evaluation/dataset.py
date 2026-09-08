from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    video_id: str
    question: str
    relevant_chunk_ids: list[int] = Field(default_factory=list)
    expected_answer: str = ""


def load_dataset(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(EvaluationCase.model_validate_json(line))
    return cases
