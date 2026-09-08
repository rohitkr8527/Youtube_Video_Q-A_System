from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.llm.groq_client import GroqLLM


class GenerationGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    faithfulness: float
    answer_relevance: float
    notes: str


def grade_generation(question: str, answer: str, evidence: str) -> GenerationGrade:
    llm = GroqLLM()
    return llm.structured(
        [
            {
                "role": "system",
                "content": "Score answer faithfulness and answer relevance from 0.0 to 1.0 using only the supplied evidence. Be strict.",
            },
            {"role": "user", "content": f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence}"},
        ],
        GenerationGrade,
        name="generation_grade",
        reasoning_effort="low",
        max_tokens=500,
    )
