from __future__ import annotations

from app.llm.groq_client import GroqLLM
from app.schemas.reasoning import EvidenceGrade
from app.schemas.retrieval import RetrievedChunk


SYSTEM = """You grade whether retrieved YouTube transcript evidence can answer a question.
Use RELEVANT when the evidence clearly supports an answer.
Use PARTIALLY_RELEVANT when it is related but incomplete and a better search query could help.
Use IRRELEVANT when the evidence does not answer the question.
Do not answer the question. improved_query should be a better standalone search query; if no rewrite is needed, repeat the original query.
"""


class EvidenceGrader:
    def __init__(self, llm: GroqLLM) -> None:
        self.llm = llm

    def grade(self, question: str, evidence: list[RetrievedChunk]) -> EvidenceGrade:
        context = "\n\n".join(
            f"[Chunk {item.chunk_id}] {item.text[:1800]}" for item in evidence
        ) or "No evidence was retrieved."
        return self.llm.structured(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
            ],
            EvidenceGrade,
            name="evidence_grade",
            reasoning_effort="low",
            max_tokens=500,
        )
