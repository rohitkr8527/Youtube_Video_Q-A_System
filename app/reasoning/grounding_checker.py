from __future__ import annotations

from app.llm.groq_client import GroqLLM
from app.schemas.reasoning import GroundingResult
from app.schemas.retrieval import RetrievedChunk


SYSTEM = """You are a grounding verifier.
Judge whether every important factual claim in the answer is supported by the supplied transcript evidence.
Minor wording changes and reasonable paraphrases are allowed.
Do not require outside knowledge. Unsupported details, invented facts, or claims not present in evidence make grounded=false.
"""


class GroundingChecker:
    def __init__(self, llm: GroqLLM) -> None:
        self.llm = llm

    def check(self, question: str, answer: str, evidence: list[RetrievedChunk]) -> GroundingResult:
        context = "\n\n".join(f"[Chunk {c.chunk_id}] {c.text}" for c in evidence)
        return self.llm.structured(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence:\n{context}",
                },
            ],
            GroundingResult,
            name="grounding_check",
            reasoning_effort="low",
            max_tokens=600,
        )
