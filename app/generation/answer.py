from __future__ import annotations

from app.llm.groq_client import GroqLLM
from app.schemas.chat import ChatTurn
from app.schemas.retrieval import RetrievedChunk


SYSTEM = """You answer questions about one YouTube video using only supplied transcript evidence.
Rules:
1. Do not use outside knowledge as factual support.
2. If the evidence does not contain the answer, say so clearly.
3. Be direct and natural. Use simple language unless the user asks for technical depth.
4. Do not invent timestamps, source numbers, quotes, people, metrics, or details.
5. Do not mention retrieval, chunks, vector databases, rerankers, models, or internal processing.
6. Do not add a Sources section; the application adds verified timestamp sources separately.
"""


class AnswerGenerator:
    def __init__(self, llm: GroqLLM) -> None:
        self.llm = llm

    @staticmethod
    def _context(evidence: list[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[Evidence {i}] ({item.start_time:.1f}s-{item.end_time:.1f}s)\n{item.text}"
            for i, item in enumerate(evidence, start=1)
        )

    def generate(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        history: list[ChatTurn],
        correction_instruction: str | None = None,
    ) -> str:
        recent_history = "\n".join(
            f"{turn.role}: {turn.content}" for turn in history[-4:]
        ) or "No prior conversation."
        correction = f"\nCorrection requirement: {correction_instruction}" if correction_instruction else ""
        return self.llm.complete(
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Conversation context:\n{recent_history}\n\n"
                        f"Question:\n{question}\n\n"
                        f"Transcript evidence:\n{self._context(evidence)}{correction}"
                    ),
                },
            ],
            temperature=0.15,
            max_tokens=1600,
            reasoning_effort="medium",
        )
