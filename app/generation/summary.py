from __future__ import annotations

from app.generation.citations import build_citations
from app.llm.groq_client import GroqLLM
from app.schemas.chat import ContentResponse
from app.schemas.retrieval import RetrievedChunk
from app.schemas.video import VideoChunk


class SummaryGenerator:
    def __init__(self, llm: GroqLLM) -> None:
        self.llm = llm

    @staticmethod
    def _batches(chunks: list[VideoChunk], max_chars: int = 48000) -> list[list[VideoChunk]]:
        batches: list[list[VideoChunk]] = []
        current: list[VideoChunk] = []
        size = 0
        for chunk in chunks:
            if current and size + len(chunk.text) > max_chars:
                batches.append(current)
                current, size = [], 0
            current.append(chunk)
            size += len(chunk.text)
        if current:
            batches.append(current)
        return batches

    def generate(self, chunks: list[VideoChunk]) -> ContentResponse:
        if not chunks:
            return ContentResponse(content="No transcript content is available for this video.")

        partials: list[str] = []
        for batch in self._batches(chunks):
            context = "\n\n".join(
                f"[{chunk.start_time:.0f}s] {chunk.text}" for chunk in batch
            )
            partials.append(
                self.llm.complete(
                    [
                        {
                            "role": "system",
                            "content": "Summarize this part of a YouTube transcript faithfully. Capture key ideas, arguments, examples and conclusions. Do not add outside facts.",
                        },
                        {"role": "user", "content": context},
                    ],
                    temperature=0.1,
                    max_tokens=1300,
                    reasoning_effort="low",
                )
            )

        combined = "\n\n".join(partials)
        final = self.llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Create a polished user-facing summary from the supplied partial summaries. "
                        "Use these exact Markdown sections: ## Overview, ## Key ideas, ## Important details, ## Main takeaways. "
                        "Be concise but complete and do not add outside information."
                    ),
                },
                {"role": "user", "content": combined},
            ],
            temperature=0.15,
            max_tokens=2200,
            reasoning_effort="medium",
        )

        representative = [
            RetrievedChunk(**chunk.model_dump())
            for chunk in chunks[:: max(1, len(chunks) // 4)][:4]
        ]
        return ContentResponse(content=final, sources=build_citations(representative, 4))
