from __future__ import annotations

from app.generation.citations import build_citations
from app.llm.groq_client import GroqLLM
from app.schemas.chat import ContentResponse
from app.schemas.retrieval import RetrievedChunk
from app.schemas.video import VideoChunk


class NotesGenerator:
    def __init__(self, llm: GroqLLM) -> None:
        self.llm = llm

    def generate(self, chunks: list[VideoChunk]) -> ContentResponse:
        if not chunks:
            return ContentResponse(content="No transcript content is available for this video.")

        # Keep broad coverage while controlling prompt size for very long videos.
        if len(chunks) > 40:
            step = max(1, len(chunks) // 40)
            selected = chunks[::step][:40]
        else:
            selected = chunks

        context = "\n\n".join(
            f"[Chunk {chunk.chunk_id} | {chunk.start_time:.0f}s] {chunk.text}" for chunk in selected
        )
        notes = self.llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Create useful study notes from a YouTube transcript. Use clear Markdown headings and bullets. "
                        "Focus on concepts, definitions, processes, examples and practical takeaways. "
                        "Keep language simple, avoid filler, and use only the supplied transcript."
                    ),
                },
                {"role": "user", "content": context},
            ],
            temperature=0.1,
            max_tokens=3000,
            reasoning_effort="medium",
        )
        sources = build_citations([RetrievedChunk(**c.model_dump()) for c in selected], 5)
        return ContentResponse(content=notes, sources=sources)
