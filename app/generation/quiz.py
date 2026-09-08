from __future__ import annotations

from app.generation.citations import build_citations
from app.llm.groq_client import GroqLLM
from app.schemas.chat import QuizQuestion, QuizResponse
from app.schemas.reasoning import QuizLLMResponse
from app.schemas.retrieval import RetrievedChunk
from app.schemas.video import VideoChunk


class QuizGenerator:
    def __init__(self, llm: GroqLLM) -> None:
        self.llm = llm

    def generate(self, chunks: list[VideoChunk], count: int = 8) -> QuizResponse:
        if not chunks:
            return QuizResponse(questions=[])

        if len(chunks) > 30:
            step = max(1, len(chunks) // 30)
            selected = chunks[::step][:30]
        else:
            selected = chunks

        chunk_map = {chunk.chunk_id: chunk for chunk in selected}
        context = "\n\n".join(
            f"[Chunk {chunk.chunk_id}] {chunk.text}" for chunk in selected
        )
        result = self.llm.structured(
            [
                {
                    "role": "system",
                    "content": (
                        f"Create exactly {count} useful multiple-choice questions from the supplied video transcript. "
                        "Each question must have exactly 4 plausible options, exactly one correct answer, a short explanation, "
                        "and the source_chunk_id that directly supports it. correct_index is zero-based. "
                        "Do not use outside information."
                    ),
                },
                {"role": "user", "content": context},
            ],
            QuizLLMResponse,
            name="video_quiz",
            reasoning_effort="medium",
            max_tokens=3000,
        )

        questions: list[QuizQuestion] = []
        for item in result.questions[:count]:
            if len(item.options) != 4 or not 0 <= item.correct_index < 4:
                continue
            source_chunk = chunk_map.get(item.source_chunk_id)
            source = None
            if source_chunk is not None:
                citations = build_citations([RetrievedChunk(**source_chunk.model_dump())], 1)
                source = citations[0] if citations else None
            questions.append(
                QuizQuestion(
                    question=item.question,
                    options=item.options,
                    correct_index=item.correct_index,
                    explanation=item.explanation,
                    source=source,
                )
            )
        return QuizResponse(questions=questions)
