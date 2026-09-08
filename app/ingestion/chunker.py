from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from app.config import get_settings
from app.schemas.video import TranscriptSegment, VideoChunk


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]*")
STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "will", "your", "you", "for", "are",
    "was", "were", "they", "their", "but", "not", "can", "what", "when", "where", "which", "into",
    "about", "then", "than", "just", "like", "there", "here", "because", "would", "could", "should",
}


@dataclass(slots=True)
class SemanticUnit:
    text: str
    start: float
    end: float
    approx_tokens: int


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text.split()) * 1.3))


def topic_hint(text: str, top_n: int = 4) -> str:
    words = [w.lower() for w in WORD_RE.findall(text) if len(w) >= 4]
    counts = Counter(w for w in words if w not in STOPWORDS)
    if not counts:
        return "Video section"
    return " · ".join(word for word, _ in counts.most_common(top_n))


class SemanticTimestampChunker:
    """Creates timestamp-preserving chunks and uses embedding similarity dips as topic boundaries."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _make_units(self, segments: list[TranscriptSegment], unit_token_target: int = 110) -> list[SemanticUnit]:
        units: list[SemanticUnit] = []
        texts: list[str] = []
        start: float | None = None
        end = 0.0
        tokens = 0

        for segment in segments:
            seg_tokens = approx_tokens(segment.text)
            if start is None:
                start = segment.start
            texts.append(segment.text)
            tokens += seg_tokens
            end = segment.start + segment.duration

            sentence_end = segment.text.rstrip().endswith((".", "?", "!"))
            pause_boundary = False
            if len(units) == 0 and False:
                pause_boundary = True
            if tokens >= unit_token_target and (sentence_end or tokens >= int(unit_token_target * 1.35)):
                text = " ".join(texts).strip()
                units.append(SemanticUnit(text, start, end, approx_tokens(text)))
                texts, start, tokens = [], None, 0

        if texts and start is not None:
            text = " ".join(texts).strip()
            units.append(SemanticUnit(text, start, end, approx_tokens(text)))
        return units

    @staticmethod
    def _cosine_adjacent(vectors: np.ndarray) -> list[float]:
        if len(vectors) < 2:
            return []
        # Vectors are normalized by the encoder, but normalize defensively.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.clip(norms, 1e-12, None)
        return [float(np.dot(normalized[i], normalized[i + 1])) for i in range(len(normalized) - 1)]

    def chunk(
        self,
        *,
        video_id: str,
        title: str,
        segments: list[TranscriptSegment],
        encode_texts,
    ) -> list[VideoChunk]:
        units = self._make_units(segments)
        if not units:
            return []

        vectors = encode_texts([unit.text for unit in units])
        similarities = self._cosine_adjacent(vectors)
        threshold = (
            float(np.quantile(similarities, self.settings.semantic_boundary_quantile))
            if similarities
            else -1.0
        )

        boundaries = {i + 1 for i, score in enumerate(similarities) if score <= threshold}
        chunks: list[VideoChunk] = []
        current: list[SemanticUnit] = []
        current_tokens = 0

        def flush() -> None:
            nonlocal current, current_tokens
            if not current:
                return
            text = " ".join(unit.text for unit in current).strip()
            chunks.append(
                VideoChunk(
                    video_id=video_id,
                    title=title,
                    chunk_id=len(chunks),
                    start_time=current[0].start,
                    end_time=current[-1].end,
                    topic=topic_hint(text),
                    text=text,
                )
            )

            overlap_budget = self.settings.overlap_tokens
            overlap: list[SemanticUnit] = []
            overlap_tokens = 0
            for unit in reversed(current):
                if overlap and overlap_tokens + unit.approx_tokens > overlap_budget:
                    break
                overlap.insert(0, unit)
                overlap_tokens += unit.approx_tokens
            current = overlap
            current_tokens = overlap_tokens

        for idx, unit in enumerate(units):
            would_overflow = current_tokens + unit.approx_tokens > self.settings.max_chunk_tokens
            semantic_boundary = idx in boundaries and current_tokens >= self.settings.min_chunk_tokens

            if current and (would_overflow or semantic_boundary):
                flush()

            current.append(unit)
            current_tokens += unit.approx_tokens

            if current_tokens >= self.settings.target_chunk_tokens and idx + 1 in boundaries:
                flush()

        if current:
            # Avoid emitting a duplicate consisting only of overlap from the last flush.
            text = " ".join(unit.text for unit in current).strip()
            if not chunks or text != chunks[-1].text:
                chunks.append(
                    VideoChunk(
                        video_id=video_id,
                        title=title,
                        chunk_id=len(chunks),
                        start_time=current[0].start,
                        end_time=current[-1].end,
                        topic=topic_hint(text),
                        text=text,
                    )
                )

        return chunks
