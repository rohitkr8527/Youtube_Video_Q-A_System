from __future__ import annotations

import math
from typing import Iterable, Sequence

from app.schemas.chat import SourceCitation
from app.schemas.retrieval import RetrievedChunk
from evaluation.benchmark_models import TimeRange


def ranges_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def chunk_is_relevant(chunk: RetrievedChunk, ranges: Sequence[TimeRange]) -> bool:
    return any(ranges_overlap(chunk.start_time, chunk.end_time, r.start, r.end) for r in ranges)


def citation_is_relevant(citation: SourceCitation, ranges: Sequence[TimeRange]) -> bool:
    return any(ranges_overlap(citation.start_time, citation.end_time, r.start, r.end) for r in ranges)


def hit_at_k(chunks: Sequence[RetrievedChunk], ranges: Sequence[TimeRange], k: int) -> float:
    if not ranges or k <= 0:
        return 0.0
    return float(any(chunk_is_relevant(chunk, ranges) for chunk in chunks[:k]))


def reciprocal_rank_by_time(chunks: Sequence[RetrievedChunk], ranges: Sequence[TimeRange]) -> float:
    if not ranges:
        return 0.0
    for rank, chunk in enumerate(chunks, start=1):
        if chunk_is_relevant(chunk, ranges):
            return 1.0 / rank
    return 0.0


def citation_hit_counts(citations: Sequence[SourceCitation], ranges: Sequence[TimeRange]) -> tuple[int, int]:
    if not ranges:
        return 0, len(citations)
    correct = sum(citation_is_relevant(citation, ranges) for citation in citations)
    return correct, len(citations)


def percentile(values: Iterable[float], percentile_value: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    p = min(100.0, max(0.0, percentile_value)) / 100.0
    position = (len(vals) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return vals[lower]
    weight = position - lower
    return vals[lower] * (1 - weight) + vals[upper] * weight
