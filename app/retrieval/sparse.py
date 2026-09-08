from __future__ import annotations

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.ingestion.metadata import VideoRepository
from app.schemas.retrieval import RetrievedChunk
from app.schemas.video import VideoChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+.#/-]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class SparseIndex:
    def __init__(self, chunks: list[VideoChunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(
        self,
        query: str,
        limit: int,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            if start_time is not None and chunk.end_time < start_time:
                continue
            if end_time is not None and chunk.start_time > end_time:
                continue
            if float(score) <= 0:
                continue
            results.append(RetrievedChunk(**chunk.model_dump(), sparse_score=float(score)))
            if len(results) >= limit:
                break
        return results


class SparseIndexRegistry:
    def __init__(self) -> None:
        self.repo = VideoRepository()
        self._cache: dict[str, SparseIndex] = {}

    def invalidate(self, video_id: str) -> None:
        self._cache.pop(video_id, None)

    def get(self, video_id: str) -> SparseIndex:
        if video_id not in self._cache:
            self._cache[video_id] = SparseIndex(self.repo.get_chunks(video_id))
        return self._cache[video_id]


@lru_cache(maxsize=1)
def get_sparse_registry() -> SparseIndexRegistry:
    return SparseIndexRegistry()
