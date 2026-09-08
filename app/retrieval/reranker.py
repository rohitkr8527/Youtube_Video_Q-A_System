from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.schemas.retrieval import RetrievedChunk


class Reranker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = CrossEncoder(self.settings.reranker_model)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        if not candidates:
            return []
        top_k = top_k or self.settings.rerank_top_k
        pairs = [(query, item.text) for item in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = []
        for item, score in zip(candidates, scores, strict=True):
            copied = item.model_copy(deep=True)
            copied.rerank_score = float(score)
            ranked.append(copied)
        ranked.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )
        return ranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
