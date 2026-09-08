from __future__ import annotations

from app.schemas.retrieval import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    limit: int = 20,
) -> list[RetrievedChunk]:
    scores: dict[int, float] = {}
    merged: dict[int, RetrievedChunk] = {}

    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            existing = merged.get(item.chunk_id)
            if existing is None:
                merged[item.chunk_id] = item.model_copy(deep=True)
            else:
                if item.dense_score is not None:
                    existing.dense_score = item.dense_score
                if item.sparse_score is not None:
                    existing.sparse_score = item.sparse_score

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    output: list[RetrievedChunk] = []
    for chunk_id in ranked_ids[:limit]:
        item = merged[chunk_id]
        item.fusion_score = scores[chunk_id]
        output.append(item)
    return output
