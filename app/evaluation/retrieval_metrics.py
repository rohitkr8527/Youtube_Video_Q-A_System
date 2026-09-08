from __future__ import annotations

import math


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(item in relevant for item in top) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(item in relevant for item in top) / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
