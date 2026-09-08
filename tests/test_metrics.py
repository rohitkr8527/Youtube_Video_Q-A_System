import pytest

from app.evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics():
    retrieved = [9, 2, 4, 7]
    relevant = {2, 7}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 4) <= 1
