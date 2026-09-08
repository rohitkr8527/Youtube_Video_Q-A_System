from __future__ import annotations

import argparse
from statistics import mean

from app.config import get_settings
from app.database.qdrant import get_vector_store
from app.evaluation.dataset import load_dataset
from app.evaluation.retrieval_metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from app.retrieval.embeddings import get_dense_encoder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import get_reranker
from app.retrieval.sparse import get_sparse_registry


def _score(ids: list[int], relevant: set[int], k: int) -> dict[str, float]:
    return {
        "precision": precision_at_k(ids, relevant, k),
        "recall": recall_at_k(ids, relevant, k),
        "mrr": reciprocal_rank(ids, relevant),
        "ndcg": ndcg_at_k(ids, relevant, k),
    }


def run(dataset_path: str, k: int = 6) -> dict[str, dict[str, float]]:
    settings = get_settings()
    encoder = get_dense_encoder()
    vector_store = get_vector_store()
    sparse_registry = get_sparse_registry()
    reranker = get_reranker()
    cases = load_dataset(dataset_path)
    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    stage_rows: dict[str, list[dict[str, float]]] = {
        "dense": [],
        "sparse": [],
        "hybrid": [],
        "hybrid_reranked": [],
    }

    for case in cases:
        query_vector = encoder.encode_query(case.question)
        dense = vector_store.search(case.video_id, query_vector, settings.dense_candidates)
        sparse = sparse_registry.get(case.video_id).search(case.question, settings.sparse_candidates)
        hybrid = reciprocal_rank_fusion(
            [dense, sparse],
            limit=max(settings.dense_candidates, settings.sparse_candidates),
        )
        reranked = reranker.rerank(case.question, hybrid, settings.rerank_top_k)

        relevant = set(case.relevant_chunk_ids)
        for stage, results in {
            "dense": dense,
            "sparse": sparse,
            "hybrid": hybrid,
            "hybrid_reranked": reranked,
        }.items():
            ids = [item.chunk_id for item in results]
            stage_rows[stage].append(_score(ids, relevant, k))

    output: dict[str, dict[str, float]] = {}
    for stage, rows in stage_rows.items():
        output[stage] = {
            metric: mean(row[metric] for row in rows)
            for metric in ("precision", "recall", "mrr", "ndcg")
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare VideoRAG retrieval stages.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    metrics = run(args.dataset, args.k)
    print(f"{'STAGE':20s} {'P@K':>8s} {'R@K':>8s} {'MRR':>8s} {'NDCG':>8s}")
    for stage, values in metrics.items():
        print(
            f"{stage:20s} {values['precision']:8.4f} {values['recall']:8.4f} "
            f"{values['mrr']:8.4f} {values['ndcg']:8.4f}"
        )


if __name__ == "__main__":
    main()
