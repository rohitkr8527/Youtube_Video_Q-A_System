from __future__ import annotations

from app.config import get_settings
from app.database.qdrant import get_vector_store
from app.retrieval.embeddings import get_dense_encoder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import get_reranker
from app.retrieval.sparse import get_sparse_registry
from app.schemas.retrieval import RetrievedChunk


class HybridRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.encoder = get_dense_encoder()
        self.vector_store = get_vector_store()
        self.sparse_registry = get_sparse_registry()

    def retrieve(
        self,
        *,
        video_id: str,
        query: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[RetrievedChunk]:
        query_vector = self.encoder.encode_query(query)
        dense = self.vector_store.search(
            video_id,
            query_vector,
            self.settings.dense_candidates,
            start_time,
            end_time,
        )
        sparse = self.sparse_registry.get(video_id).search(
            query,
            self.settings.sparse_candidates,
            start_time,
            end_time,
        )
        fused = reciprocal_rank_fusion(
            [dense, sparse],
            limit=max(self.settings.dense_candidates, self.settings.sparse_candidates),
        )
        return get_reranker().rerank(query, fused, self.settings.rerank_top_k)
