from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.schemas.retrieval import RetrievedChunk
from app.schemas.video import VideoChunk


def collection_name(video_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in video_id)
    return f"video_{safe}"


class DenseVectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = QdrantClient(path=str(settings.qdrant_path))

    def index(self, video_id: str, chunks: list[VideoChunk], vectors, dimension: int) -> None:
        name = collection_name(video_id)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )

        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector.tolist(),
                payload=chunk.model_dump(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=name, points=points, wait=True)

    def search(
        self,
        video_id: str,
        query_vector,
        limit: int,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[RetrievedChunk]:
        name = collection_name(video_id)
        if not self.client.collection_exists(name):
            raise FileNotFoundError(f"No vector index exists for video {video_id}.")

        filters: list[models.FieldCondition] = []
        if start_time is not None:
            filters.append(
                models.FieldCondition(
                    key="end_time",
                    range=models.Range(gte=float(start_time)),
                )
            )
        if end_time is not None:
            filters.append(
                models.FieldCondition(
                    key="start_time",
                    range=models.Range(lte=float(end_time)),
                )
            )
        query_filter = models.Filter(must=filters) if filters else None

        response = self.client.query_points(
            collection_name=name,
            query=query_vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        results: list[RetrievedChunk] = []
        for point in response.points:
            payload = dict(point.payload or {})
            results.append(
                RetrievedChunk(
                    **payload,
                    dense_score=float(point.score),
                )
            )
        return results


@lru_cache(maxsize=1)
def get_vector_store() -> DenseVectorStore:
    return DenseVectorStore()
