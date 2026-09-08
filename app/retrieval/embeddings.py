from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class DenseEncoder:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = SentenceTransformer(settings.embedding_model)

    @property
    def dimension(self) -> int:
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError("Unable to determine embedding dimension.")
        return int(dim)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector[0]


@lru_cache(maxsize=1)
def get_dense_encoder() -> DenseEncoder:
    return DenseEncoder()
