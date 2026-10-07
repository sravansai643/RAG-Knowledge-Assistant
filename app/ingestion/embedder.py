from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import DocumentChunk


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    embedding: list[float]


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding.astype(np.float32).tolist(),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

    def embed_query(self, query: str) -> list[float]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            normalized_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype(np.float32).tolist()