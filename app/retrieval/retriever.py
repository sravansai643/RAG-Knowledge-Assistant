from dataclasses import dataclass
from pathlib import Path

from app.ingestion.embedder import EmbeddingModel
from app.retrieval.vectorstore import ChromaVectorStore, RetrievedChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    source: str
    content: str
    chunk_index: int
    distance: float
    relevance_score: float


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vector_store: ChromaVectorStore | None = None,
        storage_path: str | Path = "storage/chroma",
        collection_name: str = "rag_documents",
    ) -> None:
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = vector_store or ChromaVectorStore(
            storage_path=storage_path,
            collection_name=collection_name,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[RetrievalResult]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.vector_store.count() == 0:
            return []

        query_embedding = self.embedding_model.embed_query(normalized_query)

        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return [
            self._to_result(chunk)
            for chunk in retrieved_chunks
        ]

    @staticmethod
    def _to_result(chunk: RetrievedChunk) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            content=chunk.content,
            chunk_index=chunk.chunk_index,
            distance=chunk.distance,
            relevance_score=max(0.0, 1.0 - chunk.distance),
        )