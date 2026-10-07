from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.ingestion.embedder import EmbeddedChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    content: str
    chunk_index: int
    distance: float


class ChromaVectorStore:
    def __init__(
        self,
        storage_path: str | Path = "storage/chroma",
        collection_name: str = "rag_documents",
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.storage_path),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        self.collection.upsert(
            ids=[item.chunk.chunk_id for item in embedded_chunks],
            embeddings=[item.embedding for item in embedded_chunks],
            documents=[item.chunk.content for item in embedded_chunks],
            metadatas=[
                {
                    "source": item.chunk.source,
                    "chunk_index": item.chunk.chunk_index,
                }
                for item in embedded_chunks
            ],
        )

    def count(self) -> int:
        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved_chunks: list[RetrievedChunk] = []

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source=str(metadata["source"]),
                    content=document,
                    chunk_index=int(metadata["chunk_index"]),
                    distance=float(distance),
                )
            )

        return retrieved_chunks