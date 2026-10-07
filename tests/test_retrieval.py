from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import EmbeddingModel
from app.ingestion.loader import load_documents
from app.retrieval.retriever import Retriever
from app.retrieval.vectorstore import ChromaVectorStore


def build_test_index(storage_path: Path) -> Retriever:
    documents_path = storage_path.parent / "documents"
    documents_path.mkdir(parents=True, exist_ok=True)

    policy_path = documents_path / "policy.txt"
    policy_path.write_text(
        "Employees must complete eight working hours each day. "
        "Planned leave should be submitted three working days in advance.",
        encoding="utf-8",
    )

    documents = load_documents(documents_path)
    chunks = chunk_documents(
        documents,
        chunk_size=300,
        chunk_overlap=50,
    )

    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.embed_documents(chunks)

    vector_store = ChromaVectorStore(
        storage_path=storage_path,
        collection_name="test_collection",
    )
    vector_store.add_chunks(embedded_chunks)

    return Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )


def test_retriever_returns_relevant_result(tmp_path: Path):
    retriever = build_test_index(tmp_path / "chroma")

    results = retriever.retrieve(
        query="How many working hours are required?",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].source.endswith("policy.txt")
    assert "eight working hours" in results[0].content
    assert results[0].relevance_score >= 0.0


def test_empty_query_raises_error(tmp_path: Path):
    retriever = build_test_index(tmp_path / "chroma")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("")


def test_invalid_top_k_raises_error(tmp_path: Path):
    retriever = build_test_index(tmp_path / "chroma")

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.retrieve(
            query="working hours",
            top_k=0,
        )


def test_empty_vector_store_returns_empty_results(tmp_path: Path):
    embedding_model = EmbeddingModel()

    vector_store = ChromaVectorStore(
        storage_path=tmp_path / "chroma",
        collection_name="empty_collection",
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    assert retriever.retrieve("working hours") == []