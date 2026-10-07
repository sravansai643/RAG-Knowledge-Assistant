import pytest

from app.ingestion.chunker import (
    DocumentChunk,
    chunk_document,
    chunk_documents,
)
from app.ingestion.loader import LoadedDocument


def test_chunk_document_returns_document_chunks():
    document = LoadedDocument(
        source="policy.txt",
        content="A" * 700,
    )

    chunks = chunk_document(
        document=document,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) >= 2
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    assert all(chunk.source == "policy.txt" for chunk in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_chunk_ids_are_stable():
    document = LoadedDocument(
        source="policy.txt",
        content="Flexible working hours are supported.",
    )

    first_result = chunk_document(document)
    second_result = chunk_document(document)

    assert first_result[0].chunk_id == second_result[0].chunk_id


def test_chunk_documents_processes_multiple_documents():
    documents = [
        LoadedDocument(source="one.txt", content="First document."),
        LoadedDocument(source="two.txt", content="Second document."),
    ]

    chunks = chunk_documents(documents)

    assert len(chunks) == 2
    assert chunks[0].source == "one.txt"
    assert chunks[1].source == "two.txt"


def test_empty_document_returns_no_chunks():
    document = LoadedDocument(
        source="empty.txt",
        content="   ",
    )

    assert chunk_document(document) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (100, 100),
        (100, 120),
        (100, -1),
    ],
)
def test_invalid_chunk_parameters_raise_error(chunk_size, chunk_overlap):
    document = LoadedDocument(
        source="policy.txt",
        content="Some content.",
    )

    with pytest.raises(ValueError):
        chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )