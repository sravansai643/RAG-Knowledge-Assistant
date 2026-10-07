from dataclasses import dataclass
from hashlib import sha256

from app.ingestion.loader import LoadedDocument


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    source: str
    content: str
    chunk_index: int


def _validate_chunk_parameters(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _create_chunk_id(source: str, chunk_index: int, content: str) -> str:
    value = f"{source}:{chunk_index}:{content}".encode("utf-8")
    return sha256(value).hexdigest()[:16]


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    normalized_text = " ".join(text.split())

    if not normalized_text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized_text)
    step = chunk_size - chunk_overlap

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = normalized_text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end

        while start < text_length and normalized_text[start].isspace():
            start += 1

        if start == end:
            start += step

    return chunks


def chunk_document(
    document: LoadedDocument,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    _validate_chunk_parameters(chunk_size, chunk_overlap)

    text_chunks = _split_text(
        text=document.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return [
        DocumentChunk(
            chunk_id=_create_chunk_id(
                source=document.source,
                chunk_index=index,
                content=content,
            ),
            source=document.source,
            content=content,
            chunk_index=index,
        )
        for index, content in enumerate(text_chunks)
    ]


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []

    for document in documents:
        chunks.extend(
            chunk_document(
                document=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks