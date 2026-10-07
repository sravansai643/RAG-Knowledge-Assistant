from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True)
class LoadedDocument:
    source: str
    content: str


def load_documents(directory: str | Path) -> list[LoadedDocument]:
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(
            f"Document directory does not exist: {directory_path}"
        )

    if not directory_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but received: {directory_path}"
        )

    documents: list[LoadedDocument] = []

    for file_path in sorted(directory_path.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = file_path.read_text(encoding="utf-8").strip()

        if not content:
            continue

        documents.append(
            LoadedDocument(
                source=file_path.as_posix(),
                content=content,
            )
        )

    return documents