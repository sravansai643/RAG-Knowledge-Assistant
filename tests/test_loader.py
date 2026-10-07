from pathlib import Path

import pytest

from app.ingestion.loader import load_documents


def test_load_txt_documents(tmp_path: Path):
    document_path = tmp_path / "policy.txt"
    document_path.write_text("Flexible working hours are supported.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].content == "Flexible working hours are supported."
    assert documents[0].source.endswith("policy.txt")


def test_load_markdown_documents(tmp_path: Path):
    document_path = tmp_path / "policy.md"
    document_path.write_text("# Leave Policy", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].content == "# Leave Policy"


def test_ignore_unsupported_files(tmp_path: Path):
    supported_file = tmp_path / "policy.txt"
    unsupported_file = tmp_path / "image.png"

    supported_file.write_text("Policy text", encoding="utf-8")
    unsupported_file.write_bytes(b"not a supported document")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].source.endswith("policy.txt")


def test_skip_empty_documents(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   ", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert documents == []


def test_missing_directory_raises_error(tmp_path: Path):
    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_directory)