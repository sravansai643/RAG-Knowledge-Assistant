# RAG Knowledge Assistant

A source-grounded Retrieval-Augmented Generation application built with Python, FastAPI, Sentence Transformers, ChromaDB, and the OpenAI Responses API.

The system loads local documents, splits them into chunks, generates embeddings, stores them in a persistent vector database, retrieves relevant context, and generates answers with source references.

## Features

- Text and Markdown document ingestion.
- Configurable document chunking with overlap.
- Local semantic embeddings using Sentence Transformers.
- Persistent ChromaDB vector storage.
- Similarity-based retrieval with relevance scores.
- Grounded answer generation using retrieved context.
- Source metadata returned with generated answers.
- FastAPI REST API.
- Interactive Swagger documentation.
- Automated tests for loading, chunking, retrieval, and API behavior.
- Reproducible dependency management with `uv`.

## Architecture

```text
Documents
   │
   ▼
Document Loader
   │
   ▼
Text Chunker
   │
   ▼
Sentence Transformer Embeddings
   │
   ▼
Persistent ChromaDB
   │
   ▼
Retriever
   │
   ▼
Grounded Prompt
   │
   ▼
OpenAI Responses API
   │
   ▼
Answer with Sources
```

## Project Structure

```text
RAG Knowledge Assistant/
├── app/
│   ├── generation/
│   │   ├── chain.py
│   │   └── prompts.py
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── loader.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── vectorstore.py
│   ├── config.py
│   ├── main.py
│   ├── monitoring.py
│   └── schemas.py
├── data/
│   └── documents/
├── scripts/
│   ├── ingest.py
│   └── query.py
├── storage/
│   └── chroma/
├── tests/
│   ├── test_api.py
│   ├── test_chunker.py
│   ├── test_loader.py
│   └── test_retrieval.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

## Requirements

- Python 3.11 or later.
- `uv`.
- An OpenAI API key for answer generation.
- Internet access for the first embedding-model download.

The embedding model is downloaded and cached locally the first time the application runs.

## Installation

Clone the repository:

```powershell
git clone [https://github.com/YOUR_USERNAME/rag-knowledge-assistant.git](https://github.com/YOUR_USERNAME/rag-knowledge-assistant.git)
cd rag-knowledge-assistant
```

Install the required Python version if necessary:

```powershell
uv python install 3.11
```

Pin the project to Python 3.11:

```powershell
uv python pin 3.11
```

Create or synchronize the project environment:

```powershell
uv sync
```

## Environment Variables

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5-mini
```

Never commit `.env` to GitHub.

The repository includes `.env.example`:

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
```

Copy the example file before adding your key:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and add your real API key.

## Add Documents

Place `.txt` or `.md` files in:

```text
data/documents/
```

Example:

```text
data/documents/company_policy.txt
```

The current loader supports:

- `.txt`
- `.md`

## Build and Test Retrieval

Run the local ingestion and retrieval pipeline:

```powershell
uv run python -m scripts.query
```

The command:

1. Loads documents.
2. Creates chunks.
3. Generates embeddings.
4. Stores embeddings in ChromaDB.
5. Embeds the query.
6. Retrieves the closest chunks.
7. Prints