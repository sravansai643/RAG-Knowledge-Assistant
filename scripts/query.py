from app.generation.chain import GenerationChain
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import EmbeddingModel
from app.ingestion.loader import load_documents
from app.retrieval.retriever import Retriever
from app.retrieval.vectorstore import ChromaVectorStore


def build_index() -> None:
    documents = load_documents("data/documents")
    chunks = chunk_documents(documents)

    embedding_model = EmbeddingModel()
    embedded_chunks = embedding_model.embed_documents(chunks)

    vector_store = ChromaVectorStore()
    vector_store.add_chunks(embedded_chunks)


def main() -> None:
    build_index()

    retriever = Retriever()

    question = "How many working hours are required each day?"

    results = retriever.retrieve(
        query=question,
        top_k=3,
    )

    chain = GenerationChain()
    generated_answer = chain.generate(
        question=question,
        retrieved_results=results,
    )

    print(f"Answer: {generated_answer.answer}")
    print(f"Sources: {generated_answer.sources}")


if __name__ == "__main__":
    main()