SYSTEM_PROMPT = """You are a grounded knowledge assistant.

Answer the user's question using only the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- If the context does not contain enough information, say that the provided documents do not contain enough information to answer the question.
- Keep the answer concise and directly address the question.
- Cite supporting context using the exact source labels provided.
- If multiple sources support the answer, cite each relevant source.
"""


def build_context(
    retrieved_chunks: list[dict[str, object]],
) -> str:
    if not retrieved_chunks:
        return "No relevant context was retrieved."

    context_parts: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source = str(chunk["source"])
        content = str(chunk["content"])
        chunk_index = int(chunk["chunk_index"])

        context_parts.append(
            f"[Source {index}]\n"
            f"File: {source}\n"
            f"Chunk: {chunk_index}\n"
            f"Content: {content}"
        )

    return "\n\n".join(context_parts)


def build_user_prompt(
    question: str,
    retrieved_chunks: list[dict[str, object]],
) -> str:
    context = build_context(retrieved_chunks)

    return (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer the question using only the context above. "
        "Include source citations such as [Source 1]."
    )