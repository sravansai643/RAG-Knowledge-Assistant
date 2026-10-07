from dataclasses import dataclass
from os import getenv

from dotenv import load_dotenv
from openai import OpenAI

from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from app.retrieval.retriever import RetrievalResult

load_dotenv()


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sources: list[str]


class GenerationChain:
    def __init__(
        self,
        client: OpenAI | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name or getenv(
            "OPENAI_MODEL",
            "gpt-5-mini",
        )

        api_key = getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to the .env file."
            )

        self.client = client or OpenAI(api_key=api_key)

    def generate(
        self,
        question: str,
        retrieved_results: list[RetrievalResult],
    ) -> GeneratedAnswer:
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("Question cannot be empty")

        if not retrieved_results:
            return GeneratedAnswer(
                answer=(
                    "The provided documents do not contain enough "
                    "information to answer this question."
                ),
                sources=[],
            )

        retrieved_chunks = [
            {
                "source": result.source,
                "content": result.content,
                "chunk_index": result.chunk_index,
            }
            for result in retrieved_results
        ]

        response = self.client.responses.create(
            model=self.model_name,
            instructions=SYSTEM_PROMPT,
            input=build_user_prompt(
                question=normalized_question,
                retrieved_chunks=retrieved_chunks,
            ),
        )

        return GeneratedAnswer(
            answer=response.output_text.strip(),
            sources=sorted(
                {
                    result.source
                    for result in retrieved_results
                }
            ),
        )