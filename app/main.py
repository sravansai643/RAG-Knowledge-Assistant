from fastapi import FastAPI
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


app = FastAPI(
    title="RAG Knowledge Assistant",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {
        "application": "RAG Knowledge Assistant",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return QueryResponse(
        answer=f"Received question: {request.question}",
        sources=[],
    )