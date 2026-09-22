"""
FastAPI backend for the AI Document Q&A Assistant.

Run locally:
    uvicorn app.main:app --reload

Endpoints:
    POST /ingest   -> (re)build the vector index from data/sample_docs
    POST /ask      -> ask a question, get a grounded answer + sources
    GET  /health   -> simple health check
"""
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.rag_pipeline import build_index, answer_question
from app import session_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ai-doc-qa")

app = FastAPI(
    title="AI Document Q&A Assistant",
    description="A Retrieval-Augmented Generation (RAG) API for querying custom documents.",
    version="1.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Basic request logging with latency, useful for debugging and demos."""
    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)")
    return response


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None  # omit to get a fresh, stateless answer


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list
    session_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    try:
        build_index()
        logger.info("Index rebuilt successfully")
        return {"status": "index built successfully"}
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())
    history = session_store.get_history(session_id)

    try:
        result = answer_question(request.question, chat_history=history)
    except FileNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="No index found. Call POST /ingest first to build the index.",
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

    session_store.append_turn(session_id, "user", request.question)
    session_store.append_turn(session_id, "assistant", result["answer"])

    return {**result, "session_id": session_id}


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    session_store.clear_session(session_id)
    return {"status": "session cleared"}
