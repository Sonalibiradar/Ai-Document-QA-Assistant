"""
API-level tests using FastAPI's TestClient. Verifies request validation and
error handling without needing a live server or a real OpenAI key.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400


@patch("app.main.answer_question")
def test_ask_returns_answer_with_session_id(mock_answer_question):
    mock_answer_question.return_value = {
        "question": "What are the working hours?",
        "answer": "9:30 AM to 6:30 PM, Monday to Friday.",
        "sources": [{"source": "handbook.txt", "excerpt": "Standard working hours..."}],
    }
    resp = client.post("/ask", json={"question": "What are the working hours?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"].startswith("9:30 AM")
    assert "session_id" in body and len(body["session_id"]) > 0


@patch("app.main.answer_question", side_effect=FileNotFoundError)
def test_ask_returns_400_when_index_missing(mock_answer_question):
    resp = client.post("/ask", json={"question": "Anything?"})
    assert resp.status_code == 400
    assert "ingest" in resp.json()["detail"].lower()
