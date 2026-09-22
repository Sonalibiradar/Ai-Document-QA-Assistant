# AI Document Q&A Assistant (RAG Pipeline)

![CI](https://github.com/<your-username>/ai-document-qa-assistant/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A production-style Retrieval-Augmented Generation (RAG) system that answers
natural-language questions over your own documents, with grounded,
source-cited responses and multi-turn conversation support. Built with
**LangChain**, **FAISS**, **OpenAI**, **FastAPI**, and **Streamlit**, fully
containerized and tested in CI.

**[Add a screenshot or short GIF of the Streamlit chat UI here before you push — it's the single highest-impact thing for a recruiter skimming your repo.]**

## Why this project

Most RAG demos are a single notebook. This one is structured the way a small
production service would be: a typed API contract, session-scoped
conversation memory, request logging, automated tests that mock external
calls, a Dockerized deployment, and a CI pipeline that runs on every push.

## Architecture

```
                    ┌─────────────────┐
   documents  ──▶   │  Ingest (chunk +│
   (.txt/.pdf)      │   embed)        │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │  FAISS vector    │
                    │  index (local)   │
                    └────────┬────────┘
                             ▼
question ──▶ retriever (top-k) ──▶ LLM (context + history) ──▶ grounded answer + sources
                                          ▲
                                          │
                              session-scoped chat history
```

- **Ingestion** (`app/ingest.py`, `app/rag_pipeline.py`) — loads `.txt`/`.pdf`
  files, splits them into overlapping chunks, embeds them, and persists a
  FAISS index.
- **Retrieval + generation** (`app/rag_pipeline.py`) — embeds the incoming
  question, retrieves the top-k most similar chunks, and prompts the LLM to
  answer strictly from that context, citing sources.
- **API** (`app/main.py`) — FastAPI service with request logging,
  session-scoped multi-turn memory, and typed request/response models.
- **UI** (`streamlit_app.py`) — a chat interface for demoing the assistant
  live, showing retrieved sources per answer.
- **Tests** (`tests/`) — unit tests for chunking/loading and API contract
  tests, with all LLM/embedding calls mocked so they run free and fast.

## Project structure

```
ai-document-qa-assistant/
├── app/
│   ├── config.py          # settings loaded from .env
│   ├── ingest.py           # CLI script to build the index
│   ├── rag_pipeline.py      # core RAG logic
│   ├── session_store.py     # in-memory multi-turn conversation store
│   └── main.py             # FastAPI app
├── tests/
│   ├── test_rag_pipeline.py
│   └── test_api.py
├── data/sample_docs/         # drop your .txt / .pdf files here
├── streamlit_app.py          # demo chat UI
├── Dockerfile                # API container
├── Dockerfile.ui              # Streamlit container
├── docker-compose.yml
├── .github/workflows/ci.yml   # test + lint on every push
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── .gitignore
```

## Quickstart (local, no Docker)

```bash
git clone https://github.com/Sonalibiradar/ai-document-qa-assistant.git
cd ai-document-qa-assistant

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements-dev.txt

cp .env.example .env          # then paste your OpenAI API key into .env

python -m app.ingest          # build the vector index

uvicorn app.main:app --reload # start the API on :8000
```

In a second terminal, launch the demo UI:
```bash
streamlit run streamlit_app.py
```

Or skip the UI and call the API directly:
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many paid leaves do employees get per year?"}'
```

## Quickstart (Docker)

```bash
cp .env.example .env          # add your key
docker compose up --build
```
- API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

You'll still need to call `POST /ingest` once (via `/docs` or curl) to build
the index inside the container.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

Tests mock all OpenAI calls, so they run without an API key or network
access — this is exactly what runs in `.github/workflows/ci.yml` on every
push and pull request.

## Example response

```json
{
  "question": "How many paid leaves do employees get per year?",
  "answer": "Employees are entitled to 18 paid leaves per calendar year, in addition to 10 public holidays. [Source 1: company_handbook.txt]",
  "sources": [
    {"source": "company_handbook.txt", "excerpt": "Employees are entitled to 18 paid leaves..."}
  ],
  "session_id": "3f1b2c4a-..."
}
```

Pass the returned `session_id` in your next request to ask a natural
follow-up ("what about the public holidays?") that resolves against the
prior turn.

## Adding your own documents

Drop `.txt` or `.pdf` files into `data/sample_docs/`, then rebuild:
```bash
python -m app.ingest
```

## Tech stack

- **LangChain** — document loading, chunking, orchestration
- **FAISS** — local vector similarity search
- **OpenAI API** — embeddings (`text-embedding-3-small`) and chat completion (`gpt-4o-mini`)
- **FastAPI** — REST API with typed request/response models and middleware logging
- **Streamlit** — demo chat UI
- **pytest** — unit + API tests with mocked LLM calls
- **Docker / docker-compose** — containerized API + UI
- **GitHub Actions** — CI running tests and linting on every push

## Known limitations & what I'd do next in production

Being upfront about these is itself a signal of engineering maturity:

- **Session store is in-memory** — history is lost on restart and won't
  work across multiple replicas. In production I'd move this to Redis.
- **FAISS index is local and unsharded** — fine for a demo corpus; at scale
  I'd move to a managed vector DB (Pinecone, Weaviate, or pgvector).
- **No auth** — the API is open. Production would add API-key or OAuth
  middleware and per-user document scoping.
- **No answer-quality evaluation harness yet** — a natural next step is a
  small eval set (question, expected-source pairs) scored with something
  like RAGAS to catch retrieval regressions before they ship.
