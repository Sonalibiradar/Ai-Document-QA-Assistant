"""
Run this script whenever you add/update documents in data/sample_docs/.
Usage:
    python -m app.ingest
"""
from app.rag_pipeline import build_index

if __name__ == "__main__":
    build_index()
