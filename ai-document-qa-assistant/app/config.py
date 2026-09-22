"""
Central configuration for the AI Document Q&A Assistant.
Reads settings from environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Where raw documents live and where the FAISS index is persisted
DATA_DIR = os.getenv("DATA_DIR", "data/sample_docs")
INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# How many chunks to retrieve per query
TOP_K = int(os.getenv("TOP_K", "4"))
