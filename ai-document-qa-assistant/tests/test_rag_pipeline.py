"""
Unit tests for the RAG pipeline's non-LLM logic (chunking, loading) and the
FastAPI contract. LLM/embedding calls are mocked so tests run fast, free,
and without a real OpenAI key -- this is what a CI pipeline needs.
"""
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.schema import Document
from app.rag_pipeline import chunk_documents, load_documents
from app import config


def test_load_documents_reads_sample_docs():
    docs = load_documents(config.DATA_DIR)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)


def test_load_documents_raises_on_empty_dir(tmp_path):
    with_no_files = tmp_path / "empty"
    with_no_files.mkdir()
    try:
        load_documents(str(with_no_files))
        assert False, "expected ValueError for empty directory"
    except ValueError:
        pass


def test_chunk_documents_respects_chunk_size():
    long_text = "word " * 2000
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs)

    assert len(chunks) > 1
    for c in chunks:
        # allow a little slack since the splitter breaks on whole words
        assert len(c.page_content) <= config.CHUNK_SIZE + 50


def test_chunk_documents_preserves_metadata():
    docs = [Document(page_content="short text", metadata={"source": "handbook.txt"})]
    chunks = chunk_documents(docs)
    assert all(c.metadata.get("source") == "handbook.txt" for c in chunks)


@patch("app.rag_pipeline.ChatOpenAI")
@patch("app.rag_pipeline.load_index")
def test_answer_question_returns_expected_shape(mock_load_index, mock_chat_cls):
    """answer_question should return question/answer/sources without hitting a real API."""
    from app.rag_pipeline import answer_question

    fake_doc = Document(page_content="Employees get 18 paid leaves a year.",
                         metadata={"source": "handbook.txt"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [fake_doc]
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever
    mock_load_index.return_value = mock_vectorstore

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="18 paid leaves per year. [Source 1: handbook.txt]")
    mock_chat_cls.return_value = mock_llm

    result = answer_question("How many paid leaves are there?")

    assert result["question"] == "How many paid leaves are there?"
    assert "18 paid leaves" in result["answer"]
    assert result["sources"][0]["source"] == "handbook.txt"
