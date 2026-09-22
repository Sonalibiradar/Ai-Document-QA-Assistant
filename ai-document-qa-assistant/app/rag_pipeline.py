"""
RAG pipeline: loads documents, chunks + embeds them, stores/retrieves
vectors with FAISS, and generates grounded answers with citations.
"""
import os
import glob
from typing import List, Dict

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document

from app import config


def load_documents(data_dir: str = config.DATA_DIR) -> List[Document]:
    """Load all .txt and .pdf files from a directory into LangChain Documents."""
    docs: List[Document] = []
    for path in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True):
        if path.lower().endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif path.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
    if not docs:
        raise ValueError(f"No .txt or .pdf documents found in '{data_dir}'.")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_index(data_dir: str = config.DATA_DIR, index_dir: str = config.INDEX_DIR) -> None:
    """Build a FAISS vector index from the documents in data_dir and persist it."""
    docs = load_documents(data_dir)
    chunks = chunk_documents(docs)

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents into '{index_dir}'.")


def load_index(index_dir: str = config.INDEX_DIR) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def answer_question(
    question: str,
    index_dir: str = config.INDEX_DIR,
    chat_history: List[Dict[str, str]] = None,
) -> Dict:
    """
    Retrieve relevant chunks for `question` and generate a grounded answer.

    `chat_history` is an optional list of prior {"role", "content"} turns,
    enabling multi-turn conversations where follow-up questions ("what about
    remote employees?") are understood in context of what was asked before.

    Returns a dict with the answer text and the source snippets used.
    """
    vectorstore = load_index(index_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source {i+1}: {os.path.basename(d.metadata.get('source', 'unknown'))}]\n{d.page_content}"
        for i, d in enumerate(relevant_docs)
    )

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context. If the answer isn't in the context, say you don't "
        "know instead of guessing. Cite sources using their [Source N] labels. "
        "Use the prior conversation only to resolve references (e.g. 'it', "
        "'that policy') -- never answer from memory outside the context."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-6:])  # keep last few turns to bound token usage
    messages.append({"role": "user", "content": user_prompt})

    llm = ChatOpenAI(model=config.CHAT_MODEL, api_key=config.OPENAI_API_KEY, temperature=0)
    response = llm.invoke(messages)

    return {
        "question": question,
        "answer": response.content,
        "sources": [
            {
                "source": os.path.basename(d.metadata.get("source", "unknown")),
                "excerpt": d.page_content[:220],
            }
            for d in relevant_docs
        ],
    }
