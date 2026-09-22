"""
Simple chat UI for the AI Document Q&A Assistant.

Run:
    streamlit run streamlit_app.py

Expects the FastAPI backend to be running at API_URL (default localhost:8000).
This file is what you screen-record or screenshot for your resume/portfolio --
a working chat UI is far more convincing to a recruiter than a README.
"""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Document Q&A Assistant", page_icon="📄")
st.title("📄 AI Document Q&A Assistant")
st.caption("Ask questions grounded in your own documents, powered by RAG.")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Setup")
    if st.button("🔄 Rebuild index from data/sample_docs"):
        with st.spinner("Ingesting documents..."):
            resp = requests.post(f"{API_URL}/ingest")
        if resp.ok:
            st.success("Index rebuilt.")
        else:
            st.error(resp.json().get("detail", "Ingest failed."))

    if st.button("🗑️ Clear conversation"):
        if st.session_state.session_id:
            requests.delete(f"{API_URL}/session/{st.session_state.session_id}")
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — {s['excerpt']}...")

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "session_id": st.session_state.session_id},
            )
        if resp.ok:
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.markdown(data["answer"])
            if data.get("sources"):
                with st.expander("Sources"):
                    for s in data["sources"]:
                        st.markdown(f"**{s['source']}** — {s['excerpt']}...")
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "sources": data.get("sources")}
            )
        else:
            error_msg = resp.json().get("detail", "Something went wrong.")
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
