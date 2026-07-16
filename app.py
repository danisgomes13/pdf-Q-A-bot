"""
app.py — Streamlit UI for the PDF Q&A bot. FREE VERSION.

Uses local embeddings (sentence-transformers) + Groq's free API for answers.

Run locally:  streamlit run app.py
Deploy:       push to GitHub, connect repo on share.streamlit.io (see README)
"""

import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag import build_index, answer_question

st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄", layout="centered")

st.title("📄 PDF Q&A Bot")
st.caption("Upload a PDF, ask questions, get answers with cited source paragraphs. "
           "Runs on free tools only — no paid API required.")

# --- API key handling -------------------------------------------------
# Get a free key at https://console.groq.com (no credit card needed).
# Locally: set GROQ_API_KEY as an environment variable.
# On Streamlit Cloud: add it under Settings -> Secrets (see README).
if "GROQ_API_KEY" not in os.environ:
    key_input = st.text_input(
        "Enter your free Groq API key (get one at console.groq.com)",
        type="password",
    )
    if key_input:
        os.environ["GROQ_API_KEY"] = key_input
    else:
        st.info("Enter a Groq API key above to continue, or set GROQ_API_KEY as a secret.")
        st.stop()

# --- Session state ------------------------------------------------------
if "doc_index" not in st.session_state:
    st.session_state.doc_index = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, sources)

# --- Upload + index -------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:
    with st.spinner("Reading and indexing your PDF... (first run also downloads "
                     "the local embedding model, ~90MB, one-time only)"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            st.session_state.doc_index = build_index(tmp_path)
            st.session_state.file_name = uploaded_file.name
            st.session_state.history = []
            st.success(f"Indexed '{uploaded_file.name}' — {len(st.session_state.doc_index.chunks)} chunks.")
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
        finally:
            os.remove(tmp_path)

# --- Q&A -------------------------------------------------------
if st.session_state.doc_index is not None:
    question = st.text_input("Ask a question about the document")
    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            result = answer_question(st.session_state.doc_index, question)
            st.session_state.history.insert(0, (question, result["answer"], result["sources"]))

    for q, a, sources in st.session_state.history:
        st.markdown(f"**Q: {q}**")
        st.markdown(a)
        with st.expander("View source paragraphs"):
            for s in sources:
                st.markdown(f"**{s['label']} (page {s['page']}):**")
                st.markdown(f"> {s['text']}")
        st.divider()
else:
    st.info("Upload a PDF above to get started.")

