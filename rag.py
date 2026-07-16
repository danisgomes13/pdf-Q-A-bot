"""
rag.py — Core RAG pipeline for the PDF Q&A bot. FREE VERSION.

Pipeline:
    PDF -> extract text per page -> chunk with overlap -> embed chunks
        -> FAISS index -> (question) -> embed question -> retrieve top-k
        -> LLM answers using only retrieved chunks -> cite source chunk(s)

This version uses:
  - sentence-transformers (all-MiniLM-L6-v2) for embeddings — runs locally
    on your machine, completely free, no API key, no internet call per
    request (just a one-time model download, ~90MB).
  - Groq's free API for the chat/answer step (llama-3.1-8b-instant) —
    free tier, no credit card required. Get a key at https://console.groq.com

No LangChain — written by hand so you can explain every step in an
interview: what a chunk is, why overlap matters, how similarity search
works, why we force the LLM to only use retrieved context.
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- Models -------------------------------------------------------------
_embedder = None  # lazy-loaded, so app startup is fast until first use

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")  # ~90MB, local, free
    return _embedder


def get_groq_client() -> Groq:
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))


CHAT_MODEL = "llama-3.1-8b-instant"   # free tier on Groq, fast and solid quality
CHUNK_SIZE_WORDS = 220                 # ~ a few paragraphs
CHUNK_OVERLAP_WORDS = 40                # keeps context across chunk boundaries


@dataclass
class Chunk:
    id: int
    page: int
    text: str


@dataclass
class DocIndex:
    chunks: list = field(default_factory=list)     # list[Chunk]
    index: faiss.IndexFlatIP = None                 # FAISS index (cosine via normalized vectors)


# ---------------------------------------------------------------------------
# 1. Extract text per page
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text), 1-indexed pages."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


# ---------------------------------------------------------------------------
# 2. Chunk with overlap (word-based, simple and effective)
# ---------------------------------------------------------------------------
def chunk_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    chunks = []
    cid = 0
    for page_num, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE_WORDS
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(Chunk(id=cid, page=page_num, text=chunk_text))
            cid += 1
            if end >= len(words):
                break
            start = end - CHUNK_OVERLAP_WORDS  # step back for overlap
    return chunks


# ---------------------------------------------------------------------------
# 3. Embed chunks + build FAISS index (LOCAL, free, no API call)
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch-embed a list of strings locally, return L2-normalized float32 array."""
    model = get_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)  # normalize so inner product == cosine similarity
    return vecs


def build_index(pdf_path: str) -> DocIndex:
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError("Could not extract any text from this PDF (is it scanned/image-only?).")

    chunks = chunk_pages(pages)

    # Batch in groups — local embedding is fast, but keep batches reasonable
    BATCH = 200
    all_vecs = []
    for i in range(0, len(chunks), BATCH):
        batch_texts = [c.text for c in chunks[i:i + BATCH]]
        all_vecs.append(embed_texts(batch_texts))
    vectors = np.vstack(all_vecs)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(vectors)

    return DocIndex(chunks=chunks, index=index)


# ---------------------------------------------------------------------------
# 4. Retrieve top-k relevant chunks for a question
# ---------------------------------------------------------------------------
def retrieve(doc_index: DocIndex, question: str, k: int = 4) -> list[Chunk]:
    q_vec = embed_texts([question])
    scores, idxs = doc_index.index.search(q_vec, k)
    return [doc_index.chunks[i] for i in idxs[0] if i != -1]


# ---------------------------------------------------------------------------
# 5. Ask the LLM (Groq, free tier), grounded strictly in retrieved chunks
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a careful assistant that answers questions using ONLY the
provided source excerpts from a document. Rules:
- If the answer isn't in the excerpts, say you don't know based on the document.
- Never use outside knowledge to fill gaps.
- After your answer, cite which excerpt number(s) you used, like: [Source 2].
- Be concise and direct."""


def answer_question(doc_index: DocIndex, question: str, k: int = 4) -> dict:
    top_chunks = retrieve(doc_index, question, k=k)

    context_block = "\n\n".join(
        f"[Source {i+1} | Page {c.page}]\n{c.text}"
        for i, c in enumerate(top_chunks)
    )

    user_prompt = f"""Document excerpts:

{context_block}

Question: {question}

Answer the question using only the excerpts above, and cite source numbers."""

    client = get_groq_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": [
            {"label": f"Source {i+1}", "page": c.page, "text": c.text}
            for i, c in enumerate(top_chunks)
        ],
    }
