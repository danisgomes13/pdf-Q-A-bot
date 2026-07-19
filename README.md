# 📄 PDF Q&A Bot

A Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask natural-language questions about it — getting back accurate, grounded answers along with the exact source paragraph and page number they came from.

**Live demo:** _add your deployed Streamlit URL here_

---

## What it does

1. Upload any PDF — a textbook chapter, research paper, class notes, or report
2. Ask a question in plain English
3. Get an answer generated **only** from the document's actual content
4. See the exact source paragraph and page number the answer was based on

No hallucinated answers, no guessing — every response is traceable back to the document itself.

---

## Why RAG instead of just pasting the PDF into a chatbot?

Pasting an entire document into a chatbot works for very short files, but breaks down as documents grow:

- Long documents may not fit in a model's context window at all
- Even when they do, resending the whole document on every question is slow and expensive
- LLMs are known to pay less attention to information buried in the middle of a long context ("lost in the middle")
- Citations from a model reading a huge pasted document are self-reported guesses, not guaranteed

RAG avoids all of this by retrieving only the *relevant* paragraphs first, then handing just those to the model — making answers faster, cheaper, more accurate, and reliably traceable to their source.

---

## How it works

```
PDF upload
   │
   ▼
Text extraction        — pypdf reads text page by page
   │
   ▼
Chunking                — split into ~220-word chunks, 40-word overlap
   │
   ▼
Embedding                — Sentence-Transformers (all-MiniLM-L6-v2), local & free
   │
   ▼
Vector indexing            — FAISS stores chunk embeddings for fast similarity search
   │
   ▼ (on each question)
Retrieval                    — question is embedded, FAISS finds top-k closest chunks
   │
   ▼
Grounded generation            — Groq LLM (Llama 3.1 8B) answers using only retrieved chunks
   │
   ▼
Answer + cited sources           — displayed in the UI, with page numbers
```

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| PDF parsing | [pypdf](https://pypdf.readthedocs.io/) | Extracts text per page |
| Chunking | Custom word-based splitter | Overlapping chunks preserve context across boundaries |
| Embeddings | [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Free, runs locally, no API key |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) | Fast similarity search over embedded chunks |
| LLM | [Groq](https://console.groq.com) (`llama-3.1-8b-instant`) | Free tier, very fast inference |
| UI | [Streamlit](https://streamlit.io) | Full web app in pure Python |
| Config | python-dotenv | Loads API keys from a local `.env` file |
| Hosting | Streamlit Community Cloud | Free deployment straight from GitHub |

An alternate implementation of the same pipeline using **LangChain** is also included, for comparing a hand-rolled approach against a framework-based one.

---

## Project structure

```
pdf-qa-bot/
├── app.py              — Streamlit UI
├── rag.py               — Core RAG pipeline (extraction, chunking, embedding, retrieval, generation)
├── requirements.txt       — Python dependencies
├── .env.example             — Template for required environment variables
├── .gitignore                 — Excludes .env, venv, and cache files from git
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/pdf-qa-bot.git
cd pdf-qa-bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows (cmd)
venv\Scripts\activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (no credit card required)
3. Go to **API Keys** → **Create API Key**
4. Copy the key

### 5. Set up your environment file

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 6. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

> **First run note:** the first time you upload a PDF, the app downloads the local embedding model (~90MB, one-time only). After that, embeddings run instantly and fully offline.

---

## Deployment (Streamlit Community Cloud — free)

1. Push this repo to GitHub (make sure `.env` is **not** committed — it's excluded via `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app** → **Deploy a public app from GitHub**
4. Set:
   - Repository: `yourusername/pdf-qa-bot`
   - Branch: `main`
   - Main file path: `app.py`
5. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
6. Click **Deploy**

First build takes a few minutes (installs `torch`/`sentence-transformers`, downloads the embedding model on first upload). You'll get a public URL like `https://your-app-name.streamlit.app`.

---

## Features

- **Automatic PDF text extraction** — no manual formatting required
- **Overlap-aware chunking** — preserves context across chunk boundaries
- **Semantic vector search** — finds relevant content by meaning, not just keywords
- **Grounded, citation-backed answers** — every answer traceable to an exact paragraph and page
- **Fully free to run and deploy** — no paid API required anywhere in the pipeline

---

## Limitations

- **No OCR** — scanned or image-only PDFs can't be processed, since there's no embedded text layer to extract
- **No relevance threshold** — retrieval always returns its closest matches, even if none are truly relevant to the question
- **No persistence** — the vector index lives only in memory for the current session; restarting the app requires re-uploading the PDF
- **Single document at a time** — no built-in support for querying across multiple PDFs in one session
- **Limited scale** — FAISS's exact search and the LLM's free-tier rate limits aren't built for heavy production traffic

---

## Possible extensions

- Add OCR (`pytesseract`) for scanned PDFs
- Add a similarity-score threshold so unrelated questions are correctly flagged as unanswerable
- Support multiple documents by tagging chunks with a document ID and merging indexes
- Replace in-memory FAISS with a persistent vector database (ChromaDB, Pinecone, Weaviate) so documents survive restarts
- Swap `IndexFlatIP` for an approximate index (IVF/HNSW) to scale to very large documents

---


