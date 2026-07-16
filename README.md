# PDF Q&A Bot (RAG) — Free Version

Upload a PDF → ask questions → get answers grounded in the document, with the
exact source paragraph cited. **No paid API required.**

## What's free here and why

| Step | Tool | Cost | Notes |
|---|---|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Free, runs locally | Downloads once (~90MB), then runs on your CPU — no API calls, no key |
| Vector search | FAISS | Free, open source | Runs locally |
| Answer generation | Groq API (llama-3.1-8b-instant) | Free tier | No credit card required to sign up, generous free rate limits |

## Get your free Groq API key

1. Go to https://console.groq.com
2. Sign up (Google/GitHub login works, no credit card)
3. Go to **API Keys** → **Create API Key**
4. Copy the key — you'll paste it into the app or set it as an env variable

## Local setup

```bash
cd pdf-qa-bot-free
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GROQ_API_KEY=gsk_your_key_here    # Windows: set GROQ_API_KEY=...
streamlit run app.py
```

The first time you upload a PDF, it'll download the local embedding model
(~90MB, one time only). After that, embeddings run instantly on your machine
with no network call.

## Why this still counts as a real RAG project

Nothing about the architecture changed — you still built: PDF parsing,
chunking with overlap, embeddings, a FAISS vector index, similarity search,
and grounded LLM generation with citations. Swapping OpenAI for local
embeddings + Groq is actually a *good* talking point in interviews: it shows
you understand that RAG components are interchangeable, and that you made a
deliberate cost/latency tradeoff rather than being locked into one vendor.

## Architecture

```
PDF file
  │
  ▼
extract_pages()          -- pypdf, per-page text
  │
  ▼
chunk_pages()             -- ~220 words/chunk, 40-word overlap
  │
  ▼
embed_texts()              -- sentence-transformers, LOCAL, free
  │
  ▼
FAISS IndexFlatIP          -- stored in memory per session
  │
  ▼ (on question)
retrieve()                  -- embed question, top-k nearest chunks
  │
  ▼
answer_question()            -- Groq llama-3.1-8b-instant, grounded prompt
  │
  ▼
Streamlit UI                  -- shows answer + expandable source paragraphs
```

## Deployment — Streamlit Community Cloud (free)

1. Push this project to a GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. "New app" → pick repo/branch → main file `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy. Note: the first load after deployment will take a bit longer as
   Streamlit's servers download the ~90MB embedding model — this is normal
   and only happens once per deploy.

## Known limitations (good to mention if asked)

- Scanned/image-only PDFs won't extract text (no OCR built in).
- Groq's free tier has rate limits — fine for a demo/portfolio project, not
  for production traffic.
- `all-MiniLM-L6-v2` is a small, fast embedding model — good enough for this
  scale, but a bigger embedding model would retrieve slightly more precisely
  on nuanced questions.
- Single in-memory FAISS index per session — restarting the app clears it
  (re-upload the PDF to rebuild).
