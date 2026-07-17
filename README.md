# DocuSense — Multi-PDF RAG Assistant

> Converse intelligently with your documents. Upload any set of PDFs and get grounded, source-cited answers powered by a production-grade RAG pipeline.

**[Live Demo →](https://aidocusense.streamlit.app)**

---

## Overview

DocuSense is an end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and interact with their content through a conversational interface. Every response is strictly grounded in the uploaded documents with page-level source attribution — the model never fabricates information outside the provided context.

Built entirely on free-tier services and deployed on Streamlit Community Cloud.

---

## Features

- **Multi-PDF ingestion** — Upload and query across multiple PDFs simultaneously
- **Semantic chunking** — Boundary-aware text splitting with overlap to preserve context
- **FAISS vector retrieval** — Cosine similarity search returning the top-4 most relevant chunks per query
- **Grounded generation** — Prompt-constrained LLM that refuses to answer outside document context
- **Conversational memory** — Sliding-window memory (k=10 turns) with automatic follow-up query reformulation before retrieval
- **Page-level source attribution** — Every answer cites the exact filename and page number of its source chunks
- **Session statistics** — Live tracking of documents loaded, queries made, and conversation turns
- **Keep-alive automation** — GitHub Actions pings the Streamlit app every 6 hours to prevent cold starts

---

## RAG Pipeline Architecture

```
PDF Upload(s)
     │
     ▼
PyPDFLoader  ──────────────────────────────────────────────
     │                                                     │
     ▼                                                     │
RecursiveCharacterTextSplitter                             │
  chunk_size=1000 │ chunk_overlap=100                      │
  separators: ["\n\n", "\n", " ", "\t", ""]                │
     │                                                     │
     ▼                                                     │
HuggingFaceEmbeddings                                      │
  model: all-MiniLM-L6-v2 (CPU)                            │
  normalize_embeddings=True                                │
     │                                                     │
     ▼                                                     │
FAISS Index  ◄─────────────────────────────────────────────
     │
     │
User Query ──► ConversationBufferWindowMemory (k=10)
                        │
                        ▼
              Query Reformulation  ← chat_history
                        │
                        ▼
              FAISS Retriever  (similarity, k=4)
                        │
                        ▼
              PromptTemplate  (context-only constraint)
                        │
                        ▼
              ChatGroq — llama-3.1-8b-instant (temp=0.2)
                        │
                        ▼
              Answer + Source Attribution
              (filename + page number per chunk)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — `llama-3.1-8b-instant` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector Store | FAISS (`faiss-cpu`) |
| RAG Framework | LangChain (`langchain`, `langchain-classic`, `langchain-core`, `langchain-community`, `langchain-groq`, `langchain-huggingface`, `langchain-text-splitters`) |
| PDF Parsing | `pypdf` via `PyPDFLoader` |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |
| Keep-Alive | GitHub Actions (cron every 6 hours) |


---

## Project Structure

```
DocuSense/
├── app.py                          # Streamlit UI — layout, session state, chat rendering
├── docu_sense.py                   # Core RAG pipeline — ingestion, embeddings, chain
├── requirements.txt                # Python dependencies
├── .env                            # Local secrets (never commit)
├── .gitignore
└── .github/
    └── workflows/
        └── keep_active.yml         # Pings the live app every 6 hours
```

---

## Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/DocuSense.git
cd DocuSense
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

**5. Run the app**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run, `sentence-transformers` downloads the embedding model (~90 MB) — subsequent runs use the cached version.

---

## Streamlit Cloud Deployment

1. Push the repository to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo and set `app.py` as the entry point
3. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy. The app reads `st.secrets["GROQ_API_KEY"]` on the cloud and `.env` locally — no code changes needed between environments.

**Keep-alive workflow:** Add your deployed app URL as a GitHub Actions secret named `STREAMLIT_APP_URL`. The `.github/workflows/keep_active.yml` workflow pings it every 6 hours to prevent Streamlit's free-tier from sleeping the app.

---

## Environment Variables

| Variable | Where | Description |
|---|---|---|
| `GROQ_API_KEY` | `.env` (local) or Streamlit Secrets (cloud) | Your Groq API key for LLM inference |
| `STREAMLIT_APP_URL` | GitHub Actions Secret | Your deployed app URL for the keep-alive ping |

---

## How It Works

**Ingestion (`docu_sense.py`)**

PDFs are loaded via `PyPDFLoader` and split with `RecursiveCharacterTextSplitter` at `chunk_size=1000` with `chunk_overlap=100`. The overlap ensures sentences split across boundaries appear in both adjacent chunks, preserving full context. Chunks are embedded locally using `all-MiniLM-L6-v2` with `normalize_embeddings=True` (required for cosine similarity to work correctly with FAISS inner product search) and stored in a FAISS index.

**Retrieval**

On each user query, the retriever performs similarity search returning the top 4 (`k=4`) most semantically relevant chunks. Before retrieval, `ConversationalRetrievalChain` reformulates follow-up questions using the chat history (window of last 10 turns) into standalone queries — preventing retrieval degradation on pronoun-heavy or context-dependent follow-ups.

**Generation**

Retrieved chunks are injected into a `PromptTemplate` that explicitly instructs the model to answer only from the provided context and return a fixed refusal string if the answer is absent. This prevents hallucination. Inference runs on `llama-3.1-8b-instant` via Groq at `temperature=0.2` for deterministic, factual output.

**Source Attribution**

`result["source_documents"]` returns the raw retrieved chunks. `extract_sources()` deduplicates by `(filename, page)` pairs and displays them in an expandable citation panel below each response. Page numbers are stored 0-indexed by PyPDF and incremented by 1 for display.

---


*Built by [Saqib](https://www.linkedin.com/in/saqib-ahmad-siddiqui) · Powered by Groq + LangChain + FAISS + Streamlit*
