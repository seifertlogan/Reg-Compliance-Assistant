# Reg Compliance Assistant

A retrieval-augmented generation (RAG) tool that answers questions about banking
regulations (Reg E, Reg CC) using only the actual regulation text, with citations
back to the specific section used.

## The problem

Bank operations and support staff regularly need quick, accurate answers to
regulatory questions — "how many days do we have to investigate an error claim?",
"when can we hold funds longer than the standard schedule?" — without digging
through dense CFR text every time. Getting this wrong isn't just an inconvenience;
incorrect handling of Reg E error resolution or Reg CC funds availability creates
real compliance and customer-harm risk.

This tool retrieves the exact regulation sections relevant to a question and has
an LLM answer strictly from that retrieved text, so answers are grounded in the
actual regulatory language rather than a model's general (and potentially
outdated or hallucinated) knowledge of banking rules.

## How it works

1. **Ingest** (`ingest.py`) — regulation text files in `data/` are split into
   chunks by section (e.g. "SECTION 1005.11 - PROCEDURES FOR RESOLVING ERRORS"),
   embedded locally using `sentence-transformers`, and stored in a persistent
   ChromaDB vector store.
2. **Retrieve** (`rag_query.py`) — a user's question is embedded and matched
   against the stored chunks to pull the most relevant sections.
3. **Generate** — the retrieved sections are passed to an LLM with a system
   prompt instructing it to answer *only* from that context and cite the
   section number(s) it used. If the context doesn't cover the question, it
   says so instead of guessing.
4. **Demo UI** (`app.py`) — a Streamlit interface for asking questions and
   inspecting exactly which regulation sections were used to generate each
   answer.

## Architecture

```
data/*.txt  →  ingest.py  →  ChromaDB (local, persistent)
                                   ↓
question  →  rag_query.py (retrieve top-k chunks)
                                   ↓
                LLM backend: Ollama (local, free) or Claude API
                                   ↓
                          app.py (Streamlit UI, shows answer + sources)
```

## Setup

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Create your `.env` file** (copy the example and fill in values):
```bash
cp .env.example .env
```
This is where API keys and backend settings live — no need to re-export
environment variables every terminal session.

**3. Choose a backend:**

*Option A — free, local (default, no API key needed):*
Install [Ollama](https://ollama.com), then pull a small model:
```bash
ollama pull llama3.2
```
Leave `LLM_BACKEND=ollama` in `.env` (or omit it — it's the default).

*Option B — Claude API (better answer quality, small usage cost):*
In `.env`, set:
```
ANTHROPIC_API_KEY=sk-ant-...
LLM_BACKEND=anthropic
```
Requires a funded Anthropic API Console account (separate from a claude.ai
subscription) — typical demo usage costs a few cents.

**4. Build the vector store and launch the app:**
```bash
python ingest.py
streamlit run app.py
```

## Sample data note

The regulation text in `data/` is representative placeholder content written
for demo purposes — not a verified verbatim copy of the current CFR. Before
using this for anything beyond a portfolio demo, replace these files with the
official text from [eCFR](https://www.ecfr.gov) or the
[CFPB](https://www.consumerfinance.gov/rules-policy/regulations/). Because
the chunking is done by `SECTION` headers, any regulation text can be dropped
in as long as sections are marked the same way (`SECTION [number] - [title]`).

## Example questions

- "How much can a consumer be liable for if they don't report a lost debit
  card for 10 days?"
- "How many business days does a bank have to investigate an error claim?"
- "When can a bank hold funds longer than the standard availability schedule?"

## Design decisions worth noting

- **Pluggable LLM backend** — generation defaults to a free, local model via
  Ollama, with a one-line config swap to the Claude API for higher-quality
  answers. This keeps the project runnable at zero cost while still
  demonstrating integration with a production-grade model.
- **Local embeddings, not an embeddings API** — `sentence-transformers` runs
  the embedding model locally, so ingestion has no API cost and works
  offline regardless of which generation backend is selected.
- **Section-aware chunking, not fixed-size splitting** — regulation text has
  natural structural boundaries (sections). Chunking along those boundaries
  keeps each chunk semantically complete and makes citations meaningful,
  instead of splitting mid-thought at an arbitrary character count.
- **Strict grounding in the system prompt** — the assistant is explicitly
  told not to use outside knowledge and to say so when context is
  insufficient, since silent hallucination in a compliance tool is worse
  than an honest "I don't have enough information."

## What I'd add next

- Support for additional regulations (Reg D, Reg DD) — architecture already
  supports this by just dropping more `.txt` files into `data/`
- An evaluation harness that checks retrieval accuracy against a labeled set
  of question/section pairs — an early test surfaced a real retrieval miss
  on a funds-availability question, which is a good example of the kind of
  gap this would catch systematically
- Conversation memory for follow-up questions