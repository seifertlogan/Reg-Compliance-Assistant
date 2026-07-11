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
Claude answer strictly from that retrieved text, so answers are grounded in the
actual regulatory language rather than a model's general (and potentially
outdated or hallucinated) knowledge of banking rules.

## How it works

1. **Ingest** (`ingest.py`) — regulation text files in `data/` are split into
   chunks by section (e.g. "SECTION 1005.11 - PROCEDURES FOR RESOLVING ERRORS"),
   embedded locally using `sentence-transformers`, and stored in a persistent
   ChromaDB vector store.
2. **Retrieve** (`rag_query.py`) — a user's question is embedded and matched
   against the stored chunks to pull the most relevant sections.
3. **Generate** — the retrieved sections are passed to Claude with a system
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
                          Claude API (answer grounded in retrieved text)
                                   ↓
                          app.py (Streamlit UI, shows answer + sources)
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # required for answer generation

python ingest.py     # builds the vector store from data/*.txt
streamlit run app.py # launches the demo UI
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

- **Local embeddings, not an embeddings API** — `sentence-transformers` runs
  the embedding model locally, so ingestion has no API cost and works
  offline. Only answer generation calls the Claude API.
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
  of question/section pairs
- Conversation memory for follow-up questions
