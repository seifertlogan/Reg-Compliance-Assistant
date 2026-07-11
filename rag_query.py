"""
rag_query.py
Core RAG logic: retrieve relevant regulation chunks from ChromaDB, then ask
an LLM to answer using only that retrieved context, citing sections used.

Backend is pluggable via the LLM_BACKEND environment variable:
  - "ollama" (default): free, runs entirely locally, no API key needed.
    Requires Ollama installed (https://ollama.com) and a model pulled,
    e.g. `ollama pull llama3.2`.
  - "anthropic": uses the Claude API for higher-quality answers. Requires
    ANTHROPIC_API_KEY to be set and a funded API account.
"""

import os
import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the project root, if present

DB_DIR = "chroma_db"
COLLECTION_NAME = "reg_compliance"
TOP_K = 4

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are a compliance assistant for a bank operations team. \
Answer the user's question using ONLY the regulation excerpts provided in the context below. \
Do not use outside knowledge. If the context does not contain enough information to answer, \
say so clearly instead of guessing.

Always cite the specific section number(s) you used, like "(Reg E section 1005.11)".

Keep answers concise and practical for someone working in bank operations, not written as legal prose."""


def get_collection():
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def retrieve(question: str, collection, top_k: int = TOP_K):
    results = collection.query(query_texts=[question], n_results=top_k)

    retrieved_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved_chunks.append({
            "text": doc,
            "section": meta.get("section"),
            "title": meta.get("title"),
            "source": meta.get("source"),
            "distance": dist,
        })
    return retrieved_chunks


def build_context_block(chunks):
    blocks = []
    for c in chunks:
        blocks.append(
            f"[Source: {c['source']} | Section {c['section']} - {c['title']}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def _generate_ollama(context: str, question: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Make sure Ollama is installed and running (it usually starts automatically "
            "after installation — try opening the Ollama app, or run `ollama serve`)."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama took too long to respond. The model may still be loading — try again.")

    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def _generate_anthropic(context: str, question: str) -> str:
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before running the app, or switch LLM_BACKEND to 'ollama' for a free local option."
        )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def answer_question(question: str, collection=None):
    """Returns (answer_text, retrieved_chunks) so the caller can display sources."""
    if collection is None:
        collection = get_collection()

    chunks = retrieve(question, collection)
    context = build_context_block(chunks)

    if LLM_BACKEND == "anthropic":
        answer_text = _generate_anthropic(context, question)
    else:
        answer_text = _generate_ollama(context, question)

    return answer_text, chunks


if __name__ == "__main__":
    # Quick CLI smoke test
    q = "How much can a consumer be liable for if they report a lost debit card after 3 days?"
    ans, sources = answer_question(q)
    print("QUESTION:", q)
    print("\nANSWER:\n", ans)
    print("\nSOURCES USED:")
    for s in sources:
        print(f"  - {s['source']} Section {s['section']} ({s['title']})")