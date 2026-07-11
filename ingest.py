"""
ingest.py
Chunks regulation text files by section and loads them into a local ChromaDB
vector store using sentence-transformers for embeddings (no API cost, runs locally).

Usage:
    python ingest.py
"""

import os
import re
import glob
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "reg_compliance"

# Matches lines like "SECTION 1005.6 - LIABILITY OF CONSUMER FOR UNAUTHORIZED TRANSFERS"
SECTION_HEADER_PATTERN = re.compile(r"^SECTION\s+([\d.]+)\s*-\s*(.+)$", re.MULTILINE)


def chunk_by_section(text: str, source_file: str):
    """Split regulation text into chunks, one per SECTION header, keeping
    the section number and title attached to each chunk for citation."""
    matches = list(SECTION_HEADER_PATTERN.finditer(text))
    chunks = []

    if not matches:
        # Fallback: no section headers found, treat whole file as one chunk
        chunks.append({
            "text": text.strip(),
            "section": "N/A",
            "title": os.path.basename(source_file),
            "source": source_file,
        })
        return chunks

    for i, match in enumerate(matches):
        section_num = match.group(1)
        section_title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        chunks.append({
            "text": chunk_text,
            "section": section_num,
            "title": section_title,
            "source": source_file,
        })

    return chunks


def main():
    print("Loading embedding model (all-MiniLM-L6-v2, runs locally)...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Fresh start each run so re-ingesting doesn't create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        print(f"No .txt files found in {DATA_DIR}/. Add regulation text files first.")
        return

    all_ids, all_docs, all_metadatas = [], [], []
    chunk_id = 0

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_by_section(text, filepath)
        print(f"{filepath}: {len(chunks)} sections found")

        for chunk in chunks:
            all_ids.append(f"chunk_{chunk_id}")
            all_docs.append(chunk["text"])
            all_metadatas.append({
                "section": chunk["section"],
                "title": chunk["title"],
                "source": os.path.basename(chunk["source"]),
            })
            chunk_id += 1

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas)
    print(f"\nDone. Loaded {chunk_id} chunks into ChromaDB at ./{DB_DIR}")


if __name__ == "__main__":
    main()
