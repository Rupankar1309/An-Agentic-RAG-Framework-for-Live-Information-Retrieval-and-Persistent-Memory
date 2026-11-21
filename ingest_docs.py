"""
ingest_docs.py

Loads your documents (.pdf, .txt, .md), chunks them, embeds each chunk, and
stores everything in a local persistent ChromaDB collection.

Also writes chunk_index.txt - a human-readable list of every chunk with its
ID and a text preview - so you can read through it and build gold_queries.json
next (which chunk correctly answers which test question).

SETUP:
    pip install chromadb pypdf sentence-transformers

RUN:
    python ingest_docs.py "C:\\path\\to\\your\\documents\\folder"
"""

import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
DB_PATH = "./chroma_store"
COLLECTION_NAME = "docs"


def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in (".txt", ".md"):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return None


def chunk_text(text, size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += (size - overlap)
    return chunks


def main():
    if len(sys.argv) < 2:
        print('Usage: python ingest_docs.py "C:\\path\\to\\your\\documents"')
        return
    folder = sys.argv[1]

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)  # start fresh on re-run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    supported = (".pdf", ".txt", ".md")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(supported))
    if not files:
        print(f"No supported files (.pdf, .txt, .md) found in {folder}")
        return

    chunk_id_counter = 0
    all_ids, all_texts, all_metadatas = [], [], []
    chunk_records = []  # for the human-readable index

    for fname in files:
        path = os.path.join(folder, fname)
        print(f"Processing {fname}...")
        text = extract_text(path)
        if not text or not text.strip():
            print("  Could not extract text, skipping.")
            continue
        for i, chunk in enumerate(chunk_text(text)):
            cid = f"chunk_{chunk_id_counter:04d}"
            chunk_id_counter += 1
            all_ids.append(cid)
            all_texts.append(chunk)
            all_metadatas.append({"source": fname, "chunk_index": i})
            chunk_records.append((cid, fname, chunk[:150].replace("\n", " ")))

    if not all_texts:
        print("No text extracted from any file - nothing to ingest.")
        return

    print(f"\nEmbedding {len(all_texts)} chunks...")
    embeddings = model.encode(all_texts, show_progress_bar=True).tolist()
    collection.add(ids=all_ids, documents=all_texts, embeddings=embeddings, metadatas=all_metadatas)

    with open("chunk_index.txt", "w", encoding="utf-8") as f:
        for cid, fname, preview in chunk_records:
            f.write(f"{cid}\t[{fname}]\t{preview}\n")

    print(f"\nDone. {len(all_texts)} chunks from {len(files)} file(s) stored in {DB_PATH}/")
    print("Chunk index for writing gold queries saved to chunk_index.txt")


if __name__ == "__main__":
    main()
