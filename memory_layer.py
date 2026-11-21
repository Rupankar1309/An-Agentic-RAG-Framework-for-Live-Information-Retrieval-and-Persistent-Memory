"""
memory_layer.py

Persistent semantic memory for the agentic RAG project.

What it does
  - Stores short user facts in a ChromaDB collection ("memory") on disk, so
    they survive program restarts.
  - Recalls the most relevant facts for a new query by cosine similarity.
  - Skips near-duplicate facts instead of storing them twice.
  - Extracts durable user facts from a chat message with the local Qwen3
    model (Ollama), used by the interactive `chat` command.

Commands (run from the folder that holds chroma_store/):
    python memory_layer.py remember "I prefer tea over coffee."
    python memory_layer.py recall "What do I like to drink?"
    python memory_layer.py list
    python memory_layer.py count
    python memory_layer.py reset
    python memory_layer.py chat          # needs Ollama running

Set MEMORY_EMBEDDER=hash to use a tiny offline hashing embedder (only for
testing the plumbing; real use needs sentence-transformers).

SETUP:
    pip install chromadb sentence-transformers requests
"""

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone

import chromadb

DB_PATH = "./chroma_store"
COLLECTION_NAME = "memory"
EMBED_MODEL = "all-MiniLM-L6-v2"
DUPLICATE_SIMILARITY = 0.92   # new fact this close to an old one is skipped
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"       # match `ollama list` on your machine

_embedder = None


def embed(texts):
    """Return a list of embedding vectors, one per input text."""
    global _embedder
    if os.environ.get("MEMORY_EMBEDDER") == "hash":
        return [_hash_embed(t) for t in texts]
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(list(texts)).tolist()


def _hash_embed(text, dim=256):
    import hashlib
    import math
    vec = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class MemoryStore:
    def __init__(self, path=DB_PATH):
        client = chromadb.PersistentClient(path=path)
        # cosine space: distance = 1 - cosine similarity
        self.col = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def count(self):
        return self.col.count()

    def add_fact(self, text, mem_type="fact", source="manual"):
        """Store a fact. Returns 'stored' or 'duplicate'."""
        text = text.strip()
        if not text:
            return "empty"
        vec = embed([text])[0]
        if self.col.count() > 0:
            near = self.col.query(query_embeddings=[vec], n_results=1)
            similarity = 1.0 - near["distances"][0][0]
            if similarity >= DUPLICATE_SIMILARITY:
                return "duplicate"
        self.col.add(
            ids=[uuid.uuid4().hex],
            documents=[text],
            embeddings=[vec],
            metadatas=[{
                "type": mem_type,
                "source": source,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }],
        )
        return "stored"

    def recall(self, query, k=3, min_similarity=0.25):
        """Return up to k (text, similarity) pairs, best first."""
        n = self.col.count()
        if n == 0:
            return []
        vec = embed([query])[0]
        res = self.col.query(query_embeddings=[vec], n_results=min(k, n))
        out = []
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            sim = 1.0 - dist
            if sim >= min_similarity:
                out.append((doc, sim))
        return out

    def all_facts(self):
        res = self.col.get()
        rows = list(zip(res["documents"], res["metadatas"]))
        rows.sort(key=lambda r: r[1].get("created_at", ""))
        return rows

    def reset(self):
        ids = self.col.get()["ids"]
        if ids:
            self.col.delete(ids=ids)


# ---------------------------------------------------------------- Ollama

EXTRACT_PROMPT = """You extract durable facts about the USER from one chat message.
Return ONLY a JSON object, no explanation, in exactly this shape:
{"facts": [{"text": "The user ...", "type": "fact"}]}
Rules:
- Only things the user says about themselves (identity, habits, plans, tastes).
- type is "preference" for likes, dislikes and style preferences, else "fact".
- Skip questions, small talk and one-off requests.
- If nothing durable is stated, return {"facts": []}.
/no_think"""


def _ollama(messages, num_predict):
    import requests
    payload = {
        "model": MODEL_NAME, "messages": messages, "stream": False,
        "think": False, "options": {"num_predict": num_predict},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    text = r.json()["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_facts(user_message):
    """Ask the local model for durable user facts. Returns a list of dicts."""
    try:
        raw = _ollama(
            [{"role": "system", "content": EXTRACT_PROMPT},
             {"role": "user", "content": user_message}],
            num_predict=400,
        )
    except Exception as e:  # network error, timeout, model missing
        print(f"  [extraction skipped: {type(e).__name__}]")
        return []
    raw = re.sub(r"^```(json)?|```$", "", raw.strip()).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        facts = json.loads(raw[start:end + 1]).get("facts", [])
    except json.JSONDecodeError:
        return []
    clean = []
    for f in facts:
        if isinstance(f, dict) and isinstance(f.get("text"), str):
            t = f.get("type", "fact")
            clean.append({"text": f["text"], "type": t if t in ("fact", "preference") else "fact"})
    return clean


def chat_loop(store):
    print("Memory chat. Type 'quit' to exit. Facts you state are saved across sessions.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("quit", "exit", ""):
            break
        memories = store.recall(user, k=3)
        system = "You are a helpful assistant."
        if memories:
            system += "\nKnown facts about the user (from memory):\n" + "\n".join(f"- {m}" for m, _ in memories)
            print("  [recalled: " + "; ".join(f"{m} ({s:.2f})" for m, s in memories) + "]")
        try:
            reply = _ollama([{"role": "system", "content": system},
                             {"role": "user", "content": user}], num_predict=600)
        except Exception as e:
            reply = f"(model unavailable: {type(e).__name__})"
        print(f"Assistant: {reply}\n")
        for f in extract_facts(user):
            status = store.add_fact(f["text"], f["type"], source="chat")
            print(f"  [memory {status}: {f['text']}]")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    store = MemoryStore()
    if cmd == "remember":
        print(store.add_fact(" ".join(sys.argv[2:])))
    elif cmd == "recall":
        for text, sim in store.recall(" ".join(sys.argv[2:]), k=3, min_similarity=0.0):
            print(f"{sim:.3f}  {text}")
    elif cmd == "list":
        for text, meta in store.all_facts():
            print(f"[{meta.get('type')}] {text}")
    elif cmd == "count":
        print(store.count())
    elif cmd == "reset":
        store.reset()
        print("memory cleared")
    elif cmd == "seed":          # seed <facts.json>: JSON list of strings
        with open(sys.argv[2], encoding="utf-8") as f:
            for t in json.load(f):
                store.add_fact(t, source="seed")
        print(store.count())
    elif cmd == "batch-recall":  # batch-recall <queries.json> <out.json>
        with open(sys.argv[2], encoding="utf-8") as f:
            queries = json.load(f)
        out = {q: [t for t, _ in store.recall(q, k=3, min_similarity=0.0)] for q in queries}
        with open(sys.argv[3], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print("done")
    elif cmd == "chat":
        chat_loop(store)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
