"""
recall_eval.py

Computes Recall@5 for two retrieval configurations against gold_queries.json:
  BASELINE:  top-5 chunks by embedding similarity alone
  RERANKED:  top-20 by embedding similarity, then reordered by a cross-encoder
             reranker, keeping the new top-5

This is what backs the CV line:
"Benchmarked agentic RAG on N queries, improving Recall@5 from X% to Y% with
semantic reranking."

SETUP:
    pip install chromadb sentence-transformers

BEFORE RUNNING:
    1. Run ingest_docs.py on the demo_docs folder first (or your own documents).
    2. gold_queries.json should be in the same folder as this script.

RUN:
    python recall_eval.py
"""

import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

DB_PATH = "./chroma_store"
COLLECTION_NAME = "docs"
CANDIDATE_POOL_SIZE = 20  # how many candidates to fetch before reranking
K = 5


def main():
    with open("gold_queries.json") as f:
        gold = json.load(f)

    print(f"Loaded {len(gold)} gold queries.")
    if len(gold) < 10:
        print("Warning: fewer than 10 queries - results will be noisy.")

    print("Loading embedding model + cross-encoder reranker (reranker downloads ~80MB on first run)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    baseline_hits = 0
    reranked_hits = 0
    per_query_results = []

    for item in gold:
        query = item["query"]
        relevant = set(item["relevant_chunk_ids"])

        query_emb = embedder.encode(query).tolist()

        # ---- BASELINE: top-K by embedding similarity ----
        baseline_res = collection.query(query_embeddings=[query_emb], n_results=K)
        baseline_ids = baseline_res["ids"][0]
        baseline_hit = bool(relevant & set(baseline_ids))
        baseline_hits += baseline_hit

        # ---- RERANKED: top-N candidates by embedding, then cross-encoder rerank to top-K ----
        pool_res = collection.query(query_embeddings=[query_emb], n_results=CANDIDATE_POOL_SIZE)
        pool_ids = pool_res["ids"][0]
        pool_docs = pool_res["documents"][0]

        pairs = [[query, doc] for doc in pool_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked_top_k = [pool_ids[i] for i in ranked[:K]]
        reranked_hit = bool(relevant & set(reranked_top_k))
        reranked_hits += reranked_hit

        per_query_results.append({
            "query": query,
            "relevant_chunk_ids": list(relevant),
            "baseline_top_k": baseline_ids,
            "baseline_hit": baseline_hit,
            "reranked_top_k": reranked_top_k,
            "reranked_hit": reranked_hit,
        })

        mark_b = "HIT " if baseline_hit else "miss"
        mark_r = "HIT " if reranked_hit else "miss"
        print(f"baseline={mark_b}  reranked={mark_r}   {query}")

    total = len(gold)
    baseline_recall = baseline_hits / total * 100
    reranked_recall = reranked_hits / total * 100

    print("\n" + "=" * 60)
    print(f"Total queries:        {total}")
    print(f"Baseline Recall@{K}:    {baseline_hits}/{total} ({baseline_recall:.1f}%)")
    print(f"Reranked Recall@{K}:    {reranked_hits}/{total} ({reranked_recall:.1f}%)")
    print("=" * 60)

    with open("recall_eval_results.json", "w") as f:
        json.dump({
            "total_queries": total,
            "k": K,
            "baseline_recall_at_k_pct": round(baseline_recall, 1),
            "reranked_recall_at_k_pct": round(reranked_recall, 1),
            "results": per_query_results,
        }, f, indent=2)
    print("\nFull results saved to recall_eval_results.json")


if __name__ == "__main__":
    main()
