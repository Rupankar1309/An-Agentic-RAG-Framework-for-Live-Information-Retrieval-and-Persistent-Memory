"""
fallback_ab_test.py

Compares the OLD keyword-based fallback against a NEW semantic (embedding-based)
fallback, using the exact same LLM failures already captured in
router_eval_results.json from your last run.

This is a controlled A/B test: same failure cases, only the fallback logic
changes. No Ollama calls needed - just embeddings, so it runs in seconds.

SETUP:
    pip install sentence-transformers

RUN (from the same folder as router_eval_results.json):
    python fallback_ab_test.py
"""

import json
from sentence_transformers import SentenceTransformer, util

# ---------- OLD fallback: exact copy of the keyword logic from router_eval.py ----------
def old_fallback_route(query: str) -> str:
    q = query.lower()
    web_keywords = ["today", "latest", "current", "now", "news", "weather",
                     "price", "score", "recent", "this week", "this year", "update"]
    doc_keywords = ["my notes", "my document", "my thesis", "my paper", "uploaded",
                     "chapter", "according to", "knowledge base", "syllabus"]
    if any(k in q for k in web_keywords):
        return "web_search"
    if any(k in q for k in doc_keywords):
        return "doc_search"
    return "direct_answer"


# ---------- NEW fallback: semantic similarity to a handful of example phrases ----------
# Instead of exact keyword matches, this compares the MEANING of the query to
# a few example phrases per route, so paraphrases without the "magic words"
# can still be classified correctly.
ANCHOR_EXAMPLES = {
    "web_search": [
        "What's the weather like right now?",
        "What's the latest news on this?",
        "Who won the game yesterday?",
        "What's the current price of this?",
        "Has this event already happened?",
        "Is there a recent update on this?",
        "Did something new just get released?",
    ],
    "doc_search": [
        "What does my document say about this?",
        "Summarize my notes on this topic.",
        "What did I write about this earlier?",
        "Find the section in my paper about this.",
        "What are the key points in the file I shared?",
        "Check what I said about this before.",
    ],
    "direct_answer": [
        "Explain how this algorithm works.",
        "What is the difference between these two concepts?",
        "Define this term for me.",
        "How do I calculate this?",
        "What is the time complexity of this operation?",
    ],
}

print("Loading embedding model (first run downloads ~80MB, then it's cached locally)...")
_model = SentenceTransformer("all-MiniLM-L6-v2")
_anchor_embeddings = {
    route: _model.encode(examples, convert_to_tensor=True)
    for route, examples in ANCHOR_EXAMPLES.items()
}


def new_fallback_route(query: str) -> str:
    query_emb = _model.encode(query, convert_to_tensor=True)
    best_route, best_score = None, -1.0
    for route, examples_emb in _anchor_embeddings.items():
        score = float(util.cos_sim(query_emb, examples_emb)[0].max())
        if score > best_score:
            best_score = score
            best_route = route
    return best_route


# ---------- Load the REAL failures from your last router_eval.py run ----------
with open("router_eval_results.json") as f:
    data = json.load(f)

all_results = data["primary"]["results"] + data["adversarial"]["results"]
failures = [r for r in all_results if not r["llm_parsed_ok"]]

if not failures:
    print("\nNo LLM failures found in router_eval_results.json - nothing to compare.")
    print("Run router_eval.py again first to generate fresh failure cases.")
else:
    print(f"\nComparing fallback strategies on {len(failures)} real LLM failures from your last run:\n")

    old_correct = 0
    new_correct = 0
    for r in failures:
        query, expected = r["query"], r["expected"]
        old_route = old_fallback_route(query)
        new_route = new_fallback_route(query)
        old_ok = (old_route == expected)
        new_ok = (new_route == expected)
        old_correct += old_ok
        new_correct += new_ok

        print(f"Q: {query}")
        print(f"   expected:      {expected}")
        print(f"   OLD (keyword)  -> {old_route:14s} {'correct' if old_ok else 'WRONG'}")
        print(f"   NEW (semantic) -> {new_route:14s} {'correct' if new_ok else 'WRONG'}")
        print()

    total = len(failures)
    print("=" * 60)
    print(f"OLD keyword fallback recovery:  {old_correct}/{total} ({old_correct/total*100:.1f}%)")
    print(f"NEW semantic fallback recovery: {new_correct}/{total} ({new_correct/total*100:.1f}%)")
    print("=" * 60)
