"""
router_eval.py

Measures how often your LLM-based agent router fails to produce valid JSON,
and how often the rule-based fallback picks the CORRECT route when it does.

This is what backs the CV line:
"Implemented a rule-based fallback router that recovered X% of failed LLM
tool-calls in testing."

Runs TWO test sets:
  1. PRIMARY: the original 24 queries (some deliberately contain words the
     fallback's keyword list also uses, e.g. "latest", "my notes")
  2. ADVERSARIAL: new queries that avoid every fallback keyword on purpose,
     so a high score here means the fallback generalizes rather than just
     pattern-matching the test data it was co-designed with.

SETUP (run these first if you haven't):
    ollama serve                      # if it's not already running in the background
    ollama pull qwen3:4b               # or whichever model you're using - check with `ollama list`
    pip install requests

RUN:
    python router_eval.py

OUTPUT:
    - Console summary with the real percentages
    - router_eval_results.json — full log, including every raw LLM response,
      which you should skim before writing the number down (and can reference
      in your README or an interview if asked "how did you measure that?")
"""

import json
import re
import time
from datetime import datetime
import requests

# ---------- Config ----------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:4b"   # <-- change this to match `ollama list` on your machine
VALID_ROUTES = {"web_search", "doc_search", "direct_answer"}

ROUTER_SYSTEM_PROMPT = """You are the routing component in an agentic RAG system.
Given a user query, decide which single tool should handle it:

- "web_search": the query needs current, real-time, or frequently changing information
- "doc_search": the query is about the user's own uploaded documents/notes
- "direct_answer": the query can be answered from general knowledge, no retrieval needed

Respond with ONLY a JSON object in exactly this format, and nothing else - no
explanation, no markdown fences:
{"route": "web_search" | "doc_search" | "direct_answer"}

/no_think"""

# ---------- Test set ----------
# Edit/expand this with your own queries once you've seen how this run goes.
# Keeping expected labels unambiguous (single clear intent) so "correct" is well-defined.
TEST_QUERIES = [
    # (query, expected_route)
    ("What is the time complexity of binary search?", "direct_answer"),
    ("Explain the difference between a list and a tuple in Python.", "direct_answer"),
    ("What is 17 multiplied by 23?", "direct_answer"),
    ("Summarize the CAP theorem in one paragraph.", "direct_answer"),
    ("What does REST stand for?", "direct_answer"),
    ("Explain backpropagation in neural networks in simple terms.", "direct_answer"),
    ("What is the difference between supervised and unsupervised learning?", "direct_answer"),
    ("Convert 100 Fahrenheit to Celsius.", "direct_answer"),

    ("What's the weather in Kolkata right now?", "web_search"),
    ("What's the latest stable version of PyTorch?", "web_search"),
    ("Who won the most recent India vs Australia cricket match?", "web_search"),
    ("What's the current price of Bitcoin in INR?", "web_search"),
    ("What's the latest news on AI regulation in India?", "web_search"),
    ("When is the next major SpaceX launch scheduled?", "web_search"),
    ("What is the current repo rate set by the RBI?", "web_search"),
    ("What are today's top trending topics on GitHub?", "web_search"),

    ("Summarize chapter 2 of my uploaded notes.", "doc_search"),
    ("What did my research paper say about the evaluation methodology?", "doc_search"),
    ("According to my thesis draft, what are the main contributions?", "doc_search"),
    ("What topics are covered in my machine learning course notes?", "doc_search"),
    ("Find the section in my document about vector databases.", "doc_search"),
    ("What did I write about chunking strategy in my notes?", "doc_search"),
    ("What are the key takeaways from the PDF I added to the knowledge base?", "doc_search"),
    ("Does my document mention anything about persistent memory design?", "doc_search"),
]

# ---------- Adversarial test set ----------
# None of these queries contain any word from fallback_route()'s keyword lists.
# If the fallback still scores well here, that's a real generalization result.
# If it scores much worse than the primary set, that's the honest, more
# defensible number for the CV - and a genuinely interesting finding either way.
ADVERSARIAL_QUERIES = [
    ("Has SpaceX launched Starship again?", "web_search"),
    ("Is Bitcoin trading above $100,000?", "web_search"),
    ("Did India win their series decider against Australia?", "web_search"),
    ("Which version of PyTorch just came out?", "web_search"),
    ("Is the RBI planning to cut interest rates?", "web_search"),
    ("Has the government announced new AI rules in India?", "web_search"),
    ("Which repositories are trending on GitHub?", "web_search"),
    ("Did NASA confirm a launch date for Artemis?", "web_search"),

    ("Can you check what I wrote about vector databases?", "doc_search"),
    ("What have I covered so far in my machine learning coursework?", "doc_search"),
    ("Pull up what I said earlier about chunking strategy.", "doc_search"),
    ("What are the main findings in the file I gave you?", "doc_search"),
    ("Summarize the second section of that research draft I shared.", "doc_search"),
    ("What does the report say about evaluation methodology?", "doc_search"),

    ("Explain how gradient descent works.", "direct_answer"),
    ("What's the difference between L1 and L2 regularization?", "direct_answer"),
    ("How does a hash table achieve constant-time lookup?", "direct_answer"),
]


def call_llm_router(query: str) -> str:
    """Calls the local Ollama model and returns raw text content."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "think": False,
        "stream": False,
        "options": {"num_predict": 400},  # enough room to finish reasoning + emit JSON, even if thinking mode leaks through
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def try_parse_route(raw_text: str):
    """Attempts to extract a valid route from the LLM's raw output.
    Returns (route, success)."""
    cleaned = raw_text.strip()
    # defensively strip any stray <think>...</think> block, just in case
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    # strip markdown fences if the model added them anyway
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    candidates = [cleaned]
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            route = obj.get("route")
            if route in VALID_ROUTES:
                return route, True
        except (json.JSONDecodeError, AttributeError):
            continue
    return None, False


def fallback_route(query: str) -> str:
    """Rule-based backup router used when the LLM's JSON fails to parse."""
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


def preflight_check():
    """Quick sanity call before burning through the whole test set."""
    try:
        call_llm_router("ping")
        return True
    except requests.exceptions.ConnectionError:
        print(f"Could not reach Ollama at {OLLAMA_URL}")
        print("-> Run 'ollama serve' in a terminal, then try again.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"Ollama returned an error: {e}")
        print(f"-> Check that you've pulled the model: ollama pull {MODEL_NAME}")
        print("-> Or check the exact name with: ollama list")
        return False


def run_eval(query_set, label):
    """Runs one labeled batch of (query, expected_route) pairs and returns per-query results."""
    results = []
    print(f"\n--- {label}: {len(query_set)} queries ---\n")

    for i, (query, expected) in enumerate(query_set, 1):
        try:
            raw = call_llm_router(query)
            route, llm_succeeded = try_parse_route(raw)
        except requests.exceptions.RequestException as e:
            # A timeout or connection drop is itself a failed tool call -
            # exactly the case the fallback router exists to handle.
            raw = f"[REQUEST ERROR] {type(e).__name__}: {e}"
            route, llm_succeeded = None, False

        if llm_succeeded:
            final_route = route
            used_fallback = False
        else:
            final_route = fallback_route(query)
            used_fallback = True

        correct = (final_route == expected)

        results.append({
            "query": query,
            "expected": expected,
            "raw_llm_output": raw,
            "llm_parsed_ok": llm_succeeded,
            "used_fallback": used_fallback,
            "final_route": final_route,
            "correct": correct,
        })

        status = "OK" if llm_succeeded else "FALLBACK"
        mark = "correct" if correct else "WRONG"
        print(f"[{i:02d}/{len(query_set)}] {status:8s} -> {final_route:14s} ({mark})  {query}")

        time.sleep(0.1)

    return results


def summarize(results, label):
    """Prints and returns aggregate stats for one batch of results."""
    total = len(results)
    failures = sum(1 for r in results if r["used_fallback"])
    parse_failure_rate = (failures / total * 100) if total else 0.0

    fallback_correct = sum(1 for r in results if r["used_fallback"] and r["correct"])
    recovery_rate = (fallback_correct / failures * 100) if failures else None

    # Accuracy if the system had NO fallback: a parse failure counts as wrong,
    # since the agent would have no valid route to act on at all.
    llm_only_correct = sum(1 for r in results if r["llm_parsed_ok"] and r["correct"])
    llm_only_accuracy = (llm_only_correct / total * 100) if total else 0.0

    overall_correct = sum(1 for r in results if r["correct"])
    overall_accuracy = (overall_correct / total * 100) if total else 0.0

    print(f"\n{'=' * 60}")
    print(f"{label} SUMMARY")
    print(f"Total queries:               {total}")
    print(f"LLM router failures:         {failures} ({parse_failure_rate:.1f}%)  [bad JSON or request errors]")
    if failures:
        print(f"Fallback recovery accuracy:  {fallback_correct}/{failures} ({recovery_rate:.1f}%)")
    else:
        print("Fallback recovery accuracy:  N/A - no failures occurred in this set.")
    print(f"LLM-only accuracy (no fallback): {llm_only_correct}/{total} ({llm_only_accuracy:.1f}%)")
    print(f"Overall accuracy (with fallback): {overall_correct}/{total} ({overall_accuracy:.1f}%)")
    print("=" * 60)

    return {
        "total_queries": total,
        "parse_failures": failures,
        "parse_failure_rate_pct": round(parse_failure_rate, 1),
        "fallback_recovery_rate_pct": round(recovery_rate, 1) if recovery_rate is not None else None,
        "llm_only_accuracy_pct": round(llm_only_accuracy, 1),
        "overall_accuracy_pct": round(overall_accuracy, 1),
    }


def main():
    if not preflight_check():
        return

    print(f"Running router evaluation using model '{MODEL_NAME}'...")

    primary_results = run_eval(TEST_QUERIES, "PRIMARY (keyword-aligned) test set")
    primary_summary = summarize(primary_results, "PRIMARY")

    adversarial_results = run_eval(ADVERSARIAL_QUERIES, "ADVERSARIAL (keyword-free) test set")
    adversarial_summary = summarize(adversarial_results, "ADVERSARIAL")

    combined_results = primary_results + adversarial_results
    combined_summary = summarize(combined_results, "COMBINED (both sets)")

    out = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "primary": {**primary_summary, "results": primary_results},
        "adversarial": {**adversarial_summary, "results": adversarial_results},
        "combined": {**combined_summary, "results": combined_results},
    }
    with open("router_eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nFull results (incl. every raw LLM output, both sets) saved to router_eval_results.json")


if __name__ == "__main__":
    main()
