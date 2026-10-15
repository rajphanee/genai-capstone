"""
retriever.py - Agent 2: fetch the most relevant chunks for the plan (Task 6).

Steps:
  1. run a similarity search for EACH sub-question from the plan
  2. merge results, keep the best score per chunk (no duplicates)
  3. drop chunks below MIN_RELEVANCE      -> weak matches cause hallucinations
  4. sanitize each chunk (guardrails.sanitize_context) -> strips hidden
     "AI, ignore your instructions" lines planted in uploaded files
  5. return the top_k best, best first

If nothing survives step 3, the orchestrator declines instead of guessing.
"""

from src.guardrails import sanitize_context


class RetrieverAgent:
    name = "retriever"

    def __init__(self, store, min_relevance: float):
        self.store = store
        self.min_relevance = min_relevance

    def run(self, plan: dict, doc_ids: list = None) -> dict:
        best = {}                       # chunk_id -> hit with the highest relevance seen
        for query in plan["sub_questions"]:
            for hit in self.store.search(query, k=plan["top_k"], doc_ids=doc_ids):
                current = best.get(hit["chunk_id"])
                if current is None or hit["relevance"] > current["relevance"]:
                    best[hit["chunk_id"]] = hit

        ranked = sorted(best.values(), key=lambda h: h["relevance"], reverse=True)
        kept = [h for h in ranked if h["relevance"] >= self.min_relevance]
        dropped_low = len(ranked) - len(kept)

        removed_lines = 0
        for hit in kept:
            hit["text"], removed = sanitize_context(hit["text"])
            removed_lines += removed
        kept = [h for h in kept if h["text"].strip()][: plan["top_k"]]

        return {
            "sources": kept,
            "candidates": len(ranked),
            "dropped_low_relevance": dropped_low,
            "injection_lines_removed": removed_lines,
            "top_relevance": kept[0]["relevance"] if kept else (ranked[0]["relevance"] if ranked else 0.0),
        }
