"""
planner.py - Agent 1: decide HOW to answer before doing any work.

Output (a Plan):
  intent          lookup | summary | comparison | list | chitchat
  sub_questions   1-3 search queries (compound questions are split)
  needs_retrieval False only for greetings - we skip the search entirely
  top_k           how many chunks to fetch; summaries and comparisons need more

Failure handling:
  If the LLM planner fails (network, bad output) we do NOT stop. We fall back
  to a safe default plan: search with the original question. A broken planner
  should cost us a little quality, never the whole answer.
"""

import logging

from src.errors import LLMError

log = logging.getLogger(__name__)

TOP_K_BY_INTENT = {"lookup": 1.0, "list": 1.5, "summary": 2.0, "comparison": 2.0, "chitchat": 0}


class PlannerAgent:
    name = "planner"

    def __init__(self, llm, base_top_k: int):
        self.llm = llm
        self.base_top_k = base_top_k

    def run(self, question: str) -> dict:
        try:
            plan = self.llm.plan(question)
            plan["fallback"] = False
        except LLMError as exc:
            log.warning("planner failed, using default plan: %s", exc)
            plan = {"intent": "lookup", "sub_questions": [question], "needs_retrieval": True,
                    "reason": f"default plan (planner error: {exc})", "fallback": True}

        multiplier = TOP_K_BY_INTENT.get(plan["intent"], 1.0)
        plan["top_k"] = max(1, round(self.base_top_k * multiplier)) if plan["needs_retrieval"] else 0
        return plan
