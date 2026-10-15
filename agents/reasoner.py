"""
reasoner.py - Agent 3: produce an answer grounded in the retrieved sources (the "G" in RAG).

Retrieved chunks are numbered [S1], [S2]... and sent to the LLM with strict rules:
cite every sentence, use only the sources, reply INSUFFICIENT_CONTEXT otherwise.

`feedback` is filled in on the second attempt, when the Validator rejected the first draft.

Failure handling:
  If Claude fails (bad key, no credit, network down) we fall back to the offline
  extractive answerer, so the user still gets a cited answer copied from the
  documents. The trace records that the fallback was used. If there is no
  fallback (already offline), the error is returned and the user sees a
  friendly "try again" message.
"""

import logging

from src.errors import LLMError
from src.llm import OfflineLLM

log = logging.getLogger(__name__)


class ReasonerAgent:
    name = "reasoner"

    def __init__(self, llm):
        self.llm = llm
        self.fallback = None if llm.name == "offline" else OfflineLLM()

    def run(self, question: str, sources: list, feedback: str = None) -> dict:
        try:
            draft = self.llm.answer(question, sources, feedback=feedback)
            return {"ok": True, "draft": draft.strip(), "error": None, "used_fallback": False}
        except LLMError as exc:
            if self.fallback is None:
                return {"ok": False, "draft": "", "error": str(exc), "used_fallback": False}
            log.warning("LLM failed, using offline extractive answer: %s", exc)
            draft = self.fallback.answer(question, sources)
            return {"ok": True, "draft": draft.strip(), "error": str(exc), "used_fallback": True}
