"""
graph.py - wires the four agents together with LangGraph (our ONE agentic framework).

LangGraph models a workflow as a graph:
  - a shared STATE dict flows through the graph
  - each NODE is a function that reads the state and returns the fields it changes
  - EDGES say which node runs next; CONDITIONAL edges choose based on the state

Our graph:

    START
      │
   guardrail ── blocked ──────────────────────────────► END   (status = blocked)
      │
   planner ──── greeting ─────────────────────────────► END   (status = chitchat)
      │
   retriever ── nothing relevant ─────────────────────► END   (status = declined)
      │
   reasoner ─── LLM error ────────────────────────────► END   (status = error)
      │
   validator ── pass ─────────────────────────────────► END   (status = answered)
      │  └───── fail, first time ──► reasoner (with the problems as feedback)
      └──────── fail twice / "not in sources" ────────► END   (status = declined)

Every node appends one entry to state["trace"]; the UI shows it under
"How the agents answered", so the agentic workflow is visible to the user.

The agents themselves (planner.py, retriever.py, reasoner.py, validator.py) are
plain Python classes; this file is only the "wiring".
"""

import operator
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.planner import PlannerAgent
from agents.reasoner import ReasonerAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from src.errors import GuardrailViolation
from src.guardrails import check_question, find_injection

DECLINE_MESSAGE = "I couldn't find a reliable answer to that in the uploaded documents."
CHITCHAT_MESSAGE = ("Hello! Upload documents and ask me a question about them - I answer only from your files "
                    "and cite where each fact came from.")
BLOCKED_MESSAGE = "I can't help with that request."
LLM_DOWN_MESSAGE = "The language model is unavailable right now. Please try again in a minute."
MAX_REASONING_ATTEMPTS = 2


class QAState(TypedDict, total=False):
    """Everything the agents share while answering ONE question."""
    question: str
    doc_ids: Optional[list]
    plan: dict
    sources: list
    draft: str
    feedback: Optional[str]
    attempts: int
    # final outputs
    status: str
    answer: str
    citations: list
    reason: str
    # `operator.add` tells LangGraph to APPEND each node's trace entries instead of replacing them
    trace: Annotated[list, operator.add]


def _step(agent: str, summary: str, details: dict, started: float) -> list:
    return [{"agent": agent, "summary": summary, "ms": round((time.perf_counter() - started) * 1000, 1),
             "details": details}]


class AgentGraph:
    def __init__(self, llm, store, settings):
        self.settings = settings
        self.planner = PlannerAgent(llm, settings.top_k)
        self.retriever = RetrieverAgent(store, settings.min_relevance)
        self.reasoner = ReasonerAgent(llm)
        self.validator = ValidatorAgent()
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _guardrail(self, state: QAState) -> dict:
        t0 = time.perf_counter()
        try:
            question = check_question(state["question"], self.settings.max_question_chars)
        except GuardrailViolation as exc:
            matched = find_injection(state["question"])
            checks = [[f"Length {len(state['question'])} (3-{self.settings.max_question_chars})", True],
                      ["Prompt-injection patterns", False]]
            return {"status": "blocked", "answer": BLOCKED_MESSAGE, "reason": str(exc),
                    "trace": _step("guardrail", "question blocked",
                                   {"reason": str(exc), "matched": matched, "checks": checks}, t0)}
        checks = [[f"Length {len(question)} (3-{self.settings.max_question_chars})", True],
                  ["Prompt-injection patterns", True]]
        return {"question": question,
                "trace": _step("guardrail", "question passed input checks",
                               {"length": len(question), "checks": checks}, t0)}

    def _plan(self, state: QAState) -> dict:
        t0 = time.perf_counter()
        plan = self.planner.run(state["question"])
        update = {"plan": plan,
                  "trace": _step("planner", f"intent={plan['intent']}, {len(plan['sub_questions'])} search query(ies)",
                                 plan, t0)}
        if not plan["needs_retrieval"]:
            update.update(status="chitchat", answer=CHITCHAT_MESSAGE, reason="no retrieval needed")
        return update

    def _retrieve(self, state: QAState) -> dict:
        t0 = time.perf_counter()
        result = self.retriever.run(state["plan"], doc_ids=state.get("doc_ids"))
        sources = result["sources"]
        details = {k: v for k, v in result.items() if k != "sources"}
        details["min_relevance"] = self.settings.min_relevance
        details["sources"] = [{"filename": s["filename"], "location": s["location"], "relevance": s["relevance"]}
                              for s in sources]
        update = {"sources": sources,
                  "trace": _step("retriever", f"{len(sources)} relevant chunk(s) kept of {result['candidates']} found",
                                 details, t0)}
        if not sources:
            update.update(status="declined", answer=DECLINE_MESSAGE,
                          reason="no document content was relevant enough")
        return update

    def _reason(self, state: QAState) -> dict:
        t0 = time.perf_counter()
        attempt = state.get("attempts", 0) + 1
        result = self.reasoner.run(state["question"], state["sources"], feedback=state.get("feedback"))
        if not result["ok"]:
            summary = "LLM error"
        elif result["used_fallback"]:
            summary = "LLM error - used offline extractive answer instead"
        else:
            summary = "draft ready"
        update = {"attempts": attempt, "draft": result["draft"],
                  "trace": _step("reasoner", f"attempt {attempt}: {summary}",
                                 {"draft": result["draft"], "error": result["error"],
                                  "used_fallback": result["used_fallback"],
                                  "model": ("offline (fallback)" if result["used_fallback"] else
                                            self.settings.claude_model if self.reasoner.llm.name == "claude"
                                            else "offline extractive"),
                                  "sources_sent": len(state["sources"]),
                                  "rewrite": bool(state.get("feedback"))}, t0)}
        if not result["ok"]:
            update.update(status="error", answer=LLM_DOWN_MESSAGE, reason=result["error"])
        return update

    def _validate(self, state: QAState) -> dict:
        t0 = time.perf_counter()
        check = self.validator.run(state["draft"], state["sources"])
        update = {"trace": _step("validator", f"attempt {state['attempts']}: {check['verdict']}",
                                 {"problems": check["problems"], "checks": check.get("checks", []),
                                  "cited": [c["tag"] for c in check.get("citations", [])]}, t0)}
        if check["verdict"] == "pass":
            update.update(status="answered", answer=check["answer"], citations=check["citations"],
                          reason="all checks passed")
        elif check["verdict"] == "insufficient":
            update.update(status="declined", answer=DECLINE_MESSAGE, reason="sources do not contain the answer")
        elif state["attempts"] >= MAX_REASONING_ATTEMPTS:
            update.update(status="declined", answer=DECLINE_MESSAGE,
                          reason="answer failed validation twice: " + "; ".join(check["problems"]))
        else:
            update["feedback"] = "; ".join(check["problems"])     # loop back to the reasoner
        return update

    # ------------------------------------------------------------------ routing
    @staticmethod
    def _finished_or(next_node: str):
        """Conditional edge: stop if a node already set a final status, else continue."""
        def route(state: QAState) -> str:
            return END if state.get("status") else next_node
        return route

    def _build(self):
        g = StateGraph(QAState)
        g.add_node("guardrail", self._guardrail)
        g.add_node("planner", self._plan)
        g.add_node("retriever", self._retrieve)
        g.add_node("reasoner", self._reason)
        g.add_node("validator", self._validate)

        g.add_edge(START, "guardrail")
        g.add_conditional_edges("guardrail", self._finished_or("planner"))
        g.add_conditional_edges("planner", self._finished_or("retriever"))
        g.add_conditional_edges("retriever", self._finished_or("reasoner"))
        g.add_conditional_edges("reasoner", self._finished_or("validator"))
        g.add_conditional_edges("validator", self._finished_or("reasoner"))   # retry loop
        return g.compile()

    # ------------------------------------------------------------------ public
    def run(self, question: str, doc_ids: list = None) -> dict:
        started = time.perf_counter()
        final = self.graph.invoke({"question": question, "doc_ids": doc_ids, "trace": [], "attempts": 0})
        return {
            "status": final.get("status", "error"),
            "answer": final.get("answer", LLM_DOWN_MESSAGE),
            "citations": final.get("citations", []),
            "reason": final.get("reason", ""),
            "trace": final.get("trace", []),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
        }
