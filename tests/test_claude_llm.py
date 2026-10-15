"""
Claude provider tested WITHOUT network: a fake 'anthropic' module is placed in
sys.modules, so we can check the exact request we would send.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from src.config import Settings
from src.errors import LLMError


class _Errors:
    class AuthenticationError(Exception): pass
    class APIConnectionError(Exception): pass
    class APITimeoutError(Exception): pass
    class RateLimitError(Exception): pass
    class APIStatusError(Exception):
        status_code = 500


def _fake(monkeypatch, reply):
    calls = []

    class Client:
        def __init__(self, **kw):
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kw):
            calls.append(kw)
            if isinstance(reply, Exception):
                raise reply
            return reply

    mod = types.ModuleType("anthropic")
    mod.Anthropic = Client
    for n in ("AuthenticationError", "APIConnectionError", "APITimeoutError", "RateLimitError", "APIStatusError"):
        setattr(mod, n, getattr(_Errors, n))
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return calls


def _llm():
    from src.llm import ClaudeLLM
    return ClaudeLLM(Settings(llm_provider="claude", anthropic_api_key="test", llm_max_retries=0))


def test_planner_uses_make_plan_tool(monkeypatch):
    block = SimpleNamespace(type="tool_use", input={"intent": "lookup", "sub_questions": ["pto days"],
                                                    "needs_retrieval": True, "reason": "x"})
    calls = _fake(monkeypatch, SimpleNamespace(content=[block]))
    plan = _llm().plan("How many PTO days?")
    assert plan["sub_questions"] == ["pto days"]
    assert calls[0]["tool_choice"] == {"type": "auto"}
    assert calls[0]["tools"][0]["name"] == "make_plan"


def test_answer_sends_numbered_sources_and_feedback(monkeypatch):
    calls = _fake(monkeypatch, SimpleNamespace(content=[SimpleNamespace(type="text", text="20 days [S1].")]))
    src = [{"text": "20 days of PTO", "filename": "h.pdf", "location": "page 1"}]
    assert _llm().answer("PTO?", src, feedback="cite every sentence") == "20 days [S1]."
    prompt = calls[0]["messages"][0]["content"]
    assert "[S1] (h.pdf, page 1)" in prompt and "cite every sentence" in prompt


def test_network_failure_becomes_llm_error(monkeypatch):
    _fake(monkeypatch, _Errors.APIConnectionError("down"))
    with pytest.raises(LLMError, match="failed after 1 attempts"):
        _llm().answer("q", [])


def test_unknown_intent_rejected(monkeypatch):
    block = SimpleNamespace(type="tool_use", input={"intent": "hack", "sub_questions": [], "needs_retrieval": True,
                                                    "reason": ""})
    _fake(monkeypatch, SimpleNamespace(content=[block]))
    with pytest.raises(LLMError):
        _llm().plan("x")
