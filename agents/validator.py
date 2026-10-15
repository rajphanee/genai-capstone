"""
validator.py - Agent 4: check the draft answer before the user sees it (Task 9: anti-hallucination).

Checks (all deterministic - the LLM never grades its own homework):
  1. draft is not empty
  2. draft is not INSUFFICIENT_CONTEXT           -> decline, no retry (the model said it doesn't know)
  3. every sentence carries at least one [S#] tag -> "uncited claim"
  4. every [S#] points to a source that exists    -> "cited a source that was never retrieved"
  5. every NUMBER in the answer appears in the sources it cites -> catches invented figures
  6. sensitive values (SSN, card, API key) are masked in the final text

If checks 1, 3, 4 or 5 fail, the orchestrator sends the problems back to the
Reasoner ONCE as feedback. If the second draft still fails, the user gets a decline.
"""

import re

from src.guardrails import mask_sensitive
from src.llm import INSUFFICIENT

CITE_RE = re.compile(r"\[S(\d+)\]")
# Split after . ! ? or a newline - but NOT when the next thing is a citation tag ("... done. [S1]").
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?!\[S\d)|\n+")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set:
    """All numbers in text, normalized: '1,200.50' -> '1200.5', '20' -> '20'."""
    found = set()
    for raw in NUMBER_RE.findall(text):
        value = raw.replace(",", "").rstrip(".")
        try:
            found.add(str(float(value)).rstrip("0").rstrip("."))
        except ValueError:
            continue
    return found


class ValidatorAgent:
    name = "validator"

    def run(self, draft: str, sources: list) -> dict:
        problems = []
        text = (draft or "").strip()

        if not text:
            return {"verdict": "fail", "problems": ["empty answer"], "answer": "", "citations": []}
        if INSUFFICIENT in text:
            return {"verdict": "insufficient", "problems": ["model reported the sources do not answer the question"],
                    "answer": "", "citations": []}

        # --- 3. every sentence cited
        sentences = [s.strip() for s in SENTENCE_RE.split(text) if s and s.strip()]
        sentences = [s for s in sentences if not CITE_RE.fullmatch(s.strip(" ."))]   # ignore lone tags
        uncited = [s for s in sentences if not CITE_RE.search(s)]
        if uncited:
            problems.append(f"{len(uncited)} sentence(s) have no [S#] citation, e.g. {uncited[0][:80]!r}")

        # --- 4. citations point to real sources
        cited = sorted({int(n) for n in CITE_RE.findall(text)})
        invalid = [n for n in cited if n < 1 or n > len(sources)]
        if invalid:
            problems.append(f"cites source(s) that were not provided: {['S%d' % n for n in invalid]}")

        # --- 5. numbers must exist in the cited sources
        valid_cited = [n for n in cited if 1 <= n <= len(sources)]
        source_numbers = set()
        for n in valid_cited:
            source_numbers |= _numbers(sources[n - 1]["text"])
        answer_numbers = _numbers(CITE_RE.sub("", text))
        invented = sorted(answer_numbers - source_numbers)
        if invented and valid_cited:
            problems.append(f"number(s) not found in the cited sources: {invented}")

        # --- 6. mask sensitive data (never a failure, just a cleanup)
        final, masked = mask_sensitive(text)

        # A readable list of every check and its result - shown in the UI workflow strip.
        checks = [
            ["Every sentence cites a source", not uncited],
            ["Citations point to retrieved chunks", not invalid],
            ["Numbers found in cited sources", not (invented and valid_cited)],
            ["Sensitive data masked" if masked else "No sensitive data in answer", True],
        ]

        citations = [{"tag": f"S{n}", "filename": sources[n - 1]["filename"],
                      "location": sources[n - 1]["location"], "doc_id": sources[n - 1]["doc_id"],
                      "relevance": sources[n - 1]["relevance"]} for n in valid_cited]
        return {
            "verdict": "fail" if problems else "pass",
            "problems": problems,
            "answer": final,
            "citations": citations,
            "masked": masked,
            "checks": checks,
        }
