"""
flow_view.py - turns one answer's agent trace into a coloured workflow with a snapshot per step.

    ① Guardrail   ② Planner   ③ Retriever   ④ Reasoner   ⑤ Validator   🏁 Outcome

Each card is coloured from what actually happened for THIS question:

    green  = step passed
    amber  = step passed after a retry, used a fallback, or cleaned something up
    red    = step blocked / failed (the flow stopped here)
    grey   = step skipped (an earlier step ended the flow)

...and shows a short snapshot of WHY: which rule fired, which files/pages were hit and
how relevant they were, which model wrote the answer, which checks passed or failed.

Only the `trace` returned by the LangGraph workflow (agents/graph.py) is read here.
"""

import html
from pathlib import Path

GREEN, AMBER, RED, GREY, BLUE = "#2E7D32", "#B26A00", "#C62828", "#8D8D8D", "#1565C0"
STEPS = [
    ("guardrail", "🛡️", "Guardrail"),
    ("planner", "🧭", "Planner"),
    ("retriever", "🔎", "Retriever"),
    ("reasoner", "✍️", "Reasoner"),
    ("validator", "✅", "Validator"),
]
OUTCOME = {
    "answered": (GREEN, "Answered"),
    "declined": (AMBER, "Declined"),
    "blocked": (RED, "Blocked"),
    "chitchat": (BLUE, "Greeting"),
    "error": (RED, "Error"),
}
FILE_ICONS = {".pdf": "📕", ".txt": "📄", ".md": "📄", ".csv": "📊", ".xlsx": "📗", ".xls": "📗"}
OK, FAIL, INFO, WARN, RETRY = "✓", "✗", "•", "⚠", "↻"


def _short(text, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def _file(name: str) -> str:
    return f"{FILE_ICONS.get(Path(name).suffix.lower(), '📁')} {_short(name, 26)}"


def _steps_by_agent(trace: list) -> dict:
    grouped = {}
    for step in trace:
        grouped.setdefault(step["agent"], []).append(step)
    return grouped


# ------------------------------------------------------------------ per-agent status + snapshot
def _guardrail(runs):
    d = runs[-1]["details"]
    lines = [(OK if ok else FAIL, label) for label, ok in d.get("checks", [])]
    if "blocked" in runs[-1]["summary"]:
        if d.get("matched"):
            lines.append((FAIL, f"Matched: “{_short(d['matched'], 40)}”"))
        return RED, "blocked", lines
    return GREEN, "input OK", lines


def _planner(runs):
    d = runs[-1]["details"]
    if not d.get("needs_retrieval", True):
        return GREEN, "greeting · no search", [(INFO, "Intent: chitchat"), (INFO, "Search skipped")]
    lines = [(INFO, f"Intent: {d.get('intent', '?')}")]
    for q in d.get("sub_questions", [])[:3]:
        lines.append((INFO, f"Search: “{_short(q, 38)}”"))
    lines.append((INFO, f"Fetch top {d.get('top_k', '?')} chunks"))
    if d.get("fallback"):
        lines.insert(0, (WARN, "LLM error → default plan"))
        return AMBER, "default plan", lines
    n = len(d.get("sub_questions", []))
    return GREEN, f"{d.get('intent', '?')} · {n} search{'es' if n != 1 else ''}", lines


def _retriever(runs):
    d = runs[-1]["details"]
    sources = d.get("sources", [])
    cutoff = d.get("min_relevance")
    cut = f"{cutoff:.2f}" if isinstance(cutoff, (int, float)) else "threshold"
    lines = []
    for s in sources[:3]:
        lines.append((OK, f"{_file(s['filename'])} · {_short(s['location'], 22)} · {s['relevance']:.2f}"))
    if len(sources) > 3:
        lines.append((INFO, f"+{len(sources) - 3} more chunk(s)"))
    dropped = d.get("dropped_low_relevance", 0)
    lines.append((INFO, f"{d.get('candidates', 0)} found · {dropped} below {cut} dropped"))
    if d.get("injection_lines_removed"):
        lines.append((WARN, f"{d['injection_lines_removed']} hidden instruction line(s) removed"))
    if not sources:
        lines.insert(0, (FAIL, f"Nothing ≥ {cut} (best {d.get('top_relevance', 0):.2f})"))
        return RED, f"0 relevant of {d.get('candidates', 0)}", lines
    note = f"{len(sources)} chunk{'s' if len(sources) != 1 else ''} · top {d.get('top_relevance', 0):.2f}"
    return (AMBER if d.get("injection_lines_removed") else GREEN), note, lines


def _reasoner(runs):
    d = runs[-1]["details"]
    lines = [(INFO, f"Model: {_short(d.get('model', '?'), 30)}"),
             (INFO, "Sources sent: " + ("S1" if d.get("sources_sent") == 1 else f"S1–S{d.get('sources_sent', '?')}"))]
    if len(runs) > 1:
        lines.append((RETRY, f"Rewrite #{len(runs) - 1} using validator feedback"))
    if d.get("error") and not d.get("used_fallback"):
        lines.append((FAIL, _short(d["error"], 60)))
        return RED, "LLM error", lines
    if d.get("used_fallback"):
        lines.append((WARN, f"Claude failed: {_short(d.get('error', ''), 45)}"))
        return AMBER, "offline answer", lines
    if len(runs) > 1:
        return AMBER, f"rewritten ({len(runs)} attempts)", lines
    return GREEN, "draft ready", lines


def _validator(runs):
    last = runs[-1]
    d = last["details"]
    verdict = last["summary"].split(": ")[-1]
    lines = []
    if len(runs) > 1:
        first_problem = (runs[0]["details"].get("problems") or ["failed"])[0]
        lines.append((RETRY, f"Attempt 1 failed: {_short(first_problem, 40)}"))
    if verdict == "insufficient":
        lines.append((FAIL, "Model said: answer not in sources"))
        return RED, "not in sources", lines
    lines += [(OK if ok else FAIL, label) for label, ok in d.get("checks", [])]
    if d.get("cited"):
        lines.append((INFO, "Cited: " + ", ".join(d["cited"])))
    if verdict == "pass":
        return (AMBER, "passed on retry", lines) if len(runs) > 1 else (GREEN, "all checks pass", lines)
    return RED, f"failed {len(runs)}×", lines


HANDLERS = {"guardrail": _guardrail, "planner": _planner, "retriever": _retriever,
            "reasoner": _reasoner, "validator": _validator}


def step_details(result: dict) -> list:
    """[(key, icon, name, colour, note, lines), ...] for the 5 agents + the outcome card."""
    steps = _steps_by_agent(result.get("trace", []))
    out = []
    for key, icon, name in STEPS:
        runs = steps.get(key)
        if not runs:
            out.append((key, icon, name, GREY, "skipped", [(INFO, "Not needed: flow ended earlier")]))
            continue
        colour, note, lines = HANDLERS[key](runs)
        out.append((key, icon, name, colour, note, lines))
    colour, label = OUTCOME.get(result.get("status"), (GREY, str(result.get("status", "?"))))
    out.append(("outcome", "🏁", label, colour, f"{result.get('total_ms', 0) / 1000:.1f} s total",
                [(INFO, _short(result.get("reason", ""), 80))]))
    return out


def step_states(result: dict) -> list:
    """Short form (no snapshot lines) - kept for tests and simple callers."""
    return [(k, i, n, c, note) for k, i, n, c, note, _ in step_details(result)]


# ------------------------------------------------------------------ HTML
SYMBOL_COLOUR = {OK: "#1B5E20", FAIL: "#B71C1C", WARN: "#8A5200", RETRY: "#8A5200", INFO: "#555"}


def _card(number, icon, name, colour, note, lines) -> str:
    faded = "opacity:.6;" if colour == GREY else ""
    rows = "".join(
        f'<div style="display:flex;gap:5px;margin-top:3px">'
        f'<span style="color:{SYMBOL_COLOUR.get(sym, "#555")};font-weight:700;min-width:10px">{sym}</span>'
        f'<span style="word-break:break-word">{html.escape(text)}</span></div>'
        for sym, text in lines
    )
    badge = f"{number}. " if number else ""
    return (
        f'<div style="flex:1 1 180px;min-width:160px;border-radius:10px;overflow:hidden;'
        f'border:1px solid {colour};background:#fff;{faded}box-shadow:0 1px 3px rgba(0,0,0,.12)">'
        f'<div style="background:{colour};color:#fff;padding:6px 9px;display:flex;justify-content:space-between;'
        f'align-items:center;gap:6px"><span style="font-weight:700;font-size:13px">{icon} {badge}{html.escape(name)}</span>'
        f'<span style="font-size:10.5px;opacity:.95;text-align:right">{html.escape(note)}</span></div>'
        f'<div style="padding:5px 9px 8px 9px;font-size:11.5px;line-height:1.35;color:#222">{rows}</div></div>'
    )


def flow_html(result: dict) -> str:
    """Workflow cards. 3 per row on a laptop, 1-2 per row on a phone."""
    cards = [_card(i + 1 if key != "outcome" else None, icon, name, colour, note, lines)
             for i, (key, icon, name, colour, note, lines) in enumerate(step_details(result))]
    legend = ('<div style="font-size:11px;color:#888;margin-top:6px">'
              f'<span style="color:{GREEN}">■</span> passed &nbsp; <span style="color:{AMBER}">■</span> retry / fallback / cleaned'
              f' &nbsp; <span style="color:{RED}">■</span> stopped here &nbsp; <span style="color:{GREY}">■</span> skipped'
              ' &nbsp;·&nbsp; ✓ check passed &nbsp; ✗ check failed</div>')
    return ('<div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 4px 0">' + "".join(cards) + "</div>" + legend)
