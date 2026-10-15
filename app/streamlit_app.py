"""
streamlit_app.py - the web interface. Works on laptop and smartphone.

Run:   streamlit run app/streamlit_app.py      ->  http://localhost:8501

Layout (single column, so it reads well on a phone):
  1. Title + one-line explanation
  2. "Your documents" box: upload files, or one tap to load sample docs
  3. Chat: ask a question, see the answer, its sources, and how the agents worked

Design notes:
  - The uploader accepts ANY file on purpose. Unsupported files (jpg, zip, exe...)
    are rejected by our own code with a clear message, which is part of the demo.
  - All logic lives in src/service.py; this file only draws the screen.
"""

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.flow_view import flow_html  # noqa: E402
from src.errors import DocQAError  # noqa: E402
from src.ingestion import SUPPORTED  # noqa: E402
from src.service import DocQAService  # noqa: E402

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered",
                   initial_sidebar_state="collapsed")

STATUS = {
    "answered": ("✅", "Answered from your documents"),
    "declined": ("🤷", "Not found in your documents"),
    "blocked": ("🛡️", "Blocked by a safety guardrail"),
    "chitchat": ("👋", "Greeting - no search needed"),
    "error": ("⚠️", "Something went wrong"),
}
EXAMPLES = [
    "How many PTO days do employees get per year?",
    "What was the West region revenue in Q2?",
    "What is the uptime SLA for Contoso Cloud?",
    "What is the capital of France?",
]


# ------------------------------------------------------------------ service (created once per server)
@st.cache_resource(show_spinner="Starting up - the first run downloads the embedding model (~80 MB)...")
def get_service() -> DocQAService:
    # On Streamlit Cloud, values from Settings -> Secrets become environment variables here.
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
    except Exception:  # noqa: BLE001 - no secrets file locally is normal
        pass
    return DocQAService.create()


try:
    svc = get_service()
except DocQAError as exc:
    st.error(f"Configuration problem: {exc}")
    st.info("Fix the .env file (see README, section 'Setup') and reload this page.")
    st.stop()

if svc.settings.warning:
    st.warning(svc.settings.warning)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)


def ingest_files(files) -> None:
    """Index uploaded files one by one; one bad file never stops the others."""
    for f in files:
        try:
            r = svc.ingest(f.name, f.getvalue())
            if r["duplicate"]:
                st.info(f"**{f.name}** is already indexed.")
            else:
                st.success(f"**{f.name}** indexed: {r['sections']} section(s), {r['chunks']} chunk(s).")
        except DocQAError as exc:
            st.error(f"**{f.name}** - {exc}")


def render_answer(result: dict) -> None:
    icon, label = STATUS.get(result["status"], ("", result["status"]))
    st.markdown(result["answer"])
    st.caption(f"{icon} {label} · {result['total_ms']/1000:.1f} s")
    if result["trace"]:
        # Live workflow: one coloured box per agent, based on what happened for THIS question
        st.markdown(flow_html(result), unsafe_allow_html=True)
    if result["citations"]:
        with st.expander(f"📚 Sources ({len(result['citations'])})"):
            for c in result["citations"]:
                st.markdown(f"**[{c['tag']}]** {c['filename']} · {c['location']} · relevance {c['relevance']:.2f}")
    if result["trace"]:
        with st.expander("🧠 Agent details (step by step)"):
            for step in result["trace"]:
                st.markdown(f"**{step['agent'].title()}** - {step['summary']} · {step['ms']} ms")
                st.json(step["details"], expanded=False)
            st.caption(f"Outcome: {result['reason']}")


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        result = svc.ask(question)
    except DocQAError as exc:     # e.g. empty or too-long question
        result = {"status": "error", "answer": str(exc), "citations": [], "trace": [],
                  "reason": exc.code, "total_ms": 0}
    st.session_state.messages.append({"role": "assistant", "content": result})


# ------------------------------------------------------------------ header
st.title("📄 Document Q&A")
st.caption("Upload PDF, TXT, CSV or Excel files and ask questions. AI agents find the answer in your files "
           "and show where every fact came from.")

# ------------------------------------------------------------------ documents
docs = svc.list_documents()
with st.expander(f"📁 Your documents ({len(docs)})", expanded=not docs):
    files = st.file_uploader("Choose one or more files", accept_multiple_files=True,
                             key=f"up_{st.session_state.uploader_key}",
                             help=f"Supported: {', '.join(sorted(SUPPORTED))}. Max {svc.settings.max_file_mb} MB each.")
    c1, c2 = st.columns(2)
    if c1.button("⬆️ Index files", disabled=not files, use_container_width=True, type="primary"):
        with st.spinner("Reading, chunking and embedding..."):
            ingest_files(files)
        st.session_state.uploader_key += 1          # clears the uploader for the next batch
    if c2.button("🧪 Load sample docs", use_container_width=True):
        with st.spinner("Loading 4 sample documents..."):
            for path in sorted(p for p in (ROOT / "sample_docs").iterdir() if p.is_file()):
                try:
                    svc.ingest(path.name, path.read_bytes())
                except DocQAError as exc:
                    st.error(f"{path.name}: {exc}")
        st.rerun()

    docs = svc.list_documents()
    for d in docs:
        a, b = st.columns([5, 1])
        a.markdown(f"• {d['filename']} <span style='color:gray'>({d['chunks']} chunks)</span>",
                   unsafe_allow_html=True)
        if b.button("🗑", key=f"del_{d['doc_id']}", help="Remove this document"):
            svc.delete_document(d["doc_id"])
            st.rerun()

# ------------------------------------------------------------------ chat
if not st.session_state.messages:
    if docs:
        st.markdown("**Try a question:**")
        for i, example in enumerate(EXAMPLES):
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                ask(example)
                st.rerun()
    else:
        st.info("Start by uploading files above, or tap **Load sample docs**.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["content"])

question = st.chat_input("Ask about your documents...")
if question:
    with st.spinner("Planning → searching → answering → checking..."):
        ask(question)
    st.rerun()

if st.session_state.messages and st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()

info = svc.info()
st.caption(f"LLM: {info['llm']} ({info['model']}) · Embeddings: {info['embeddings']} · "
           f"Vector DB: ChromaDB · {info['chunks']} chunks indexed")
