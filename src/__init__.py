"""
src - core application code (no web code, no agents, no vector DB here).

Project map (read in this order - it follows one request from upload to answer):

  src/config.py          settings from .env
  src/ingestion.py       file -> text      (PDF, TXT, CSV, Excel; everything else rejected)
  src/chunking.py        text -> overlapping chunks
  vectordb/embeddings.py chunk -> vector   (all-MiniLM-L6-v2)
  vectordb/store.py      vectors -> ChromaDB, similarity search
  agents/planner.py      Agent 1: question -> plan
  agents/retriever.py    Agent 2: plan -> relevant chunks
  agents/reasoner.py     Agent 3: chunks -> cited answer (Claude)
  agents/validator.py    Agent 4: answer -> pass / fix / decline
  agents/graph.py        LangGraph wiring of the 4 agents
  src/guardrails.py      safety checks used by the agents
  src/service.py         one object used by the UI, API and CLI
  app/streamlit_app.py   the web UI
"""

__version__ = "1.1.0"
