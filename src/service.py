"""
service.py - the ONE object that both the Streamlit UI and the FastAPI API use.

    svc = DocQAService.create()          # reads .env, opens ChromaDB, picks the LLM
    svc.ingest("policy.pdf", file_bytes) # Tasks 3-5: read -> chunk -> embed -> store
    svc.ask("How many PTO days?")        # Tasks 6-9: plan -> retrieve -> reason -> validate
    svc.list_documents() / svc.delete_document(doc_id)

Every question and its outcome are appended to storage/history.jsonl, which
gives you an audit trail and data for later evaluation.
"""

import json
import logging
import threading
from datetime import datetime, timezone

from agents.graph import AgentGraph
from src.chunking import chunk_sections
from src.config import Settings, load_settings
from vectordb.embeddings import get_embedder
from src.errors import IngestionError
from src.ingestion import extract_sections
from src.llm import get_llm
from vectordb.store import VectorStore, document_id

log = logging.getLogger(__name__)

# Similarity scores differ by embedding model, so the "too weak to use" cut-off does too.
DEFAULT_MIN_RELEVANCE = {"minilm": 0.30, "hashing": 0.12}


class DocQAService:
    def __init__(self, settings: Settings, llm, embedder):
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        if settings.min_relevance < 0:
            settings.min_relevance = DEFAULT_MIN_RELEVANCE.get(embedder.name, 0.3)
        self.store = VectorStore(settings, embedder)
        self.agents = AgentGraph(llm, self.store, settings)    # LangGraph workflow
        self.history_path = settings.data_dir / "history.jsonl"
        self._history_lock = threading.Lock()

    @classmethod
    def create(cls, settings: Settings = None, llm=None, embedder=None) -> "DocQAService":
        settings = settings or load_settings()
        return cls(settings, llm or get_llm(settings), embedder or get_embedder(settings))

    # ------------------------------------------------------------ status
    def info(self) -> dict:
        return {
            "llm": self.llm.name,
            "model": self.settings.claude_model if self.llm.name == "claude" else "rule-based (offline)",
            "embeddings": self.embedder.name,
            "documents": len(self.store.list_documents()),
            "chunks": self.store.collection.count(),
            "min_relevance": self.settings.min_relevance,
            "warning": self.settings.warning,
        }

    # ------------------------------------------------------------ documents
    def ingest(self, filename: str, data: bytes) -> dict:
        """Read, chunk, embed and store one file. Returns the registry entry."""
        sections = extract_sections(filename, data, self.settings.max_file_mb)      # raises on bad files
        chunks = chunk_sections(sections, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise IngestionError(f"'{filename}' produced no text chunks")
        doc_id = document_id(data)
        entry = self.store.add_document(doc_id, filename, chunks)
        if not entry["duplicate"]:
            # Keep the original file too, so it can be re-indexed later (e.g. after changing chunk size).
            (self.settings.uploads_dir / f"{doc_id}_{filename}").write_bytes(data)
        log.info("ingested %s -> %s (%d chunks, duplicate=%s)", filename, doc_id, len(chunks), entry["duplicate"])
        return dict(entry, sections=len(sections))

    def list_documents(self) -> list:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> None:
        doc = self.store.get_document(doc_id)
        self.store.delete_document(doc_id)
        stored = self.settings.uploads_dir / f"{doc_id}_{doc['filename']}"
        if stored.exists():
            stored.unlink()

    # ------------------------------------------------------------ questions
    def ask(self, question: str, doc_ids: list = None) -> dict:
        result = self.agents.run(question, doc_ids=doc_ids)
        self._log_history(question, result)
        return result

    def _log_history(self, question, result) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": question,
            "status": result["status"],
            "reason": result["reason"],
            "citations": [c["tag"] + ":" + c["filename"] for c in result["citations"]],
            "total_ms": result["total_ms"],
        }
        try:
            with self._history_lock, self.history_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # History is nice-to-have; a full disk must not break answering.
            log.warning("could not write history: %s", exc)
