"""
vectorstore.py - Task 5 (part 2) + Task 6: store chunk vectors in ChromaDB and search them.

Storage layout (all under DATA_DIR, default ./storage):
  storage/chroma/          ChromaDB's own files (persistent across restarts)
  storage/documents.json   our small registry: id, filename, chunk count, upload time

Document id = first 12 chars of the SHA-256 of the file's bytes.
  -> uploading the exact same file twice is detected and skipped (no duplicates).

Similarity: the collection uses cosine distance (0 = identical, 2 = opposite).
We report relevance = 1 - distance, so 1.0 is a perfect match.
"""

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone

from src.config import Settings
from src.errors import NotFoundError, VectorStoreError

log = logging.getLogger(__name__)


def document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


class VectorStore:
    def __init__(self, settings: Settings, embedder):
        import chromadb

        self.settings = settings
        self.embedder = embedder
        self.registry_path = settings.data_dir / "documents.json"
        self._lock = threading.Lock()
        try:
            self.client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            # One collection per embedding backend: their vectors have different sizes/meanings.
            self.collection = self.client.get_or_create_collection(
                name=f"documents_{embedder.name}",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,     # we always pass vectors ourselves
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not open ChromaDB at {settings.chroma_dir}: {exc}")

    # ------------------------------------------------------------ registry
    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VectorStoreError(f"{self.registry_path} is corrupted ({exc}). Delete the storage folder to reset.")

    def _save_registry(self, registry: dict) -> None:
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        os.replace(tmp, self.registry_path)        # atomic: never a half-written file

    def list_documents(self) -> list:
        docs = [d for d in self._load_registry().values() if d.get("backend") == self.embedder.name]
        return sorted(docs, key=lambda d: d["uploaded_at"])

    def get_document(self, doc_id: str) -> dict:
        doc = self._load_registry().get(doc_id)
        if not doc or doc.get("backend") != self.embedder.name:
            raise NotFoundError(f"document {doc_id!r} not found")
        return doc

    # ------------------------------------------------------------ write
    def add_document(self, doc_id: str, filename: str, chunks: list) -> dict:
        """Embed and store chunks. Returns the registry entry. Idempotent per doc_id."""
        with self._lock:
            registry = self._load_registry()
            existing = registry.get(doc_id)
            if existing and existing.get("backend") == self.embedder.name:
                return dict(existing, duplicate=True)

            texts = [c.text for c in chunks]
            try:
                vectors = self.embedder.embed(texts)
                # Chroma metadata values must be str/int/float/bool - no None, no dicts.
                metadatas = [{
                    "doc_id": doc_id,
                    "filename": filename,
                    "location": c.location,
                    "chunk_index": c.index,
                } for c in chunks]
                ids = [f"{doc_id}:{c.index}" for c in chunks]
                # Add in batches: Chroma has a max batch size, big Excel files can exceed it.
                for start in range(0, len(ids), 500):
                    end = start + 500
                    self.collection.add(ids=ids[start:end], embeddings=vectors[start:end],
                                        documents=texts[start:end], metadatas=metadatas[start:end])
            except Exception as exc:  # noqa: BLE001
                # Roll back anything partly written so the store never holds half a document.
                try:
                    self.collection.delete(where={"doc_id": doc_id})
                except Exception:  # noqa: BLE001
                    pass
                raise VectorStoreError(f"could not index '{filename}': {exc}")

            entry = {
                "doc_id": doc_id,
                "filename": filename,
                "chunks": len(chunks),
                "backend": self.embedder.name,
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            registry[doc_id] = entry
            self._save_registry(registry)
            return dict(entry, duplicate=False)

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            registry = self._load_registry()
            if doc_id not in registry:
                raise NotFoundError(f"document {doc_id!r} not found")
            try:
                self.collection.delete(where={"doc_id": doc_id})
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"could not delete {doc_id}: {exc}")
            del registry[doc_id]
            self._save_registry(registry)

    # ------------------------------------------------------------ search
    def search(self, query: str, k: int, doc_ids: list = None) -> list:
        """
        Return up to k hits, best first:
          [{"chunk_id", "doc_id", "filename", "location", "text", "relevance"}, ...]
        doc_ids: optional list to search only inside some documents.
        """
        count = self.collection.count()
        if count == 0:
            return []
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": list(doc_ids)}}
        try:
            result = self.collection.query(
                query_embeddings=self.embedder.embed([query]),
                n_results=min(k, count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"search failed: {exc}")

        hits = []
        for cid, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                         result["metadatas"][0], result["distances"][0]):
            hits.append({
                "chunk_id": cid,
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "location": meta["location"],
                "text": text,
                "relevance": round(1.0 - float(dist), 4),
            })
        return hits
