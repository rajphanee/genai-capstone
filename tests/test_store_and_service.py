"""Tasks 5 + 6 + service: indexing, duplicates, deletion, search filters, persistence."""

import pytest

from src.errors import NotFoundError
from src.service import DocQAService

from .conftest import SAMPLES


def test_ingest_and_duplicate_detection(svc):
    data = (SAMPLES / "employee_handbook.pdf").read_bytes()
    first = svc.ingest("employee_handbook.pdf", data)
    again = svc.ingest("renamed_copy.pdf", data)          # same bytes, different name
    assert first["duplicate"] is False and again["duplicate"] is True
    assert len(svc.list_documents()) == 1


def test_search_returns_best_match_first(loaded):
    hits = loaded.store.search("paid time off days per year", k=3)
    assert hits[0]["filename"] == "employee_handbook.pdf"
    assert hits[0]["relevance"] >= hits[-1]["relevance"]


def test_search_can_be_limited_to_one_document(loaded):
    csv_id = next(d["doc_id"] for d in loaded.list_documents() if d["filename"].endswith(".csv"))
    hits = loaded.store.search("paid time off", k=5, doc_ids=[csv_id])
    assert hits and all(h["doc_id"] == csv_id for h in hits)


def test_delete_document(loaded):
    doc = loaded.list_documents()[0]
    loaded.delete_document(doc["doc_id"])
    assert doc["doc_id"] not in {d["doc_id"] for d in loaded.list_documents()}
    with pytest.raises(NotFoundError):
        loaded.delete_document(doc["doc_id"])


def test_data_survives_restart(loaded, settings):
    count = loaded.store.collection.count()
    reopened = DocQAService.create(settings)                # new object, same folder
    assert reopened.store.collection.count() == count
    assert reopened.ask("How many PTO days do employees get per year?")["status"] == "answered"


def test_history_is_logged(loaded, settings):
    loaded.ask("How many PTO days do employees get per year?")
    assert (settings.data_dir / "history.jsonl").read_text().count("\n") == 1
