"""Task 2: the REST API, including its error contract."""

import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.service import DocQAService

from .conftest import SAMPLES


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(api, "_service", DocQAService.create(settings))   # use the temp-folder service
    return TestClient(api.app, raise_server_exceptions=False)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["llm"] == "offline"


def test_upload_ask_list_delete(client):
    files = [("files", (p.name, p.read_bytes())) for p in sorted(SAMPLES.glob("*.*"))]
    files.append(("files", ("bad.exe", b"MZ")))
    r = client.post("/documents", files=files)
    assert r.status_code == 200
    results = {x["filename"]: x for x in r.json()["results"]}
    assert results["bad.exe"]["ok"] is False and results["bad.exe"]["error"]["code"] == "UNSUPPORTED_FILE"
    assert sum(x["ok"] for x in results.values()) == 4

    answer = client.post("/ask", json={"question": "How many PTO days do employees get per year?"}).json()
    assert answer["status"] == "answered" and answer["citations"]

    docs = client.get("/documents").json()["documents"]
    assert len(docs) == 4
    assert client.delete(f"/documents/{docs[0]['doc_id']}").status_code == 200
    missing = client.delete(f"/documents/{docs[0]['doc_id']}")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "NOT_FOUND"


def test_bad_question_gives_400_with_error_shape(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"


def test_all_bad_uploads_give_422(client):
    r = client.post("/documents", files=[("files", ("empty.txt", b""))])
    assert r.status_code == 422
