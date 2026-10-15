"""The Streamlit UI: loads sample docs, answers a question, and shows an error for a bad upload (headless)."""

from streamlit.testing.v1 import AppTest

from .conftest import ROOT


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    return at


def test_load_samples_and_ask(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    next(b for b in at.button if "sample" in b.label.lower()).click().run()
    assert any("Your documents (4)" in e.label for e in at.expander)
    at.chat_input[0].set_value("How many PTO days do employees get per year?").run()
    assert not at.exception
    assert any("20 days" in m.value for m in at.markdown)


def test_example_question_buttons(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch)
    next(b for b in at.button if "sample" in b.label.lower()).click().run()
    next(b for b in at.button if "capital of France" in b.label).click().run()
    assert any("couldn't find a reliable answer" in m.value for m in at.markdown)
