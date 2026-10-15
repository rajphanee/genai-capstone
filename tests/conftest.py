"""
Shared test fixtures.

Every test gets a brand-new temp storage folder and uses:
  - OfflineLLM        (no API key, no network)
  - hashing embeddings (no model download)
so the suite runs anywhere in a few seconds, including GitHub Actions.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.service import DocQAService  # noqa: E402

SAMPLES = ROOT / "sample_docs"


@pytest.fixture
def settings(tmp_path):
    return load_settings(env={"DATA_DIR": str(tmp_path), "EMBEDDING_BACKEND": "hashing",
                              "LLM_PROVIDER": "offline"}, use_dotenv=False)


@pytest.fixture
def svc(settings):
    return DocQAService.create(settings)


@pytest.fixture
def loaded(svc):
    """A service with all four sample documents indexed."""
    for path in sorted(SAMPLES.glob("*.*")):
        svc.ingest(path.name, path.read_bytes())
    return svc
