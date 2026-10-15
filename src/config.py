"""
config.py - all settings in one place, read from environment variables / .env.

You never edit code to change behaviour. Copy .env.example to .env and edit:

  LLM_PROVIDER        offline (default, no key needed) | claude
  ANTHROPIC_API_KEY   needed only when LLM_PROVIDER=claude
  CLAUDE_MODEL        model name for Claude
  EMBEDDING_BACKEND   auto (default) | minilm | hashing
                        minilm  = real semantic embeddings (all-MiniLM-L6-v2, ~80MB, downloaded once)
                        hashing = tiny offline embeddings, no download (tests/CI)
                        auto    = try minilm, fall back to hashing if download fails
  ANTHROPIC_WORKSPACE_ID  only for keys not tied to a workspace (error says 'anthropic-workspace-id')
  DATA_DIR            where ChromaDB + uploaded files are stored (default ./vectordb/data)
  CHUNK_SIZE          characters per chunk (default 800)
  CHUNK_OVERLAP       characters shared by neighbouring chunks (default 150)
  TOP_K               how many chunks the retriever returns (default 4)
  MIN_RELEVANCE       0..1; chunks scoring below this are ignored.
                      Default depends on the embeddings: 0.30 for minilm, 0.12 for hashing
                      (hashing scores are naturally lower because it only matches exact words).
  MAX_FILE_MB         biggest file you may upload (default 20)
"""

import os
from dataclasses import dataclass
from pathlib import Path

from src.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"


@dataclass
class Settings:
    llm_provider: str = "offline"
    warning: str = ""                # shown in the UI, e.g. 'no API key, running offline'
    anthropic_api_key: str = ""
    anthropic_workspace_id: str = ""
    claude_model: str = DEFAULT_CLAUDE_MODEL
    embedding_backend: str = "auto"
    data_dir: Path = PROJECT_ROOT / "vectordb" / "data"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    min_relevance: float = -1.0       # -1 = "not set": service picks a default per embedding backend
    max_file_mb: int = 20
    max_question_chars: int = 1000
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


def _read_dotenv(path: Path) -> dict:
    """Minimal .env parser: KEY=VALUE lines, '#' comments, optional quotes."""
    values = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _int(env: dict, name: str, default: int, low: int, high: int) -> int:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}")
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def _float(env: dict, name: str, default: float, low: float, high: float) -> float:
    raw = str(env.get(name, "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}")
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}, got {value}")
    return value


def load_settings(env: dict = None, use_dotenv: bool = True) -> Settings:
    """
    Build Settings. Real environment variables win over .env values.
    Tests pass `env={...}, use_dotenv=False` so the developer's machine never leaks in.
    """
    merged = _read_dotenv(PROJECT_ROOT / ".env") if use_dotenv else {}
    merged.update(dict(os.environ) if env is None else env)

    # Streamlit Cloud keeps secrets in st.secrets, which it also exposes as env vars,
    # so nothing special is needed here for deployment.

    provider = merged.get("LLM_PROVIDER", "offline").strip().lower()
    if provider not in ("offline", "claude"):
        raise ConfigError(f"LLM_PROVIDER must be 'offline' or 'claude', got {provider!r}")

    api_key = merged.get("ANTHROPIC_API_KEY", "").strip()
    warning = ""
    if provider == "claude" and (not api_key or "your-key" in api_key or api_key.endswith("...")):
        # Never crash on a missing key: run offline and tell the user clearly why.
        provider = "offline"
        api_key = ""
        warning = ("No Anthropic API key found in .env, so answers are extracted from your documents "
                   "without Claude. Add ANTHROPIC_API_KEY to .env and restart for written answers.")

    backend = merged.get("EMBEDDING_BACKEND", "auto").strip().lower()
    if backend not in ("auto", "minilm", "hashing"):
        raise ConfigError(f"EMBEDDING_BACKEND must be auto, minilm or hashing, got {backend!r}")

    s = Settings(
        llm_provider=provider,
        warning=warning,
        anthropic_api_key=api_key,
        anthropic_workspace_id=merged.get("ANTHROPIC_WORKSPACE_ID", "").strip(),
        claude_model=(merged.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL).strip(),
        embedding_backend=backend,
        chunk_size=_int(merged, "CHUNK_SIZE", 800, 200, 4000),
        chunk_overlap=_int(merged, "CHUNK_OVERLAP", 150, 0, 1000),
        top_k=_int(merged, "TOP_K", 4, 1, 20),
        min_relevance=_float(merged, "MIN_RELEVANCE", -1.0, -1.0, 1.0),
        max_file_mb=_int(merged, "MAX_FILE_MB", 20, 1, 200),
    )
    if s.chunk_overlap >= s.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    data_dir = merged.get("DATA_DIR", "").strip()
    if data_dir:
        s.data_dir = Path(data_dir).expanduser().resolve()
    try:
        s.chroma_dir.mkdir(parents=True, exist_ok=True)
        s.uploads_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(f"cannot create data folder {s.data_dir}: {exc}")
    return s
