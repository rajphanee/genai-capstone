"""
embeddings.py - Task 5 (part 1): turn text into vectors.

Two backends, same interface: embed(list_of_texts) -> list_of_vectors.

  MiniLMEmbedder   all-MiniLM-L6-v2 through ChromaDB's built-in ONNX runtime.
                   Real semantic embeddings ("car" is close to "vehicle").
                   ~80 MB, downloaded automatically the first time, then cached.

  HashingEmbedder  "feature hashing": each word (and word pair) is hashed into
                   one of 512 slots. No download, no GPU, deterministic.
                   Good enough for tests/CI; it matches WORDS, not MEANING.

EMBEDDING_BACKEND=auto tries MiniLM and silently falls back to hashing if the
download is blocked (corporate proxy, no internet), and logs a warning.

NOTE: vectors from different backends are NOT comparable, so vectorstore.py
keeps a separate Chroma collection per backend.
"""

import hashlib
import logging
import math
import re

from src.config import Settings
from src.errors import ConfigError

log = logging.getLogger(__name__)

STOPWORDS = set("""
a an the and or of to in on for is are was were be been do does did i my we our you your it its
what how when who which can with at by from this that these those much many get need there any
about as if after before per please tell me show give
""".split())


def tokenize(text: str) -> list:
    """lowercase words, minus stopwords, with very light plural stripping."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


class HashingEmbedder:
    name = "hashing"
    dims = 512

    def _slot(self, token: str) -> tuple:
        digest = hashlib.md5(token.encode("utf-8")).digest()   # md5 = stable across runs (Python's hash() is not)
        index = int.from_bytes(digest[:4], "little") % self.dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0             # random sign reduces collision bias
        return index, sign

    def embed(self, texts: list) -> list:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dims
            words = tokenize(text)
            features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]   # unigrams + bigrams
            for feature in features:
                i, sign = self._slot(feature)
                vec[i] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._fn = DefaultEmbeddingFunction()
        self._fn(["warm up"])     # forces the one-time model download NOW, so failures surface early

    def embed(self, texts: list) -> list:
        return [list(map(float, v)) for v in self._fn(list(texts))]


def get_embedder(settings: Settings):
    backend = settings.embedding_backend
    if backend == "hashing":
        return HashingEmbedder()
    try:
        return MiniLMEmbedder()
    except Exception as exc:  # noqa: BLE001 - network, proxy, onnx errors all mean "not available"
        if backend == "minilm":
            raise ConfigError(f"MiniLM embeddings unavailable ({exc}). Check internet or use EMBEDDING_BACKEND=hashing")
        log.warning("MiniLM unavailable (%s) - falling back to hashing embeddings", exc)
        return HashingEmbedder()
