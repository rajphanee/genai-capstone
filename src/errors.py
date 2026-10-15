"""
errors.py - every EXPECTED error in one place.

Rule of this project:
  - Expected problems (bad file, empty question, missing API key...) raise a
    subclass of DocQAError. The UI shows the message; the API returns HTTP 4xx.
  - Anything else is a real bug: it is logged with a full traceback and the
    API returns HTTP 500 with a generic message (we never leak internals).
"""


class DocQAError(Exception):
    """Base class. `code` is a short id; `http_status` is used by the API."""
    code = "DOCQA_ERROR"
    http_status = 400


class ConfigError(DocQAError):
    """Wrong/missing setting, e.g. LLM_PROVIDER=claude without an API key."""
    code = "CONFIG_ERROR"
    http_status = 500


class ValidationError(DocQAError):
    """The user's input is wrong: empty question, question too long, etc."""
    code = "VALIDATION_ERROR"


class UnsupportedFileError(DocQAError):
    """File extension not in PDF/TXT/CSV/XLSX, or file too large."""
    code = "UNSUPPORTED_FILE"
    http_status = 415


class IngestionError(DocQAError):
    """The file has the right extension but could not be read (corrupt, empty, encrypted)."""
    code = "INGESTION_ERROR"
    http_status = 422


class NotFoundError(DocQAError):
    """A document id that does not exist."""
    code = "NOT_FOUND"
    http_status = 404


class VectorStoreError(DocQAError):
    """ChromaDB failed to add/query/delete."""
    code = "VECTOR_STORE_ERROR"
    http_status = 500


class LLMError(DocQAError):
    """The LLM call failed (network, timeout, rate limit, bad key)."""
    code = "LLM_ERROR"
    http_status = 502


class GuardrailViolation(DocQAError):
    """The question was blocked by an input guardrail (e.g. prompt injection)."""
    code = "GUARDRAIL_BLOCKED"
