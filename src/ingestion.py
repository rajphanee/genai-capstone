"""
ingestion.py - Task 3: turn an uploaded file into clean text "sections".

A Section is a piece of text plus WHERE it came from (page 3, rows 1-20,
sheet "Q3"...). Keeping the location lets the final answer say exactly where
a fact was found, which is what makes the answer checkable.

Supported formats:
  .pdf          one section per page            (pypdf)
  .txt / .md    one section for the whole file  (tries utf-8, then latin-1)
  .csv          one section per 20 rows         (pandas)
  .xlsx / .xls  one section per 20 rows, per sheet (pandas + openpyxl)

WHY rows become "column: value" text:
  An embedding model understands "Region: West | Revenue: 4.2M" much better
  than a raw "West,4.2M" line with no headers.

Every failure raises a clear DocQAError so the UI can show a friendly message.
"""

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.errors import IngestionError, UnsupportedFileError

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls"}
ROWS_PER_SECTION = 20


@dataclass
class Section:
    text: str
    location: str                       # human readable, e.g. "page 2" or "sheet Sales, rows 1-20"
    meta: dict = field(default_factory=dict)


def clean_text(text: str) -> str:
    """Remove null bytes, collapse runs of spaces, keep paragraph breaks."""
    text = (text or "").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)   # 3+ blank lines -> 1 blank line
    return text.strip()


def validate_upload(filename: str, size_bytes: int, max_mb: int) -> str:
    """Check name + size BEFORE reading the content. Returns the lowercase extension."""
    if not filename or not filename.strip():
        raise UnsupportedFileError("file has no name")
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise UnsupportedFileError(
            f"'{filename}': {ext or 'files without an extension'} is not supported. "
            f"Please upload PDF, TXT, CSV or Excel (.xlsx) files."
        )
    if size_bytes == 0:
        raise IngestionError(f"'{filename}' is empty")
    if size_bytes > max_mb * 1024 * 1024:
        raise UnsupportedFileError(f"'{filename}' is {size_bytes / 1_048_576:.1f} MB; the limit is {max_mb} MB")
    return ext


# First bytes ("magic numbers") of common binary formats. Used to catch a file
# whose content does not match its extension, e.g. a photo renamed to notes.txt.
MAGIC = {
    b"\xff\xd8\xff": "a JPEG image",
    b"\x89PNG": "a PNG image",
    b"GIF8": "a GIF image",
    b"PK\x03\x04": "a ZIP archive (or Office file)",
    b"%PDF": "a PDF",
    b"MZ": "a Windows program",
}


def sniff(data: bytes) -> str:
    """Return a description if the bytes look like a known binary format, else ''."""
    for magic, label in MAGIC.items():
        if data.startswith(magic):
            return label
    return ""


def check_content_matches(filename: str, ext: str, data: bytes) -> None:
    kind = sniff(data[:8])
    if ext == ".pdf" and not data.lstrip()[:5].startswith(b"%PDF"):
        raise IngestionError(f"'{filename}' is named .pdf but is {kind or 'not a real PDF file'}.")
    if ext == ".xlsx" and kind and not kind.startswith("a ZIP"):
        raise IngestionError(f"'{filename}' is named .xlsx but is actually {kind}.")
    if ext in (".txt", ".md", ".csv"):
        if kind:
            raise IngestionError(f"'{filename}' is named {ext} but is actually {kind}.")
        if b"\x00" in data[:4096]:
            raise IngestionError(f"'{filename}' contains binary data, not text.")


# ------------------------------------------------------------------ readers
def _join_pdf_lines(raw: str) -> str:
    """
    PDFs break lines at the page edge, mid-sentence. We re-join them, except:
      - blank lines stay (paragraph breaks)
      - a short line with no ending punctuation is treated as a HEADING and
        kept on its own line, so "Remote Work" does not glue onto the next sentence.
    """
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            out.append("\n\n")
        elif len(line) < 60 and not line.endswith((".", ",", ";", ":", "?", "!")):
            out.append("\n" + line + "\n")          # heading
        else:
            out.append(line + " ")
    return "".join(out)


def _read_pdf(data: bytes, name: str) -> list:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Many "encrypted" PDFs only have an owner password; an empty user password opens them.
            try:
                opened = reader.decrypt("")
            except Exception:  # noqa: BLE001 - pypdf raises several types here
                opened = 0
            if not opened:
                raise IngestionError(f"'{name}' is password-protected. Remove the password and upload again.")
        sections = []
        for number, page in enumerate(reader.pages, start=1):
            text = clean_text(_join_pdf_lines(page.extract_text() or ""))
            if text:
                sections.append(Section(text, f"page {number}", {"page": number}))
    except IngestionError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise IngestionError(f"'{name}' is not a readable PDF: {exc}")
    if not sections:
        # NOTE: scanned PDFs are images; they need OCR, which we do not do (see README limitations).
        raise IngestionError(f"'{name}' has no extractable text (is it a scanned image? OCR is not supported).")
    return sections


def _read_text(data: bytes, name: str) -> list:
    text = ""
    for encoding in ("utf-8-sig", "latin-1"):     # utf-8-sig also strips a BOM if present
        try:
            text = clean_text(data.decode(encoding))
            break
        except UnicodeDecodeError:
            continue
    if not text:
        raise IngestionError(f"'{name}' contains no text")
    return [Section(text, "full text", {})]


def _frame_to_sections(df, name: str, sheet: str = None) -> list:
    """Turn a pandas DataFrame into Sections of ROWS_PER_SECTION rows each."""
    df = df.dropna(how="all")                       # drop fully empty rows
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]  # drop unnamed index columns
    if df.empty:
        return []
    columns = [str(c).strip() for c in df.columns]
    sections = []
    for start in range(0, len(df), ROWS_PER_SECTION):
        block = df.iloc[start:start + ROWS_PER_SECTION]
        lines = []
        for _, row in block.iterrows():
            parts = [f"{col}: {val}" for col, val in zip(columns, row.tolist()) if str(val).strip() not in ("", "nan", "None")]
            if parts:
                lines.append(" | ".join(parts))
        if not lines:
            continue
        first, last = start + 1, start + len(block)
        where = f"rows {first}-{last}" if sheet is None else f"sheet '{sheet}', rows {first}-{last}"
        header = f"Table from {name}" + (f", sheet {sheet}" if sheet else "") + f". Columns: {', '.join(columns)}."
        sections.append(Section(header + "\n" + "\n".join(lines), where,
                                {"sheet": sheet or "", "row_start": first, "row_end": last}))
    return sections


def _read_csv(data: bytes, name: str) -> list:
    import pandas as pd

    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            # sep=None + python engine = auto-detect comma / semicolon / tab
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise IngestionError(f"'{name}' is not a valid CSV: {exc}")
    else:
        raise IngestionError(f"'{name}' has an unknown text encoding: {last_error}")
    sections = _frame_to_sections(df, name)
    if not sections:
        raise IngestionError(f"'{name}' has no data rows")
    return sections


def _read_excel(data: bytes, name: str, ext: str) -> list:
    import pandas as pd

    engine = "openpyxl" if ext == ".xlsx" else None   # .xls needs the optional 'xlrd' package
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, engine=engine)  # None = all sheets
    except ImportError as exc:
        raise IngestionError(f"'{name}': old .xls files need 'pip install xlrd' ({exc}). Or save as .xlsx.")
    except Exception as exc:  # noqa: BLE001 - openpyxl/zipfile raise many different types for corrupt files
        raise IngestionError(f"'{name}' is not a readable Excel file: {exc}")
    sections = []
    for sheet_name, df in sheets.items():
        sections += _frame_to_sections(df, name, sheet=str(sheet_name))
    if not sections:
        raise IngestionError(f"'{name}' has no data in any sheet")
    return sections


def extract_sections(filename: str, data: bytes, max_mb: int = 20) -> list:
    """Main entry point: validate, then dispatch to the right reader."""
    ext = validate_upload(filename, len(data), max_mb)
    check_content_matches(filename, ext, data)
    log.info("ingesting %s (%s, %d bytes)", filename, ext, len(data))
    if ext == ".pdf":
        return _read_pdf(data, filename)
    if ext in (".txt", ".md"):
        return _read_text(data, filename)
    if ext == ".csv":
        return _read_csv(data, filename)
    return _read_excel(data, filename, ext)
