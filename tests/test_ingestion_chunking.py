"""Tasks 3 + 4: reading every file format, rejecting bad files, chunking."""

import pytest

from src.chunking import split_text
from src.errors import IngestionError, UnsupportedFileError
from src.ingestion import extract_sections

from .conftest import SAMPLES


def test_pdf_one_section_per_page():
    sections = extract_sections("h.pdf", (SAMPLES / "employee_handbook.pdf").read_bytes())
    assert [s.location for s in sections] == ["page 1", "page 2", "page 3"]
    assert "20 days of paid time off" in sections[0].text


def test_txt_csv_xlsx():
    txt = extract_sections("p.txt", (SAMPLES / "it_security_policy.txt").read_bytes())
    assert "14 characters" in txt[0].text
    csv = extract_sections("s.csv", (SAMPLES / "sales_by_region.csv").read_bytes())
    assert "Region: West | Quarter: Q2 | Revenue_USD_M: 4.8" in csv[0].text
    xlsx = extract_sections("v.xlsx", (SAMPLES / "vendor_contracts.xlsx").read_bytes())
    assert {s.meta["sheet"] for s in xlsx} == {"Contracts", "SLAs"}


def test_semicolon_csv_and_latin1_text():
    csv = extract_sections("x.csv", "name;city\nJosé;Málaga\n".encode("latin-1"))
    assert "city: Málaga" in csv[0].text


@pytest.mark.parametrize("name,data,error", [
    ("virus.exe", b"MZ...", UnsupportedFileError),
    ("noext", b"hello", UnsupportedFileError),
    ("empty.txt", b"", IngestionError),
    ("broken.pdf", b"this is not a pdf", IngestionError),
    ("broken.xlsx", b"PK not really a zip", IngestionError),
    ("blank.txt", b"   \n\n  ", IngestionError),
    ("header_only.csv", b"a,b,c\n", IngestionError),
])
def test_bad_files_raise_clear_errors(name, data, error):
    with pytest.raises(error):
        extract_sections(name, data)


def test_file_size_limit():
    with pytest.raises(UnsupportedFileError, match="limit"):
        extract_sections("big.txt", b"x" * (2 * 1024 * 1024), max_mb=1)


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(200))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0][-20:].split()[-1] in chunks[1]       # overlap: end of chunk 0 reappears in chunk 1
    assert split_text("short", 300, 60) == ["short"]
    assert split_text("   ", 300, 60) == []


# ---------------------------------------------------------------- real edge-case files in sample_docs/edge_cases
EDGE = SAMPLES / "edge_cases"


@pytest.mark.parametrize("name,error,message", [
    ("photo.jpg", UnsupportedFileError, "not supported"),
    ("archive.zip", UnsupportedFileError, "not supported"),
    ("photo_renamed.txt", IngestionError, "actually a JPEG"),
    ("not_really.pdf", IngestionError, "not a real PDF"),
    ("corrupt_budget.xlsx", IngestionError, "not a readable Excel"),
    ("empty.txt", IngestionError, "empty"),
    ("header_only.csv", IngestionError, "no data rows"),
    ("scanned_page.pdf", IngestionError, "no extractable text"),
    ("password_protected.pdf", IngestionError, "password-protected"),
])
def test_edge_case_files_are_rejected_with_clear_messages(name, error, message):
    with pytest.raises(error, match=message):
        extract_sections(name, (EDGE / name).read_bytes())
