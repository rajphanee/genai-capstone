"""
chunking.py - Task 4: split section text into overlapping chunks for embedding.

WHY CHUNK AT ALL?
  Embedding models work best on a few hundred words. A 50-page PDF as one
  vector would match everything a little and nothing well.

WHY OVERLAP?
  If a key sentence sits on the border between chunk 3 and chunk 4, the
  overlap makes sure at least one chunk contains it whole.

HOW WE SPLIT (in order of preference):
  paragraph break  ->  sentence end  ->  any space  ->  hard cut
  so chunks end at natural boundaries whenever possible.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    location: str
    index: int          # position of this chunk inside its document (0, 1, 2...)
    meta: dict


def _best_cut(text: str, start: int, end: int) -> int:
    """Return the best place to cut text[start:end] - prefer natural boundaries."""
    if end >= len(text):
        return len(text)
    window = text[start:end]
    min_keep = len(window) // 2          # never cut in the first half; chunks would get tiny
    for boundary in ("\n\n", ". ", "? ", "! ", "\n", " "):
        pos = window.rfind(boundary)
        if pos >= min_keep:
            return start + pos + len(boundary)
    return end                            # no boundary found: hard cut


def split_text(text: str, size: int, overlap: int) -> list:
    """Split one string into overlapping pieces of at most `size` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        cut = _best_cut(text, start, start + size)
        piece = text[start:cut].strip()
        if piece:
            pieces.append(piece)
        if cut >= len(text):
            break
        # Step back by `overlap`, but always move forward at least 1 char (avoids an endless loop).
        start = max(cut - overlap, start + 1)
    return pieces


def chunk_sections(sections: list, size: int, overlap: int) -> list:
    """Chunk every Section and keep its location on each chunk."""
    chunks = []
    for section in sections:
        for piece in split_text(section.text, size, overlap):
            chunks.append(Chunk(piece, section.location, len(chunks), dict(section.meta)))
    return chunks
