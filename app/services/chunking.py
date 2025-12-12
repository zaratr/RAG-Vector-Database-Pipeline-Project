"""Chunking utilities."""
from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[dict]:
    """Split text into overlapping chunks returning metadata."""

    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append(
            {
                "index": index,
                "text": chunk_text,
                "start_offset": start,
                "end_offset": end,
            }
        )
        index += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks
