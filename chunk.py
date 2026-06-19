"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import re

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str

def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Extracts the intro as Chunk 0, then applies a sliding window
    across the entire text. Injects the title into all chunks.
    """
    page_id = int(record.get("page_id", 0))
    title = record.get("title", "").strip()
    content = record.get("content", "").strip()
    
    chunks = []
    chunk_id = 0
    
    # 1. The Intro Paragraph (Chunk 0)
    # Split by structural newlines to isolate the first paragraph cleanly
    paragraphs = [p.strip() for p in re.split(r'\n+', content) if p.strip()]
    if paragraphs:
        intro = paragraphs[0]
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=f"{title} - {intro}"))
        chunk_id += 1
    else:
        # Failsafe for entirely empty documents
        return [Chunk(page_id=page_id, chunk_id=chunk_id, text=title)]

    # 2. The Sliding Window
    # Start from Word 0 of the entire content to prevent a dead zone
    # between the intro and the body.
    words = content.split()
    window_size = 200
    stride = 150  # 200 size - 150 stride = 50 word overlap
    
    for i in range(0, len(words), stride):
        window_words = words[i:i + window_size]
        
        # Skip tiny trailing fragments to avoid polluting the index
        if len(window_words) < 20 and i > 0:
            continue
            
        chunk_text = f"{title} - {' '.join(window_words)}"
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))
        chunk_id += 1
        
    return chunks

def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks