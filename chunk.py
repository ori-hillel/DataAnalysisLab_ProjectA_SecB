"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str

def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Content-Aware Paragraph Chunking with Title Injection.
    Preserves syntactic boundaries and forces global context into every chunk.
    """
    page_id = int(record.get("page_id", 0))
    title = record.get("title", "").strip()
    content = record.get("content", "").strip()
    
    # Content-Aware Split: Use actual paragraphs to prevent breaking thoughts
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    
    if not paragraphs:
        return [Chunk(page_id=page_id, chunk_id=0, text=title)]
        
    chunks = []
    chunk_id = 0
    current_chunk_words = []
    current_word_count = 0
    
    # Group paragraphs up to ~150 words to maintain optimal token length
    for para in paragraphs:
        words = para.split()
        
        if current_word_count + len(words) > 150 and current_chunk_words:
            # INJECT TITLE: This guarantees the model knows the subject
            chunk_text = f"{title} - {' '.join(current_chunk_words)}"
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))
            chunk_id += 1
            current_chunk_words = []
            current_word_count = 0
        
        current_chunk_words.extend(words)
        current_word_count += len(words)
        
    # Flush the remaining words
    if current_chunk_words:
        chunk_text = f"{title} - {' '.join(current_chunk_words)}"
        chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id, text=chunk_text))
        
    return chunks

def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    total = len(records)
    print(f"\n--- Starting Content-Aware Chunking ({total} entries) ---")
    for i, record in enumerate(records):
        if i > 0 and i % 500 == 0:
            print(f"  ...Processed {i}/{total}")
        chunks.extend(chunk_entry(record))
    print(f"Total semantic chunks generated: {len(chunks)}\n")
    return chunks