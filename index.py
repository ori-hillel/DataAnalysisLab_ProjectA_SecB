"""Offline index build: Selective Indexing for Tiny File Size & Exact Math."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
BM25_DATA_NAME = "bm25_data.json"

# We put the STOPWORDS in index.py to prevent garbage words from bloating the JSON
STOPWORDS = {
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "in", "out", "on", "off", "over", "under", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "which", "who", "whom", "what", "when",
    "where", "how", "why", "not", "no", "so", "it", "its", "he", "she",
    "they", "we", "you", "me", "him", "her", "us", "them", "my"
}

def build_bm25(records: List[Dict[str, Any]], out_dir: Path) -> None:
    inverted_index: Dict[str, Dict[str, int]] = {}
    doc_lengths: Dict[str, int] = {}
    total_length = 0
    total_docs = 0

    for record in records:
        page_id = str(record.get("page_id"))
        text = str(record.get("title", "")) + " " + str(record.get("content", ""))

        tokens = re.findall(r'\w+', text.lower())
        bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]

        # We calculate length using EVERYTHING so the BM25 formula behaves exactly the same
        doc_len = len(tokens) + len(bigrams)
        doc_lengths[page_id] = doc_len
        total_length += doc_len
        total_docs += 1

        # SELECTIVE STORAGE (Shrinking the file from 1GB to ~25MB)
        valid_terms = []
        
        # Save unigrams only if they aren't stopwords
        for t in tokens:
            if t not in STOPWORDS:
                valid_terms.append(t)

        # Save bigrams ONLY if both words carry semantic meaning (e.g. "apple_inc")
        # This completely eliminates millions of junk bigrams like "in_the" or "he_was"
        for i in range(len(tokens) - 1):
            w1 = tokens[i]
            w2 = tokens[i + 1]
            if w1 not in STOPWORDS and w2 not in STOPWORDS:
                valid_terms.append(w1 + "_" + w2)

        # Build the index using only the high-value terms
        term_counts = Counter(valid_terms)
        for term, count in term_counts.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][page_id] = count

    avgdl = total_length / total_docs if total_docs > 0 else 1.0

    bm25_data = {
        "N": total_docs,
        "avgdl": avgdl,
        "doc_lengths": doc_lengths,
        "inverted_index": inverted_index,
    }

    # Save natively. No zip, no lzma, no extra imports.
    (out_dir / BM25_DATA_NAME).write_text(
        json.dumps(bm25_data), encoding="utf-8"
    )

def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))

    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)
    page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    build_bm25(records, out_dir)
    return vectors, page_ids

def load_index(artifacts_dir: Optional[Path] = None) -> Tuple[np.ndarray, List[int]]:
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids