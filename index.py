"""Offline index build and load (not timed at grading)."""
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


def build_bm25(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Build a BM25 inverted index with unigrams + bigrams.
    Bigrams capture phrase-level matching that significantly improves retrieval.
    """
    inverted_index: Dict[str, Dict[str, int]] = {}
    doc_lengths: Dict[str, int] = {}
    total_length = 0
    total_docs = 0

    for record in records:
        page_id = str(record.get("page_id"))
        text = str(record.get("title", "")) + " " + str(record.get("content", ""))

        tokens = re.findall(r'\w+', text.lower())
        bigrams = [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
        all_tokens = tokens + bigrams

        doc_len = len(all_tokens)
        doc_lengths[page_id] = doc_len
        total_length += doc_len
        total_docs += 1

        term_counts = Counter(all_tokens)
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

    (out_dir / BM25_DATA_NAME).write_text(
        json.dumps(bm25_data), encoding="utf-8"
    )


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full corpus, persist dense artifacts, and build BM25 bigram index.
    """
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


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed vectors and chunk page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids
