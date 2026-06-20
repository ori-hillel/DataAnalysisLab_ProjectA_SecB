"""Hybrid Retrieval: dense (FAISS) + sparse (BM25 with bigrams) fusion."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np

from embed import embed_queries
from index import load_index
from utils import ARTIFACTS_DIR, K_EVAL

BM25_DATA_NAME = "bm25_data.json"

STOPWORDS = {
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "in", "out", "on", "off", "over", "under", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "which", "who", "whom", "what", "when",
    "where", "how", "why", "not", "no", "so", "it", "its", "he", "she",
    "they", "we", "you", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "s", "t", "can", "will", "just", "don", "should",
    "now",
}


def tokenize(text: str) -> List[str]:
    """Tokenize and remove stopwords."""
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def score_bm25_all(query_tokens: List[str], bm25_data: Dict) -> Dict[str, float]:
    """Score all documents using BM25 with unigram + bigram query terms."""
    N = bm25_data["N"]
    avgdl = bm25_data["avgdl"]
    doc_lengths = bm25_data["doc_lengths"]
    inverted_index = bm25_data["inverted_index"]
    k1, b = 1.2, 0.75

    # Build bigrams from query tokens
    bigrams = [query_tokens[i] + "_" + query_tokens[i + 1]
               for i in range(len(query_tokens) - 1)]

    scores: Dict[str, float] = {}
    for term in query_tokens + bigrams:
        if term not in inverted_index:
            continue
        doc_freqs = inverted_index[term]
        df = len(doc_freqs)
        idf = math.log(((N - df + 0.5) / (df + 0.5)) + 1.0)
        for pid, tf in doc_freqs.items():
            dl = doc_lengths.get(pid, avgdl)
            score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl))))
            scores[pid] = scores.get(pid, 0.0) + score
    return scores


def _min_max_norm(vals: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not vals:
        return {}
    v = list(vals.values())
    lo, hi = min(v), max(v)
    rng = hi - lo
    if rng < 1e-12:
        return {k: 0.5 for k in vals}
    return {k: (s - lo) / rng for k, s in vals.items()}


def search_batch(queries: List[str], *, top_k: int = K_EVAL,
                 artifacts_dir: Path | None = None) -> List[List[int]]:
    root = artifacts_dir or ARTIFACTS_DIR
    corpus_vectors, page_ids = load_index(root)
    index = faiss.IndexFlatIP(corpus_vectors.shape[1])
    index.add(corpus_vectors)

    with open(root / BM25_DATA_NAME, 'r') as f:
        bm25_data = json.load(f)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    # Dense retrieval: FAISS top-100
    scores_mat, faiss_indices = index.search(query_vectors, 100)
    ranked = []

    # Fusion weights: alpha for dense, (1-alpha) for sparse
    alpha = 0.45

    for q_idx, query in enumerate(queries):
        # Dense scores per page_id
        dense_scores: Dict[str, float] = {}
        for j in range(100):
            idx = int(faiss_indices[q_idx][j])
            if idx == -1:
                continue
            pid = str(page_ids[idx])
            sc = float(scores_mat[q_idx][j])
            if pid not in dense_scores or sc > dense_scores[pid]:
                dense_scores[pid] = sc

        # Sparse BM25 scores (bigram-enhanced)
        bm25_scores = score_bm25_all(tokenize(query), bm25_data)

        # Min-max normalize both
        d_norm = _min_max_norm(dense_scores)
        s_norm = _min_max_norm(bm25_scores)

        # Union of all candidates
        all_pids = set(d_norm.keys()) | set(s_norm.keys())

        # Weighted linear combination
        final_scores: Dict[str, float] = {}
        for pid in all_pids:
            d = d_norm.get(pid, 0.0)
            s = s_norm.get(pid, 0.0)
            final_scores[pid] = alpha * d + (1 - alpha) * s

        top_k_ids = [int(p) for p, _ in sorted(final_scores.items(),
                                                 key=lambda x: x[1],
                                                 reverse=True)[:top_k]]
        ranked.append(top_k_ids + [-1] * (top_k - len(top_k_ids)))

    return ranked
