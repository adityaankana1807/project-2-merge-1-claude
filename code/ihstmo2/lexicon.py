"""Multilingual MO-narrative normalization and BM25 text similarity.

Loads the seed lexicon (data/raw/multilingual_mo_lexicon.csv) mapping ~15 MO concepts
across English + 5 Indian languages (+ common Hindi/Hinglish slang) onto single
canonical tokens, so a Hindi FIR narrative mentioning "taala tod kar" and an English
narrative saying "forced lock" collapse to the same token before retrieval. This is a
small, unvalidated, hand-authored seed (documented as such deliberately, following the
disclosure standard used across all reviewed prior systems) — it is a starting point
for a real corpus-driven Indic MO lexicon, not a finished NLP asset.

Real multilingual transformer embeddings (e.g. LaBSE/IndicBERT) would be a strict
improvement over this lexical/BM25 layer but require model downloads and GPU-friendly
inference infra out of scope here; BM25 + lexicon-normalization is deliberately chosen
as a fully offline, auditable, reproducible baseline (consistent with one of the four
prior systems' honest self-assessment that hashed n-gram "embeddings" were not real
semantic embeddings either).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .config import DATA_RAW

_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "was", "were",
    "accused", "complainant", "fir", "police", "station", "case", "no",
    "thana", "thanadhyaksh", "tehrir", "mauka", "hone", "gaya", "ke", "ki",
    "ka", "se", "ko", "hai", "the", "par", "aur",
}


def load_lexicon(path: Path | None = None) -> Dict[str, str]:
    """Return {surface_phrase_lowercased: canonical_token} across all languages."""
    path = path or (DATA_RAW / "multilingual_mo_lexicon.csv")
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            concept, _lang, phrase = parts[0], parts[1], parts[2]
            mapping[phrase.strip().lower()] = f"mo_{concept}"
    return mapping


class MOTokenizer:
    """Normalizes MO narrative text: lowercases, replaces lexicon phrases with a single
    canonical token (longest-phrase-first so multi-word phrases match before their
    substrings), strips punctuation/stopwords, keeps everything else as word tokens."""

    def __init__(self, lexicon: Dict[str, str] | None = None):
        self.lexicon = lexicon if lexicon is not None else load_lexicon()
        self._phrases_desc = sorted(self.lexicon.keys(), key=len, reverse=True)

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        t = text.lower()
        for phrase in self._phrases_desc:
            if phrase in t:
                t = t.replace(phrase, f" {self.lexicon[phrase]} ")
        t = re.sub(r"[^a-z0-9_ऀ-ॿঀ-৿஀-௿ఀ-౿ಀ-೿\s]", " ", t)
        tokens = [tok for tok in t.split() if tok not in _STOPWORDS and len(tok) > 1]
        return tokens


class BM25:
    """Standard Okapi BM25 (k1=1.2, b=0.75) over a fixed document corpus, IDF fit once
    at construction time (never refit at query time, avoiding train/test leakage)."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.n_docs = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        df: Counter = Counter()
        for doc in corpus_tokens:
            for tok in set(doc):
                df[tok] += 1
        self.idf = {
            tok: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for tok, n in df.items()
        }
        self._doc_tf = [Counter(d) for d in corpus_tokens]

    def score(self, query_tokens: List[str], doc_index: int) -> float:
        if self.avgdl == 0:
            return 0.0
        tf = self._doc_tf[doc_index]
        dl = self.doc_len[doc_index]
        score = 0.0
        for tok in query_tokens:
            if tok not in tf:
                continue
            idf = self.idf.get(tok, 0.0)
            f = tf[tok]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
        return score

    def score_pair_tokens(self, tokens_a: List[str], tokens_b: List[str]) -> float:
        """Symmetric BM25-style score between two token lists not in the fitted corpus
        (used for pairwise case-linkage scoring rather than corpus retrieval): averages
        scoring each as a 'query' against a synthetic single-document 'corpus' made from
        the other, using the globally fitted IDF table."""
        def one_way(query: List[str], doc: List[str]) -> float:
            if not doc:
                return 0.0
            tf = Counter(doc)
            dl = len(doc)
            score = 0.0
            for tok in query:
                if tok not in tf:
                    continue
                idf = self.idf.get(tok, math.log(1 + (self.n_docs + 0.5) / 1.5))
                f = tf[tok]
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (f * (self.k1 + 1)) / max(denom, 1e-9)
            return score

        fwd = one_way(tokens_a, tokens_b)
        bwd = one_way(tokens_b, tokens_a)
        return 0.5 * (fwd + bwd)


def cosine_bow(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Plain cosine similarity over raw bag-of-words token counts (no IDF weighting) —
    used as the second component of the blended text score, matching the cosine+BM25
    blend pattern independently converged upon by more than one prior system."""
    if not tokens_a or not tokens_b:
        return 0.0
    ca, cb = Counter(tokens_a), Counter(tokens_b)
    shared = set(ca) & set(cb)
    dot = sum(ca[t] * cb[t] for t in shared)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
