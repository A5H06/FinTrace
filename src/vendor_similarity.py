"""Local text similarity and secure bank-account comparison helpers.

Character n-gram hashing is deterministic and avoids training a separate embedding
vocabulary. It is useful for near-match screening, not proof that two vendors are
identical. Bank-account identifiers are compared exactly after normalization;
cosine similarity is never used to declare two account numbers equivalent.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

EMBEDDING_DIMENSIONS = 512


def _vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char", ngram_range=(2, 4), n_features=EMBEDDING_DIMENSIONS,
        alternate_sign=False, norm="l2", lowercase=True, strip_accents="unicode",
    )


def normalize_bank_identifier(value: object) -> str:
    """Normalize a bank-account identifier for exact matching, without storing it."""
    if value is None:
        return ""
    raw = str(value).strip()
    if raw.casefold() in {"", "nan", "none", "<na>", "null", "nat"}:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", raw).casefold()


def bank_identifiers_match(expected: object, submitted: object) -> bool | None:
    """Return equality for two supplied identifiers, or None if either is absent."""
    left = normalize_bank_identifier(expected)
    right = normalize_bank_identifier(submitted)
    if not left or not right:
        return None
    return left == right


def pairwise_text_similarity(left_values: Iterable[object], right_values: Iterable[object]) -> np.ndarray:
    """Compute row-wise character n-gram cosine similarity for text pairs.

    A missing side returns NaN rather than pretending that absent data is a match.
    """
    left = ["" if item is None else str(item).strip() for item in left_values]
    right = ["" if item is None else str(item).strip() for item in right_values]
    if len(left) != len(right):
        raise ValueError("Text pair collections must have equal lengths")
    result = np.full(len(left), np.nan, dtype=float)
    active = [i for i, (a, b) in enumerate(zip(left, right)) if a and b]
    if not active:
        return result
    vectorizer = _vectorizer()
    left_matrix = vectorizer.transform([left[i] for i in active])
    right_matrix = vectorizer.transform([right[i] for i in active])
    similarities = np.asarray(left_matrix.multiply(right_matrix).sum(axis=1)).reshape(-1)
    for index, similarity in zip(active, similarities):
        result[index] = float(np.clip(similarity, 0.0, 1.0))
    return result


def cosine_similarity_text(left: object, right: object) -> float | None:
    """Compute local character n-gram cosine similarity for one pair."""
    values = pairwise_text_similarity([left], [right])
    return None if np.isnan(values[0]) else float(values[0])


def make_text_embedding(text: str) -> list[float]:
    """Create a fixed-dimension normalized vector that can be stored in pgvector."""
    matrix = _vectorizer().transform([str(text or "")])
    vector = np.asarray(matrix.toarray()[0], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(float).tolist()
