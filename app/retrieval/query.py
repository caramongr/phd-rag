"""Conservative query preprocessing (RAG-009).

Allowed: trim, Unicode-normalize, collapse whitespace. Nothing else - no LLM
rewriting, no synonym expansion, no HyDE, no decomposition. RAG-009 measures what
plain dense retrieval plus a reranker achieves *before* any of that is added, so
query handling must stay auditable.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Return the canonical query string used for embedding and reranking.

    NFKC normalization folds compatibility characters (full-width digits, ligatures)
    without touching meaning, and repeated spaces/newlines from a pasted query
    collapse to single spaces. Case is preserved on purpose: the models are
    case-aware and lowercasing would silently change the input.
    """
    if text is None:
        raise ValueError("query must be a string")
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u00a0", " ")
    return _WHITESPACE.sub(" ", normalized).strip()
