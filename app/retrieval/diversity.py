"""Document diversity and near-duplicate suppression (RAG-009).

Two different problems, two different deterministic rules:

* **near-duplicates** - the same passage reaching the result twice. Two signals:
  (a) an identical normalised text hash, which is suppressed *regardless of
  document id* because the corpus really does contain several physical copies of
  the same published work (different files, different SHA-256, identical extracted
  text for that passage); (b) the same document with adjacent ``chunk_index`` and
  very high token overlap, which is what overlapping RAG-007 chunks look like.
* **work diversity** - eight chunks all from one paper crowd out the rest of the
  corpus, and the corpus contains several copies of some papers under different
  filenames, so the cap is applied per **work** (a distinctive title), not per
  physical document. Capped at ``max_chunks_per_document`` (default 3), which is a
  cap, not a quota: one chunk per work is never forced.

Both operate on an already ranked list and always keep the higher-ranked member,
so ordering is preserved and the outcome is deterministic. Nothing here uses a
model, a threshold learned from data, or randomness.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Sequence

from app.retrieval.models import Candidate, DroppedCandidate

#: Share of the smaller chunk's tokens that must appear in the other chunk.
DEFAULT_OVERLAP_THRESHOLD = 0.85
#: Only chunks this close together in one document can be overlapping neighbours.
DEFAULT_MAX_INDEX_DISTANCE = 1

REASON_DUPLICATE_TEXT = "identical_text"
REASON_DUPLICATE_NEIGHBOUR = "overlapping_neighbour"
REASON_DOCUMENT_LIMIT = "document_limit"

#: A title only identifies a *work* when it is distinctive enough: short generic
#: titles ("Form", "Book Reviews", "PRIVACY POLICY") are shared by unrelated
#: documents, so those fall back to the document id.
MIN_WORK_TITLE_TOKENS = 3
MIN_WORK_TITLE_CHARS = 15

_TOKEN = re.compile(r"[0-9a-z]+")


def normalized_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens of ``text`` (deterministic, no stemming).

    NFKC first, so a ligature inside a title (``Artiﬁcial``) yields the same tokens
    as the plain spelling (``Artificial``) - which is what makes two copies of one
    paper recognisable as the same work.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN.findall(folded)


def normalized_text_hash(text: str) -> str:
    """Stable hash of the normalised text, for exact-duplicate detection."""
    joined = " ".join(normalized_tokens(text))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def token_overlap(left: str, right: str) -> float:
    """Containment of the smaller token set in the larger one (0.0 - 1.0)."""
    left_tokens = set(normalized_tokens(left))
    right_tokens = set(normalized_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    smaller = left_tokens if len(left_tokens) <= len(right_tokens) else right_tokens
    larger = right_tokens if smaller is left_tokens else left_tokens
    return len(smaller & larger) / len(smaller)


def suppress_near_duplicates(
    candidates: Sequence[Candidate],
    *,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    max_index_distance: int = DEFAULT_MAX_INDEX_DISTANCE,
    suppress_identical_across_documents: bool = True,
) -> tuple[list[Candidate], list[DroppedCandidate]]:
    """Drop near-duplicate chunks, keeping the first (best-ranked) of each pair.

    Identical text is suppressed across documents as well (copies of one work),
    unless the caller explicitly turns that off; the adjacency rule stays strictly
    inside one document.
    """
    kept: list[Candidate] = []
    kept_hashes: list[str] = []
    dropped: list[DroppedCandidate] = []
    for candidate in candidates:
        text = candidate.text or candidate.retrieval_text
        text_hash = normalized_text_hash(text)
        duplicate_of: Candidate | None = None
        detail: str | None = None
        reason: str | None = None
        for index, previous in enumerate(kept):
            previous_hash = kept_hashes[index]
            if previous_hash == text_hash and (
                suppress_identical_across_documents
                or previous.document_id == candidate.document_id
            ):
                duplicate_of = previous
                reason = REASON_DUPLICATE_TEXT
                detail = "same normalised text"
                if previous.document_id != candidate.document_id:
                    detail += f" as {previous.chunk_id} in another copy of the work"
                break
            if previous.document_id != candidate.document_id:
                continue
            if (
                previous.chunk_index is not None
                and candidate.chunk_index is not None
                and abs(previous.chunk_index - candidate.chunk_index)
                <= max_index_distance
            ):
                overlap = token_overlap(
                    previous.text or previous.retrieval_text, text
                )
                if overlap >= overlap_threshold:
                    duplicate_of = previous
                    reason = REASON_DUPLICATE_NEIGHBOUR
                    detail = f"token overlap {overlap:.2f} with adjacent chunk"
                    break
        if duplicate_of is not None:
            dropped.append(
                DroppedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    reason=reason or REASON_DUPLICATE_TEXT,
                    detail=f"{detail}; kept {duplicate_of.chunk_id}",
                    dense_rank=candidate.dense_rank,
                    rerank_score=candidate.rerank_score,
                )
            )
            continue
        kept.append(candidate)
        kept_hashes.append(text_hash)
    return kept, dropped


def apply_document_limit(
    candidates: Sequence[Candidate],
    *,
    max_chunks_per_document: int | None = 3,
    group_by_work: bool = True,
) -> tuple[list[Candidate], list[DroppedCandidate]]:
    """Cap how many chunks one work may contribute to the result list.

    The cap applies to a **work**, not to a physical file: the corpus holds several
    copies of the same published paper under different filenames (and therefore
    different SHA-256 hashes and document ids), so capping per document id alone let
    one paper fill the whole result list. With ``group_by_work`` a distinctive title
    is the work key; documents without one fall back to their document id.
    """
    if not max_chunks_per_document or max_chunks_per_document <= 0:
        return list(candidates), []
    counts: dict[str, int] = {}
    kept: list[Candidate] = []
    dropped: list[DroppedCandidate] = []
    for candidate in candidates:
        key = work_key(candidate) if group_by_work else f"document:{candidate.document_id}"
        seen = counts.get(key, 0)
        if seen >= max_chunks_per_document:
            dropped.append(
                DroppedCandidate(
                    chunk_id=candidate.chunk_id,
                    document_id=candidate.document_id,
                    reason=REASON_DOCUMENT_LIMIT,
                    detail=(
                        f"{key} already contributed {seen} chunks "
                        f"(limit {max_chunks_per_document})"
                    ),
                    dense_rank=candidate.dense_rank,
                    rerank_score=candidate.rerank_score,
                )
            )
            continue
        counts[key] = seen + 1
        kept.append(candidate)
    return kept, dropped


def work_key(candidate: Candidate) -> str:
    """Deterministic identity of the *work* a candidate comes from.

    A distinctive title is used when one exists, so copies of the same paper share a
    key; otherwise the document id keeps unrelated untitled documents apart.
    """
    tokens = normalized_tokens(candidate.title or "")
    joined = " ".join(tokens)
    if len(tokens) >= MIN_WORK_TITLE_TOKENS and len(joined) >= MIN_WORK_TITLE_CHARS:
        return f"title:{joined}"
    return f"document:{candidate.document_id}"


def dropped_reasons(dropped: Iterable[DroppedCandidate]) -> dict[str, int]:
    """Count drops per reason (audit/report helper)."""
    counts: dict[str, int] = {}
    for item in dropped:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return counts
