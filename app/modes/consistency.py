"""Deterministic comparison of the author's work with the literature (RAG-013 §4).

CONSISTENCY mode answers a question RAG-011 and RAG-012 cannot: *does what I already wrote
agree with the literature and with my own rules?* It does that with measurements, not with a
model, because the author's own work is not evidence and a model asked to "check for
contradictions" would be inventing findings about the author's text.

Five checks, each explainable from its inputs:

* **terminology** — content terms the draft uses that the retrieved literature does not (and
  the reverse);
* **duplication** — passages of the draft that repeat each other, or that are largely
  contained in a retrieved paper (a close paraphrase to check);
* **unsupported claim** — a substantive sentence of the draft that no retrieved literature
  passage covers;
* **numeric discrepancy** — a figure in the draft where the best-matching literature passage
  states figures of the same kind but not that value;
* **style** — the author's own writing preferences (no em dash, short varied sentences, no
  promotional language) applied to the draft.

Every finding records the method that produced it and, where one exists, the chunk it rests
on. ``my_work`` is never cited as evidence: findings about the draft name the draft, and
findings about the literature name the literature chunk.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from app.citations.claims import (
    build_numeric_index,
    content_terms,
    content_tokens,
    extract_numeric_mentions,
    source_sentences,
)
from app.generation.grounding import split_sentences
from app.modes.models import (
    EM_DASH,
    LONG_SENTENCE_WORDS,
    PROMOTIONAL_TERMS,
    ConsistencyFinding,
    ConsistencyFindingKind,
    WRITING_PREFERENCES,
)

#: Provenance recorded when a style finding rests on RAG-013's preference list rather than
#: on a retrieved rules chunk.
BUILTIN_PREFERENCE_SOURCE = "rag013:writing_preferences"


@dataclass(frozen=True)
class ComparisonPolicy:
    """The thresholds behind every CONSISTENCY finding."""

    #: How much of the shorter passage must occur in the longer one to look duplicated.
    duplication_overlap: float = 0.60
    #: Passages this short are not compared: overlap between few terms means nothing.
    duplication_min_terms: int = 8
    #: A draft sentence whose best literature overlap is below this is unsupported.
    unsupported_max_coverage: float = 0.30
    #: Draft sentences this short are skipped in the unsupported-claim check.
    unsupported_min_terms: int = 6
    #: How well a literature passage must match before its figures are compared.
    discrepancy_min_overlap: float = 0.35
    #: A term must occur at least this often to be reported as a terminology difference.
    terminology_min_count: int = 2
    #: At most this many terms are listed per terminology finding.
    terminology_top: int = 8
    #: At most this many findings are produced per kind.
    max_findings_per_kind: int = 12


def _excerpt(text: str, width: int = 260) -> str:
    """A single-line excerpt, cut at a word boundary so it never ends mid-word."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= width:
        return collapsed
    return collapsed[:width].rsplit(" ", 1)[0] + " ..."


def _window(text: str, needle: str, *, width: int = 120) -> str:
    """An excerpt centred on ``needle``, so a character-level finding shows its evidence."""
    collapsed = " ".join((text or "").split())
    index = collapsed.find(needle)
    if index < 0:
        return _excerpt(collapsed, width * 2)
    start = max(0, index - width)
    end = min(len(collapsed), index + len(needle) + width)
    prefix = "..." if start else ""
    suffix = "..." if end < len(collapsed) else ""
    return f"{prefix}{collapsed[start:end]}{suffix}"


def _containment(smaller: frozenset[str], larger: frozenset[str]) -> float:
    """How much of ``smaller`` occurs in ``larger`` (0.0 when ``smaller`` is empty)."""
    if not smaller:
        return 0.0
    return len(smaller & larger) / len(smaller)


def _capped(
    findings: list[ConsistencyFinding], policy: ComparisonPolicy
) -> tuple[ConsistencyFinding, ...]:
    """Bound the findings of one kind, and say so when the bound bites.

    A truncated list that does not admit it reads as a complete list, which is exactly the
    impression a comparison report must not give.
    """
    cap = policy.max_findings_per_kind
    if len(findings) <= cap:
        return tuple(findings)
    kept = list(findings[:cap])
    note = f"showing {cap} of {len(findings)} findings of this kind"
    last = kept[-1]
    kept[-1] = last.model_copy(
        update={"detail": f"{last.detail} ({note})".strip()}
    )
    return tuple(kept)


# --- individual checks --------------------------------------------------------


def _term_counts(chunks: Sequence[Any]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    """Canonical-term counts plus, per term, the surface words it was built from.

    A term counts once per chunk however often it occurs in it, so ``terminology_min_count``
    keeps meaning "repeated across the retrieved passages" rather than "repeated in one".
    """
    counts: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = {}
    for chunk in chunks:
        seen: dict[str, Counter[str]] = {}
        for canonical, surface in content_tokens(getattr(chunk, "text", "") or ""):
            seen.setdefault(canonical, Counter())[surface] += 1
        for canonical, words in seen.items():
            counts[canonical] += 1
            surfaces.setdefault(canonical, Counter()).update(words)
    return counts, surfaces


def _surface(surfaces: Counter[str]) -> str:
    """The most frequent surface word for a term, so reports show words, not stems."""
    return sorted(surfaces.items(), key=lambda item: (-item[1], item[0]))[0][0]


def terminology_findings(
    your_chunks: Sequence[Any],
    literature: Sequence[Any],
    policy: ComparisonPolicy,
) -> tuple[ConsistencyFinding, ...]:
    """Terms the draft uses that the literature does not, and the reverse."""
    if not your_chunks or not literature:
        return ()
    yours, your_surfaces = _term_counts(your_chunks)
    theirs, their_surfaces = _term_counts(literature)
    only_yours = [
        term
        for term, count in yours.most_common()
        if term not in theirs and count >= policy.terminology_min_count
    ][: policy.terminology_top]
    only_theirs = [
        term
        for term, count in theirs.most_common()
        if term not in yours and count >= policy.terminology_min_count
    ][: policy.terminology_top]
    your_words = ", ".join(_surface(your_surfaces[term]) for term in only_yours)
    their_words = ", ".join(_surface(their_surfaces[term]) for term in only_theirs)
    evidence = tuple(
        str(getattr(chunk, "chunk_id", "")) for chunk in literature if getattr(chunk, "chunk_id", None)
    )
    findings: list[ConsistencyFinding] = []
    if only_yours:
        findings.append(
            ConsistencyFinding(
                kind=ConsistencyFindingKind.TERMINOLOGY,
                summary=(
                    f"your draft uses {len(only_yours)} term(s) that the retrieved literature "
                    "does not: " + your_words
                ),
                detail=(
                    "terms repeated in your work but absent from the retrieved literature; "
                    "check whether they are your own coinage, an established term spelled "
                    "differently, or a gap in the retrieval"
                ),
                evidence=evidence,
                method="content_term_frequency",
            )
        )
    if only_theirs:
        findings.append(
            ConsistencyFinding(
                kind=ConsistencyFindingKind.TERMINOLOGY,
                summary=(
                    f"the retrieved literature uses {len(only_theirs)} term(s) your draft does "
                    "not: " + their_words
                ),
                detail=(
                    "terms repeated in the literature but absent from your work; check whether "
                    "the literature's own vocabulary should appear in your text"
                ),
                evidence=evidence,
                method="content_term_frequency",
            )
        )
    return _capped(findings, policy)


def duplication_findings(
    your_chunks: Sequence[Any],
    literature: Sequence[Any],
    policy: ComparisonPolicy,
) -> tuple[ConsistencyFinding, ...]:
    """Passages of the draft that repeat each other or a retrieved paper."""
    findings: list[ConsistencyFinding] = []
    terms = [content_terms(getattr(chunk, "text", "") or "") for chunk in your_chunks]
    for left in range(len(your_chunks)):
        for right in range(left + 1, len(your_chunks)):
            if min(len(terms[left]), len(terms[right])) < policy.duplication_min_terms:
                continue
            smaller, larger = sorted((terms[left], terms[right]), key=len)
            overlap = _containment(smaller, larger)
            if overlap < policy.duplication_overlap:
                continue
            findings.append(
                ConsistencyFinding(
                    kind=ConsistencyFindingKind.DUPLICATION,
                    summary=(
                        "two passages of your own work cover the same material "
                        f"({overlap:.0%} of the shorter passage)"
                    ),
                    detail="possible duplicated idea or near-duplicate section in your draft",
                    your_text=_excerpt(your_chunks[left].text),
                    other_text=_excerpt(your_chunks[right].text),
                    your_chunk_id=getattr(your_chunks[left], "chunk_id", None),
                    other_chunk_id=getattr(your_chunks[right], "chunk_id", None),
                    method="content_term_containment",
                )
            )
    for index, chunk in enumerate(your_chunks):
        if len(terms[index]) < policy.duplication_min_terms:
            continue
        best_overlap, best_chunk = 0.0, None
        for candidate in literature:
            candidate_terms = content_terms(getattr(candidate, "text", "") or "")
            overlap = _containment(terms[index], candidate_terms)
            if overlap > best_overlap:
                best_overlap, best_chunk = overlap, candidate
        if best_chunk is None or best_overlap < policy.duplication_overlap:
            continue
        findings.append(
            ConsistencyFinding(
                kind=ConsistencyFindingKind.DUPLICATION,
                summary=(
                    f"{best_overlap:.0%} of one of your passages also occurs in "
                    f"{getattr(best_chunk, 'title', None) or 'a retrieved paper'}"
                ),
                detail=(
                    "close lexical overlap with a retrieved paper: check whether this is your "
                    "own summary, an unmarked paraphrase, or a duplicated quotation"
                ),
                your_text=_excerpt(chunk.text),
                other_text=_excerpt(getattr(best_chunk, "text", "")),
                your_chunk_id=getattr(chunk, "chunk_id", None),
                other_chunk_id=getattr(best_chunk, "chunk_id", None),
                evidence=(str(getattr(best_chunk, "chunk_id", "")),),
                method="content_term_containment",
            )
        )
    return _capped(findings, policy)


def unsupported_claim_findings(
    your_chunks: Sequence[Any],
    literature: Sequence[Any],
    policy: ComparisonPolicy,
) -> tuple[ConsistencyFinding, ...]:
    """Substantive draft sentences that no retrieved literature passage covers."""
    if not literature:
        return ()
    literature_sentences = [
        sentence
        for chunk in literature
        for sentence in source_sentences(getattr(chunk, "text", "") or "")
    ]
    if not literature_sentences:
        return ()
    evidence = tuple(
        str(getattr(chunk, "chunk_id", ""))
        for chunk in literature
        if getattr(chunk, "chunk_id", None)
    )
    findings: list[ConsistencyFinding] = []
    for chunk in your_chunks:
        for sentence in split_sentences(getattr(chunk, "text", "") or ""):
            terms = content_terms(sentence)
            if len(terms) < policy.unsupported_min_terms:
                continue
            best = max(
                (_containment(terms, content_terms(candidate)) for candidate in literature_sentences),
                default=0.0,
            )
            if best >= policy.unsupported_max_coverage:
                continue
            findings.append(
                ConsistencyFinding(
                    kind=ConsistencyFindingKind.UNSUPPORTED_CLAIM,
                    summary=(
                        "no retrieved literature passage covers this sentence of your draft "
                        f"(best overlap {best:.0%})"
                    ),
                    detail=(
                        f"checked against {len(literature)} retrieved literature chunk(s) and "
                        f"{len(literature_sentences)} of their sentences; this means no match "
                        "was found in what was retrieved, not that the claim is wrong"
                    ),
                    your_text=_excerpt(sentence, 300),
                    your_chunk_id=getattr(chunk, "chunk_id", None),
                    evidence=evidence,
                    method="content_term_containment",
                )
            )
    return _capped(findings, policy)


def numeric_discrepancy_findings(
    your_chunks: Sequence[Any],
    literature: Sequence[Any],
    policy: ComparisonPolicy,
) -> tuple[ConsistencyFinding, ...]:
    """Draft figures where the matching literature passage states different values."""
    if not literature:
        return ()
    literature_sentences = [
        sentence
        for chunk in literature
        for sentence in source_sentences(getattr(chunk, "text", "") or "")
    ]
    findings: list[ConsistencyFinding] = []
    for chunk in your_chunks:
        for sentence in split_sentences(getattr(chunk, "text", "") or ""):
            mentions = extract_numeric_mentions(sentence)
            if not mentions:
                continue
            terms = content_terms(sentence)
            best_sentence, best_overlap, best_index = None, 0.0, None
            for candidate in literature_sentences:
                overlap = _containment(terms, content_terms(candidate))
                if overlap > best_overlap:
                    best_sentence, best_overlap = candidate, overlap
                    best_index = build_numeric_index(candidate)
            if best_overlap < policy.discrepancy_min_overlap or best_index is None:
                continue
            others = best_index.all_values()
            if not others:
                continue
            mismatched = [
                mention
                for mention in mentions
                if mention.value is not None
                and mention.kind not in ("year",)
                and not any(abs(mention.value - value) <= 1e-6 for value in others)
            ]
            if not mismatched:
                continue
            findings.append(
                ConsistencyFinding(
                    kind=ConsistencyFindingKind.NUMERIC_DISCREPANCY,
                    summary=(
                        "your figure(s) do not appear in the closest literature passage, which "
                        "states other figures of the same kind: "
                        + ", ".join(mention.display for mention in mismatched)
                    ),
                    detail=(
                        f"closest passage overlaps this sentence by {best_overlap:.0%}; "
                        "figures present there: "
                        + ", ".join(sorted({f"{value:g}" for value in others}))
                    ),
                    your_text=_excerpt(sentence, 300),
                    other_text=_excerpt(best_sentence or "", 300),
                    your_chunk_id=getattr(chunk, "chunk_id", None),
                    method="numeric_value_comparison",
                )
            )
    return _capped(findings, policy)


def style_findings(
    your_chunks: Sequence[Any],
    rules: Sequence[Any],
    policy: ComparisonPolicy,
) -> tuple[ConsistencyFinding, ...]:
    """The author's writing preferences applied to the draft (§3, §4)."""
    rule_evidence = tuple(
        str(getattr(chunk, "chunk_id", ""))
        for chunk in rules
        if getattr(chunk, "chunk_id", None)
    ) or (BUILTIN_PREFERENCE_SOURCE,)
    findings: list[ConsistencyFinding] = []
    for chunk in your_chunks:
        text = getattr(chunk, "text", "") or ""
        chunk_id = getattr(chunk, "chunk_id", None)
        if EM_DASH in text:
            findings.append(
                ConsistencyFinding(
                    kind=ConsistencyFindingKind.STYLE,
                    summary="your draft uses an em dash, which the writing preferences rule out",
                    detail=f"preference: {WRITING_PREFERENCES[3]!r}",
                    your_text=_window(text, EM_DASH),
                    your_chunk_id=chunk_id,
                    evidence=rule_evidence,
                    method="character_check",
                )
            )
        lowered = text.casefold()
        promotional = sorted(
            {term for term in PROMOTIONAL_TERMS if term in lowered}
        )
        if promotional:
            findings.append(
                ConsistencyFinding(
                    kind=ConsistencyFindingKind.STYLE,
                    summary="promotional language in your draft: " + ", ".join(promotional),
                    detail=f"preference: {WRITING_PREFERENCES[6]!r}",
                    your_text=_window(text, promotional[0]),
                    your_chunk_id=chunk_id,
                    evidence=rule_evidence,
                    method="term_list_check",
                )
            )
        long_sentences = [
            sentence
            for sentence in split_sentences(text)
            if len(sentence.split()) > LONG_SENTENCE_WORDS
        ]
        if long_sentences:
            longest = max(long_sentences, key=lambda sentence: len(sentence.split()))
            longest_words = len(longest.split())
            findings.append(
                ConsistencyFinding(
                    kind=ConsistencyFindingKind.STYLE,
                    summary=(
                        f"{len(long_sentences)} sentence(s) longer than {LONG_SENTENCE_WORDS} "
                        f"words in your draft (longest {longest_words} words)"
                    ),
                    detail=f"preference: {WRITING_PREFERENCES[1]!r}",
                    your_text=_excerpt(longest, 300),
                    your_chunk_id=chunk_id,
                    evidence=rule_evidence,
                    method="sentence_length_check",
                )
            )
    return _capped(findings, policy)


# --- the whole comparison ------------------------------------------------------


def compare(
    your_chunks: Sequence[Any],
    *,
    literature: Sequence[Any] = (),
    rules: Sequence[Any] = (),
    policy: ComparisonPolicy | None = None,
) -> tuple[ConsistencyFinding, ...]:
    """Every CONSISTENCY finding, in a fixed order (terminology, duplication, ...)."""
    policy = policy or ComparisonPolicy()
    findings: list[ConsistencyFinding] = []
    findings.extend(terminology_findings(your_chunks, literature, policy))
    findings.extend(duplication_findings(your_chunks, literature, policy))
    findings.extend(unsupported_claim_findings(your_chunks, literature, policy))
    findings.extend(numeric_discrepancy_findings(your_chunks, literature, policy))
    findings.extend(style_findings(your_chunks, rules, policy))
    return tuple(findings)


def finding_counts(findings: Sequence[ConsistencyFinding]) -> dict[str, int]:
    """How many findings of each kind, for the summary line."""
    counts: Counter[str] = Counter(finding.kind.value for finding in findings)
    return dict(sorted(counts.items()))


def render_report(
    query: str,
    findings: Sequence[ConsistencyFinding],
    *,
    literature_count: int,
    rules_count: int,
    your_count: int,
) -> str:
    """A deterministic, human-readable comparison report (no model involved)."""
    lines = [
        "CONSISTENCY COMPARISON (deterministic: lexical overlap, figures and the writing "
        "preferences; no model was used)",
        f"query: {query}",
        (
            f"input: {your_count} chunk(s) of your own work, "
            f"{literature_count} literature chunk(s), {rules_count} rules/guidance chunk(s)"
        ),
        "",
    ]
    if not findings:
        lines.append(
            "No differences were found in what was retrieved. Repeat with a broader query "
            "before drawing a conclusion."
        )
        return "\n".join(lines)
    by_kind: dict[str, list[ConsistencyFinding]] = {}
    for finding in findings:
        by_kind.setdefault(finding.kind.value, []).append(finding)
    for kind in (
        ConsistencyFindingKind.TERMINOLOGY.value,
        ConsistencyFindingKind.DUPLICATION.value,
        ConsistencyFindingKind.UNSUPPORTED_CLAIM.value,
        ConsistencyFindingKind.NUMERIC_DISCREPANCY.value,
        ConsistencyFindingKind.STYLE.value,
    ):
        items = by_kind.get(kind)
        if not items:
            continue
        lines.append(f"{kind.upper()} ({len(items)})")
        for index, finding in enumerate(items, start=1):
            lines.append(f"  {index}. {finding.summary}")
            if finding.detail:
                lines.append(f"     {finding.detail}")
            if finding.your_text:
                lines.append(f"     your text : {finding.your_text}")
            if finding.your_chunk_id:
                lines.append(f"     your chunk: {finding.your_chunk_id}")
            if finding.other_text:
                lines.append(f"     literature: {finding.other_text}")
            if finding.other_chunk_id:
                lines.append(f"     lit. chunk: {finding.other_chunk_id}")
            if finding.method:
                lines.append(f"     method    : {finding.method}")
        lines.append("")
    return "\n".join(lines).rstrip()


__all__ = [
    "BUILTIN_PREFERENCE_SOURCE",
    "ComparisonPolicy",
    "compare",
    "duplication_findings",
    "finding_counts",
    "numeric_discrepancy_findings",
    "render_report",
    "style_findings",
    "terminology_findings",
    "unsupported_claim_findings",
]
