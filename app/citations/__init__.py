"""Claim validation and Harvard citation output (RAG-012).

RAG-011 answers a question and marks every claim sentence with an internal ``[S1]``
marker. RAG-012 does not generate anything: it *checks* those claims against the sources
they cite, and turns the markers into Harvard citations with a reference list.

```python
from app.citations import CitationService

service = CitationService()                      # RAG-011 generation + deterministic checks
response, result = service.answer_and_validate("What is AI governance in higher education?")
print(result.validated_answer)
print("\n".join(result.reference_list))
print(result.validation_summary.supported, result.validation_summary.unsupported)
```

The default validator is deterministic: no language model is consulted unless a caller
passes ``semantic=True`` (or an explicit checker), and even then the checks in
:mod:`app.citations.support` run first and are never overridden. Claims that fail
validation are withheld from ``validated_answer``, not published; partial and uncertain
claims are annotated. See :data:`app.citations.service.POLICY_NOTE`.
"""

from __future__ import annotations

from app.citations.claims import (
    best_sentence_coverage,
    build_numeric_index,
    classify_sentence,
    content_terms,
    coverage,
    extract_numeric_mentions,
    extract_quotations,
    marker_groups,
    match_numeric,
    sentence_spans,
    split_sentences,
    strip_markers,
)
from app.citations.harvard import (
    author_label,
    build_reference,
    build_reference_list,
    citation_for_ids,
    citation_map,
    format_reference,
    reference_identity,
    split_author_string,
)
from app.citations.models import (
    ClaimDecision,
    ClaimKind,
    ClaimValidation,
    CitationMapEntry,
    CitationValidationResult,
    NumericKind,
    NumericMention,
    QuotationCheck,
    ReferenceEntry,
    SourceSupport,
    SupportVerdict,
    ValidationPolicy,
    ValidationSummary,
    ValidatorMethod,
    worst_verdict,
)
from app.citations.service import POLICY_NOTE, CitationService
from app.citations.support import (
    EmbeddingEntailmentChecker,
    EntailmentChecker,
    EntailmentResult,
    LexicalEntailmentChecker,
    lexical_result,
    validate_answer,
    validate_claim,
)

__all__ = [
    "ClaimDecision",
    "ClaimKind",
    "ClaimValidation",
    "CitationMapEntry",
    "CitationService",
    "CitationValidationResult",
    "EmbeddingEntailmentChecker",
    "EntailmentChecker",
    "EntailmentResult",
    "LexicalEntailmentChecker",
    "NumericKind",
    "NumericMention",
    "POLICY_NOTE",
    "QuotationCheck",
    "ReferenceEntry",
    "SourceSupport",
    "SupportVerdict",
    "ValidationPolicy",
    "ValidationSummary",
    "ValidatorMethod",
    "author_label",
    "best_sentence_coverage",
    "build_numeric_index",
    "build_reference",
    "build_reference_list",
    "citation_for_ids",
    "citation_map",
    "classify_sentence",
    "content_terms",
    "coverage",
    "extract_numeric_mentions",
    "extract_quotations",
    "format_reference",
    "lexical_result",
    "marker_groups",
    "match_numeric",
    "reference_identity",
    "sentence_spans",
    "split_author_string",
    "split_sentences",
    "strip_markers",
    "validate_answer",
    "validate_claim",
    "worst_verdict",
]
