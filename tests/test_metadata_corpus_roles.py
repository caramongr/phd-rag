"""RAG-006 tests 23-25: corpus-role gating of external enrichment."""

from __future__ import annotations

import pytest

from metadata_support import FakeTransport, build_test_enricher, crossref_doi_handler, crossref_message

from app.metadata.corpus import corpus_role_for_path, is_enrichable
from app.metadata.enums import Confidence, CorpusRole, MetadataSource
from app.metadata.models import AcademicMetadata, MetadataValue


def _draft(*, role: CorpusRole, doi: str | None = None) -> AcademicMetadata:
    candidates: dict = {}
    doi_value = None
    if doi is not None:
        doi_value = MetadataValue(
            value=doi,
            source=MetadataSource.DOCUMENT_TEXT,
            confidence=Confidence.HIGH,
            evidence="DOI in draft",
        )
        candidates["doi"] = [doi_value]
    return AcademicMetadata(
        filename="draft.docx",
        source_path="draft.docx",
        extension=".docx",
        corpus_role=role,
        doi=doi_value,
        candidates=candidates,
    )


@pytest.mark.parametrize("role", [CorpusRole.RULES, CorpusRole.SUPERVISION, CorpusRole.FUTURE_CHAPTERS, CorpusRole.SUMMARIES, CorpusRole.GUIDANCE])
def test_non_literature_roles_never_queried(role: CorpusRole) -> None:
    transport = FakeTransport(crossref_doi_handler(crossref_message()))
    enricher = build_test_enricher(transport, crossref=True, openalex=True)

    result = enricher.enrich(_draft(role=role, doi="10.1000/xyz"))

    assert transport.calls == []
    assert result.external_match_status is None


def test_my_work_draft_not_enriched_by_default() -> None:
    transport = FakeTransport(crossref_doi_handler(crossref_message()))
    enricher = build_test_enricher(transport, crossref=True, openalex=True)

    result = enricher.enrich(_draft(role=CorpusRole.MY_WORK))

    assert transport.calls == []
    assert result.external_match_status is None


def test_my_work_with_doi_may_be_enriched() -> None:
    transport = FakeTransport(crossref_doi_handler(crossref_message(doi="10.1000/xyz")))
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    result = enricher.enrich(_draft(role=CorpusRole.MY_WORK, doi="10.1000/xyz"))

    assert len(transport.calls) == 1
    assert result.external_match_status.value == "EXACT"


def test_rules_file_does_not_get_crossref_lookup() -> None:
    transport = FakeTransport(crossref_doi_handler(crossref_message()))
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    enricher.enrich(_draft(role=CorpusRole.RULES, doi="10.1000/xyz"))

    assert transport.calls == []


def test_future_chapters_file_does_not_get_crossref_lookup() -> None:
    transport = FakeTransport(crossref_doi_handler(crossref_message()))
    enricher = build_test_enricher(transport, crossref=True, openalex=False)

    enricher.enrich(_draft(role=CorpusRole.FUTURE_CHAPTERS, doi="10.1000/xyz"))

    assert transport.calls == []


@pytest.mark.parametrize(
    "path, expected",
    [
        (r"F:\phd-rag\data\literature\AI\paper.pdf", CorpusRole.LITERATURE),
        (r"F:\phd-rag\data\my_work\thesis\draft.docx", CorpusRole.MY_WORK),
        (r"F:\phd-rag\data\rules\style.md", CorpusRole.RULES),
        (r"F:\phd-rag\data\supervision\notes.docx", CorpusRole.SUPERVISION),
        (r"F:\phd-rag\data\future_chapters\next.md", CorpusRole.FUTURE_CHAPTERS),
        (r"F:\phd-rag\data\summaries\s.md", CorpusRole.SUMMARIES),
        (r"F:\phd-rag\data\guidance\g.md", CorpusRole.GUIDANCE),
    ],
)
def test_corpus_role_for_path(path: str, expected: CorpusRole) -> None:
    assert corpus_role_for_path(path) is expected


def test_is_enrichable_rules() -> None:
    assert is_enrichable(CorpusRole.LITERATURE, has_doi=False)
    assert not is_enrichable(CorpusRole.MY_WORK, has_doi=False)
    assert is_enrichable(CorpusRole.MY_WORK, has_doi=True)
    assert not is_enrichable(CorpusRole.RULES, has_doi=True)
