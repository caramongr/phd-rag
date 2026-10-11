"""RAG-007 tests 19-27: reference sections, corpus roles and evidence policy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from chunking_support import build_document, heading, paragraph, table

from app.chunking.chunker import chunk_document
from app.chunking.models import ChunkingOptions
from app.chunking.policy import ROLE_POLICIES, eligibility_for, policy_for_role
from app.metadata.enums import CorpusRole
from app.models.enums import AuthorityLevel, ContentType

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)
OPTIONS = ChunkingOptions(created_at=FIXED)

_REFERENCE_ENTRY = (
    "Smith, J. (2022). Immersive virtual reality in higher education. "
    "Journal of Educational Technology, 14(2), 100-118."
)


def test_references_section_is_tagged_references() -> None:
    document = build_document(
        [
            heading("References", 1),
            paragraph(f"{_REFERENCE_ENTRY}\n\nLee, K. (2021). Another study. Press.", path=("References",)),
        ]
    )

    result = chunk_document(document, options=OPTIONS)

    assert len(result.chunks) == 1
    assert result.chunks[0].content_type is ContentType.REFERENCES
    assert result.chunks[0].heading_path == ["References"]


def test_references_chunks_are_not_retrieval_eligible_by_default() -> None:
    document = build_document(
        [
            heading("References", 1),
            paragraph(_REFERENCE_ENTRY, path=("References",)),
            heading("Bibliography", 1),
            paragraph("Bell, G. (2020). Another entry. Publisher.", path=("Bibliography",)),
        ]
    )

    result = chunk_document(document, options=OPTIONS)

    assert result.chunks
    for chunk in result.chunks:
        assert chunk.retrieval_eligible is False
        assert chunk.evidence_eligible is False


def test_reference_content_is_preserved_not_deleted() -> None:
    document = build_document(
        [heading("References", 1), paragraph(_REFERENCE_ENTRY, path=("References",))]
    )

    result = chunk_document(document, options=OPTIONS)

    assert "Smith, J. (2022)" in result.chunks[0].text
    assert result.chunks[0].token_count > 0


def test_a_table_inside_a_reference_list_is_not_retrieval_eligible() -> None:
    document = build_document(
        [
            heading("References", 1),
            paragraph(_REFERENCE_ENTRY, path=("References",)),
            table([["Author", "Year"], ["Smith", "2022"]], page=1, path=("References",)),
        ]
    )

    result = chunk_document(document, options=OPTIONS)

    assert all(chunk.content_type is ContentType.REFERENCES for chunk in result.chunks)
    assert all(not chunk.retrieval_eligible for chunk in result.chunks)


def test_literature_chunks_are_evidence_eligible() -> None:
    document = build_document(
        [paragraph("Body prose.", path=("Introduction",))],
        source_path=r"F:\phd-rag\data\literature\AI\paper.pdf",
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.corpus_role is CorpusRole.LITERATURE
    assert chunk.retrieval_eligible is True
    assert chunk.evidence_eligible is True
    assert chunk.authority is AuthorityLevel.PRIMARY_EVIDENCE


def test_my_work_is_retrievable_but_not_external_evidence() -> None:
    document = build_document(
        [paragraph("Draft prose about the framework.", path=("Introduction",))],
        source_path=r"F:\phd-rag\data\my_work\submitted_paper\draft.docx",
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.corpus_role is CorpusRole.MY_WORK
    assert chunk.retrieval_eligible is True
    assert chunk.evidence_eligible is False
    assert chunk.authority is AuthorityLevel.HIGH


def test_summaries_are_navigation_only() -> None:
    document = build_document(
        [paragraph("Summary of the paper.", path=("Summary",))],
        source_path=r"F:\phd-rag\data\summaries\summary.md",
        extension=".md",
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.corpus_role is CorpusRole.SUMMARIES
    assert chunk.content_type is ContentType.SUMMARY_ENTRY
    assert chunk.retrieval_eligible is True
    assert chunk.evidence_eligible is False
    assert chunk.authority is AuthorityLevel.NAVIGATION_ONLY


def test_supervision_is_not_scholarly_evidence() -> None:
    document = build_document(
        [paragraph("Notes from the supervision meeting.", path=("Meeting",))],
        source_path=r"F:\phd-rag\data\supervision\2024-05\notes.docx",
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.corpus_role is CorpusRole.SUPERVISION
    assert chunk.evidence_eligible is False
    assert chunk.authority is AuthorityLevel.MEDIUM


def test_rules_are_instruction_context() -> None:
    document = build_document(
        [paragraph("Submission rules.", path=("Rules",))],
        source_path=r"F:\phd-rag\data\rules\submission_rules.pdf",
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.corpus_role is CorpusRole.RULES
    assert chunk.evidence_eligible is False
    assert chunk.authority is AuthorityLevel.INSTRUCTION


def test_future_chapters_are_planning_guidance_only() -> None:
    document = build_document(
        [paragraph("Ideas for chapter 5.", path=("Chapter 5",))],
        source_path=r"F:\phd-rag\data\future_chapters\chapter5.docx",
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.corpus_role is CorpusRole.FUTURE_CHAPTERS
    assert chunk.retrieval_eligible is True
    assert chunk.evidence_eligible is False
    assert chunk.authority is AuthorityLevel.NAVIGATION_ONLY


def test_guidance_is_instruction_context() -> None:
    document = build_document(
        [paragraph("Supervisor guidance on writing.", path=("Guidance",))],
        source_path=r"F:\phd-rag\data\guidance\writing.pdf",
    )

    chunk = chunk_document(document, options=OPTIONS).chunks[0]

    assert chunk.corpus_role is CorpusRole.GUIDANCE
    assert chunk.evidence_eligible is False
    assert chunk.authority is AuthorityLevel.INSTRUCTION


def test_explicit_role_override_wins() -> None:
    document = build_document([paragraph("Body.", path=("Body",))])

    chunk = chunk_document(document, options=OPTIONS, corpus_role=CorpusRole.MY_WORK).chunks[0]

    assert chunk.corpus_role is CorpusRole.MY_WORK
    assert chunk.evidence_eligible is False


def test_only_literature_is_evidence_eligible() -> None:
    for role in CorpusRole:
        policy = policy_for_role(role)
        assert policy.retrieval_eligible is True
        if role is CorpusRole.LITERATURE:
            assert policy.evidence_eligible is True
        else:
            assert policy.evidence_eligible is False


def test_every_role_has_a_policy() -> None:
    assert set(ROLE_POLICIES) == set(CorpusRole)


def test_unknown_role_is_never_evidence() -> None:
    retrieval, evidence = eligibility_for(None, ContentType.BODY)
    assert retrieval is True
    assert evidence is False
    assert eligibility_for(CorpusRole.LITERATURE, ContentType.REFERENCES) == (False, False)


@pytest.mark.parametrize(
    "content_type",
    [ContentType.BODY, ContentType.ABSTRACT, ContentType.CONCLUSION, ContentType.TABLE],
)
def test_reference_override_applies_only_to_references(content_type: ContentType) -> None:
    retrieval, evidence = eligibility_for(CorpusRole.LITERATURE, content_type)
    assert (retrieval, evidence) == (True, True)
