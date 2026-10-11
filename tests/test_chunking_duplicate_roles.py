"""RAG-007 correction: role safety for deduplicated (SHA-256 identical) documents.

A logical document is one set of identical bytes, but it can exist in several
folders and therefore in several corpus roles. The role must be resolved from
*all* physical locations, and the most conservative role must win, so a copy filed
under ``literature`` can never promote the author's own work - or a summary, a
supervision note, a rule file or a future chapter - to scholarly evidence.
"""

from __future__ import annotations

from pathlib import Path

from chunking_support import build_document, heading, paragraph

from app.chunking.chunker import chunk_document
from app.chunking.models import ChunkingOptions
from app.chunking.report import collect
from app.ingestion.scanner import scan_directory
from app.metadata.corpus import ROLE_PRECEDENCE, resolve_corpus_role
from app.metadata.enums import CorpusRole
from app.parsing.parser import parse_document

_BODY = (
    "# A Draft Paper\n\n"
    "This draft describes the study design, the participants, the instrument and "
    "the analysis plan for the second paper of the thesis, together with the "
    "expected contribution and the limitations that still have to be addressed.\n"
)


def _write(root: Path, relative: str, text: str = _BODY) -> Path:
    """Create ``relative`` under ``root`` and return the file path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _chunk_logical_document(root: Path) -> object:
    """Chunk the single logical document found under ``root``."""
    document = scan_directory(root).documents[0]
    parsed = parse_document(Path(document.canonical_path), file_hash=document.file_hash)
    from app.metadata.extractor import extract_metadata

    metadata = extract_metadata(
        Path(document.canonical_path),
        file_hash=document.file_hash,
        physical_paths=[location.path for location in document.physical_locations],
    )
    return chunk_document(parsed, metadata=metadata, options=ChunkingOptions()), metadata


def test_only_literature_is_evidence(tmp_path: Path) -> None:
    _write(tmp_path, "literature/ai/paper.md")
    result, metadata = _chunk_logical_document(tmp_path)

    assert metadata.corpus_role is CorpusRole.LITERATURE
    assert result.chunk_count == 1
    assert result.chunks[0].corpus_role is CorpusRole.LITERATURE
    assert result.chunks[0].evidence_eligible is True
    assert result.chunks[0].retrieval_eligible is True
    assert metadata.role_resolution.reason == "single physical location"


def test_only_my_work_is_not_evidence(tmp_path: Path) -> None:
    _write(tmp_path, "my_work/current_paper/draft.md")
    result, metadata = _chunk_logical_document(tmp_path)

    assert metadata.corpus_role is CorpusRole.MY_WORK
    assert result.chunks[0].corpus_role is CorpusRole.MY_WORK
    assert result.chunks[0].evidence_eligible is False
    assert result.chunks[0].retrieval_eligible is True


def test_duplicate_in_literature_and_my_work_prefers_my_work(tmp_path: Path) -> None:
    _write(tmp_path, "literature/Second paper/draft.md")
    _write(tmp_path, "my_work/current_paper/draft.md")

    resolution = resolve_corpus_role(
        [
            str(tmp_path / "literature" / "Second paper" / "draft.md"),
            str(tmp_path / "my_work" / "current_paper" / "draft.md"),
        ]
    )
    assert resolution.effective_role is CorpusRole.MY_WORK
    assert resolution.has_conflicting_roles
    assert "my_work preferred over literature" in resolution.reason

    result, metadata = _chunk_logical_document(tmp_path)
    assert metadata.corpus_role is CorpusRole.MY_WORK
    assert result.chunk_count == 1
    assert result.chunks[0].evidence_eligible is False
    assert result.chunks[0].retrieval_eligible is True


def test_duplicate_in_literature_and_summaries_is_not_evidence(tmp_path: Path) -> None:
    _write(tmp_path, "literature/Second paper/draft.md")
    _write(tmp_path, "summaries/draft.md")
    result, metadata = _chunk_logical_document(tmp_path)

    assert metadata.corpus_role is CorpusRole.SUMMARIES
    assert result.chunks[0].evidence_eligible is False


def test_duplicate_in_literature_and_future_chapters_is_not_evidence(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "literature/Second paper/draft.md")
    _write(tmp_path, "future_chapters/draft.md")
    result, metadata = _chunk_logical_document(tmp_path)

    assert metadata.corpus_role is CorpusRole.FUTURE_CHAPTERS
    assert result.chunks[0].evidence_eligible is False


def test_every_non_literature_role_beats_literature() -> None:
    for role in ROLE_PRECEDENCE:
        if role in (CorpusRole.LITERATURE, CorpusRole.OTHER):
            continue
        assert ROLE_PRECEDENCE.index(role) < ROLE_PRECEDENCE.index(CorpusRole.LITERATURE)
    assert ROLE_PRECEDENCE[-1] is CorpusRole.OTHER


def test_discovery_order_does_not_change_the_effective_role(tmp_path: Path) -> None:
    # Forward: alphabetical discovery finds ``literature`` before ``my_work``.
    forward = tmp_path / "forward"
    _write(forward, "literature/Second paper/draft.md")
    _write(forward, "my_work/current_paper/draft.md")
    # Reversed: the my_work copy is discovered first.
    reversed_root = tmp_path / "reversed"
    _write(reversed_root, "aaa/my_work/draft.md")
    _write(reversed_root, "zzz/literature/draft.md")

    forward_document = scan_directory(forward).documents[0]
    reversed_document = scan_directory(reversed_root).documents[0]
    assert "literature" in forward_document.canonical_path
    assert "my_work" in reversed_document.canonical_path

    forward_result, forward_metadata = _chunk_logical_document(forward)
    reversed_result, reversed_metadata = _chunk_logical_document(reversed_root)

    assert forward_metadata.corpus_role is reversed_metadata.corpus_role
    assert forward_metadata.corpus_role is CorpusRole.MY_WORK
    assert forward_result.chunks[0].corpus_role is reversed_result.chunks[0].corpus_role
    assert forward_result.chunks[0].evidence_eligible is False
    assert reversed_result.chunks[0].evidence_eligible is False
    assert [chunk.id for chunk in forward_result.chunks] == [
        chunk.id for chunk in reversed_result.chunks
    ]
    assert [chunk.chunk_index for chunk in forward_result.chunks] == [
        chunk.chunk_index for chunk in reversed_result.chunks
    ]


def test_physical_locations_stay_inspectable(tmp_path: Path) -> None:
    _write(tmp_path, "literature/Second paper/draft.md")
    _write(tmp_path, "my_work/current_paper/draft.md")
    result, metadata = _chunk_logical_document(tmp_path)

    resolution = metadata.role_resolution
    assert resolution is not None
    assert len(resolution.physical_roles) == 2
    assert [physical.role for physical in resolution.physical_roles] == [
        CorpusRole.LITERATURE,
        CorpusRole.MY_WORK,
    ]
    assert resolution.physical_paths == sorted(resolution.physical_paths)
    assert len(resolution.applied_precedence) == len(ROLE_PRECEDENCE)

    ref = result.chunks[0].metadata_ref
    assert ref is not None
    assert ref.duplicate_locations == resolution.physical_paths
    assert ref.role_resolution is not None
    assert "my_work" in ref.role_resolution


def test_duplicate_copies_produce_one_logical_document(tmp_path: Path) -> None:
    duplicated_root = tmp_path / "duplicated"
    single = tmp_path / "single"
    _write(duplicated_root, "literature/Second paper/draft.md")
    _write(duplicated_root, "my_work/current_paper/draft.md")
    _write(single, "literature/Second paper/draft.md")

    scan = scan_directory(duplicated_root)
    assert scan.unique_documents == 1
    assert scan.duplicate_copies == 1

    duplicated, duplicated_metadata = _chunk_logical_document(duplicated_root)
    alone, alone_metadata = _chunk_logical_document(single)

    assert duplicated.chunk_count == alone.chunk_count == 1
    assert [chunk.id for chunk in duplicated.chunks] == [
        chunk.id for chunk in alone.chunks
    ]
    assert duplicated_metadata.file_hash == alone_metadata.file_hash
    assert len(duplicated_metadata.role_resolution.physical_roles) == 2
    assert alone_metadata.corpus_role is CorpusRole.LITERATURE

    report = collect(duplicated_root)
    assert report.unique_documents == 1
    assert report.multi_location_documents == 1
    assert report.multi_role_documents == 1
    assert report.total_chunks == 1
    assert report.by_corpus_role == {"my_work": 1}
    assert report.evidence_eligible_chunks == 0
    assert report.retrieval_eligible_chunks == 1
    audit = report.role_audit_rows[0]
    assert audit.effective_role_before == "literature"
    assert audit.effective_role_after == "my_work"
    assert audit.evidence_eligible_before is True
    assert audit.evidence_eligible_after is False


def test_doi_stage_gate_uses_all_locations(tmp_path: Path) -> None:
    """The RAG-006 stage wiring resolves the role from every physical copy too."""
    from metadata_support import FakeResponse, FakeTransport, build_test_stack, crossref_message, make_pdf

    from app.metadata.doi_stage import run_doi_stage
    from app.metadata.external.cache import JsonCache

    root = tmp_path / "data"
    literature = root / "literature" / "Second paper"
    literature.mkdir(parents=True)
    source = make_pdf(
        literature,
        "summary.pdf",
        lines=[
            ("A Prominent First Page Title", 18.0, 80.0),
            ("Body text with doi:10.1000/xyz here.", 10.0, 130.0),
        ],
    )
    summaries = root / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "summary.pdf").write_bytes(source.read_bytes())

    transport = FakeTransport(lambda url, params: FakeResponse(200, crossref_message()))
    enricher, components = build_test_stack(transport, JsonCache(tmp_path / "cache.json"))

    result = run_doi_stage(root, enricher=enricher, components=components)

    assert result.unique_documents == 1
    assert result.with_local_doi == 1
    assert result.skipped_not_enrichable == 1
    assert transport.calls == []


def test_inspect_cli_resolves_role_across_corpus(tmp_path: Path) -> None:
    from app.chunking.inspect import render as render_inspection

    _write(tmp_path, "literature/Second paper/draft.md")
    _write(tmp_path, "my_work/current_paper/draft.md")
    literature_copy = tmp_path / "literature" / "Second paper" / "draft.md"

    alone = render_inspection(literature_copy, ocr=False)
    assert "Corpus role: literature" in alone
    assert "evidence yes" in alone

    resolved = render_inspection(literature_copy, ocr=False, corpus_root=tmp_path)
    assert "Corpus role: my_work" in resolved
    assert "conservative precedence: my_work preferred over literature" in resolved
    assert "evidence no" in resolved


def test_chunk_without_metadata_keeps_path_derived_role() -> None:
    document = build_document(
        [heading("Introduction", 1), paragraph("Body text.", path=("Introduction",))],
        source_path=r"F:\phd-rag\data\my_work\current_paper\draft.pdf",
    )
    result = chunk_document(document, options=ChunkingOptions())

    assert result.chunks[0].corpus_role is CorpusRole.MY_WORK
    assert result.chunks[0].evidence_eligible is False
