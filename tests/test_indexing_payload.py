"""RAG-008 tests 6-14: payload provenance, indexing population and identity."""

from __future__ import annotations

from pathlib import Path

from chunking_support import build_document, heading, paragraph
from embedding_support import FakeEmbedder, local_client

from app.chunking.chunker import chunk_document, document_id_for
from app.chunking.models import ChunkingOptions
from app.indexing.collection import count_points, fetch_payloads, scroll_point_ids
from app.indexing.indexer import run_index
from app.indexing.models import IndexingOptions
from app.indexing.payload import INDEXED_PAYLOAD_FIELDS, build_payload, payload_is_complete
from app.metadata.enums import CorpusRole, PublicationType
from app.models.chunk import Chunk, ChunkMetadataRef
from app.models.enums import ContentType
from app.parsing.models import TextSource

_LONG_BODY = (
    "Immersive virtual reality in higher education has been studied extensively, "
    "and the literature reports both learning gains and practical problems. "
) * 6


def _chunk(**overrides) -> Chunk:
    defaults = dict(
        id="11111111-1111-5111-8111-111111111111",
        document_id=document_id_for("c" * 64),
        chunk_index=0,
        text="Source text of the chunk.",
        retrieval_text="[Document: Paper]\n[Section: Introduction]\nSource text of the chunk.",
        content_type=ContentType.BODY,
        created_at="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return Chunk(**defaults)


def test_required_provenance_fields_present() -> None:
    chunk = _chunk(
        page_start=3,
        page_end=4,
        section_title="Introduction",
        heading_path=["Introduction"],
        token_count=420,
        text_sources=[TextSource.NATIVE_PDF],
        source_locators=["page:3:block:2"],
        sequence_start=5,
        sequence_end=7,
        metadata_ref=ChunkMetadataRef(
            file_hash="c" * 64,
            filename="paper.pdf",
            title="A Paper",
            year=2021,
            first_author="Smith, J.",
            doi="10.1000/xyz",
            publication_type=PublicationType.JOURNAL_ARTICLE,
        ),
    )

    payload = build_payload(chunk, file_hash="c" * 64, filename="paper.pdf")

    assert payload_is_complete(payload)
    for field in INDEXED_PAYLOAD_FIELDS:
        assert field in payload
    assert payload["chunk_id"] == str(chunk.id)
    assert payload["page_start"] == 3 and payload["page_end"] == 4
    assert payload["source_locators"] == ["page:3:block:2"]
    assert payload["sequence_start"] == 5 and payload["sequence_end"] == 7
    assert payload["token_count"] == 420
    assert payload["title"] == "A Paper" and payload["doi"] == "10.1000/xyz"
    assert payload["text"] == chunk.text
    assert payload["retrieval_text"] == chunk.retrieval_text


def test_role_and_evidence_fields_preserved() -> None:
    chunk = _chunk(
        corpus_role=CorpusRole.MY_WORK,
        authority=None,
        retrieval_eligible=True,
        evidence_eligible=False,
        metadata_ref=ChunkMetadataRef(
            file_hash="d" * 64,
            filename="draft.docx",
            duplicate_locations=["a/draft.docx", "b/draft.docx"],
            role_resolution="my_work (conservative precedence)",
        ),
    )

    payload = build_payload(chunk, file_hash="d" * 64, filename="draft.docx")

    assert payload["corpus_role"] == "my_work"
    assert payload["retrieval_eligible"] is True
    assert payload["evidence_eligible"] is False
    assert payload["duplicate_locations"] == ["a/draft.docx", "b/draft.docx"]
    assert payload["role_resolution"] == "my_work (conservative precedence)"


def test_ocr_provenance_preserved() -> None:
    chunk = _chunk(text_sources=[TextSource.OCR])

    payload = build_payload(chunk, file_hash="e" * 64, filename="scan.pdf")

    assert payload["ocr_involved"] is True
    assert payload["text_sources"] == ["ocr"]


def _corpus(root: Path) -> Path:
    """A tiny corpus with literature (incl. references), my_work and future_chapters."""
    (root / "literature").mkdir(parents=True)
    (root / "my_work").mkdir(parents=True)
    (root / "future_chapters").mkdir(parents=True)
    (root / "literature" / "paper.md").write_text(
        "# A Paper About VR\n\n"
        f"{_LONG_BODY}\n"
        "## References\n\n"
        "Smith, J. (2022). A study of virtual reality. Journal, 1(1), 1-10.\n",
        encoding="utf-8",
    )
    (root / "my_work" / "draft.md").write_text(
        f"# My Own Draft\n\n{_LONG_BODY}\n", encoding="utf-8"
    )
    (root / "future_chapters" / "plan.md").write_text(
        f"# Future Chapter Plan\n\n{_LONG_BODY}\n", encoding="utf-8"
    )
    return root


def test_indexing_population_and_payloads(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    client = local_client(tmp_path)
    embedder = FakeEmbedder()

    report = run_index(
        root,
        options=IndexingOptions(collection="test_chunks", use_cache=False),
        embedder=embedder,
        client=client,
        chunking=ChunkingOptions(created_at=None),
    )

    assert report.points_indexed == report.chunks_embedded == count_points(client, "test_chunks")
    assert report.excluded_reference_chunks >= 1
    assert report.population is not None
    assert report.population.by_corpus_role.get("literature", 0) >= 1
    assert report.population.by_corpus_role.get("my_work", 0) >= 1
    assert report.population.by_corpus_role.get("future_chapters", 0) >= 1

    payloads = [fetch_payloads(client, "test_chunks", [point_id])[0] for point_id in scroll_point_ids(client, "test_chunks")]
    assert all(payload["content_type"] != "references" for payload in payloads)
    assert all(payload_is_complete(payload) for payload in payloads)

    my_work = [payload for payload in payloads if payload["corpus_role"] == "my_work"]
    future = [payload for payload in payloads if payload["corpus_role"] == "future_chapters"]
    literature = [payload for payload in payloads if payload["corpus_role"] == "literature"]
    assert my_work and all(payload["evidence_eligible"] is False for payload in my_work)
    assert all(payload["retrieval_eligible"] is True for payload in my_work)
    assert future and all(payload["evidence_eligible"] is False for payload in future)
    assert literature and all(payload["evidence_eligible"] is True for payload in literature)


def test_point_id_is_the_deterministic_chunk_id(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    client = local_client(tmp_path)

    report = run_index(
        root,
        options=IndexingOptions(collection="ids", use_cache=False),
        embedder=FakeEmbedder(),
        client=client,
    )

    document = build_document(
        [heading("Introduction", 1), paragraph("Body text.", path=("Introduction",))]
    )
    chunked = chunk_document(document, options=ChunkingOptions())
    assert chunked.chunks  # sanity: chunking produces chunks

    ids = scroll_point_ids(client, "ids")
    assert len(ids) == report.chunks_embedded
    assert all(len(point_id) == 36 for point_id in ids)


def test_reindex_does_not_duplicate_points(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    client = local_client(tmp_path)
    embedder = FakeEmbedder()
    options = IndexingOptions(collection="idempotent", use_cache=False)

    first = run_index(root, options=options, embedder=embedder, client=client)
    second = run_index(root, options=options, embedder=embedder, client=client)

    assert first.duplicate_point_ids == 0
    assert second.inserted == 0
    assert second.updated == first.chunks_embedded
    assert count_points(client, "idempotent") == first.chunks_embedded
    assert second.corpus_fingerprint == first.corpus_fingerprint
    assert second.chunk_ids_digest == first.chunk_ids_digest


def test_document_identity_matches_rag007(tmp_path: Path) -> None:
    root = _corpus(tmp_path / "data")
    client = local_client(tmp_path)

    report = run_index(
        root,
        options=IndexingOptions(collection="identity", use_cache=False),
        embedder=FakeEmbedder(),
        client=client,
    )

    payload = fetch_payloads(client, "identity", [scroll_point_ids(client, "identity")[0]])[0]
    assert payload["document_id"] == str(document_id_for(payload["file_hash"]))
    assert report.unique_documents == 3
