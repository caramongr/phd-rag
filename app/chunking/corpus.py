"""Shared read-only corpus iteration for the chunking stages (RAG-007 / RAG-008).

Both the RAG-007 corpus report and the RAG-008 indexing pipeline must see exactly
the same documents, the same parsed content, the same RAG-006 metadata (including
the role resolved from *all* physical locations) and therefore the same chunks.
Iterating in one place is what keeps them in agreement; duplicating the sequence
would let the two drift apart.

The iteration is deterministic: documents are visited in filename order, the scan
never writes to the corpus, and every failure is reported as a status instead of
aborting the run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.chunking.chunker import chunk_document
from app.chunking.models import ChunkedDocument, ChunkingOptions
from app.ingestion.models import DiscoveredDocument, ScanResult
from app.ingestion.scanner import scan_directory
from app.metadata.extractor import extract_metadata
from app.models.chunk import Chunk
from app.parsing.exceptions import (
    DocumentParseError,
    EncryptedDocumentError,
    UnsupportedDocumentFormat,
)
from app.parsing.parser import parse_document

#: Extensions this project can chunk.
CHUNKABLE_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".md"})

#: Formats the scanner supports but chunking deliberately does not.
UNSUPPORTED_FOR_CHUNKING: frozenset[str] = frozenset({".xlsx", ".xls", ".doc", ".rtf", ".txt"})

#: Outcome of one logical document in a corpus pass.
STATUS_CHUNKED = "chunked"
STATUS_WITHOUT_CHUNKS = "without_chunks"
STATUS_UNSUPPORTED = "unsupported_for_chunking"
STATUS_PARSE_ERROR = "parse_error"
STATUS_CHUNKING_ERROR = "chunking_error"


class CorpusScan(BaseModel):
    """The scan of a corpus root plus the deterministic visit order."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    root: str
    unique_documents: int = 0
    duplicate_copies: int = 0
    ordered: list[DiscoveredDocument] = Field(default_factory=list)


class CorpusDocument(BaseModel):
    """One logical document visited by a corpus pass, with its outcome."""

    model_config = ConfigDict(extra="forbid")

    document: DiscoveredDocument
    status: str
    chunked: ChunkedDocument | None = None
    detail: str | None = None

    @property
    def filename(self) -> str:
        """Basename of the canonical copy."""
        return self.document.filename

    @property
    def extension(self) -> str:
        """Lower-cased extension of the canonical copy."""
        return self.document.extension.lower()

    @property
    def file_hash(self) -> str:
        """SHA-256 identity of the logical document."""
        return self.document.file_hash

    @property
    def document_id(self) -> UUID | None:
        """Deterministic logical-document id, when the document was chunked."""
        return self.chunked.document_id if self.chunked is not None else None

    @property
    def chunks(self) -> list[Chunk]:
        """Chunks produced for this document (empty when it was skipped)."""
        return self.chunked.chunks if self.chunked is not None else []


class CorpusPass(BaseModel):
    """Aggregate of one corpus pass (used by the RAG-008 dry run)."""

    model_config = ConfigDict(extra="forbid")

    root: str
    unique_documents: int = 0
    duplicate_copies: int = 0
    documents: list[CorpusDocument] = Field(default_factory=list)

    def count_status(self, status: str) -> int:
        """Number of documents that ended with ``status``."""
        return sum(1 for record in self.documents if record.status == status)

    @property
    def all_chunks(self) -> list[Chunk]:
        """Every chunk of the pass, in deterministic order."""
        return [chunk for record in self.documents for chunk in record.chunks]


def scan_corpus(root: str | Path, *, limit: int = 0) -> CorpusScan:
    """Scan ``root`` and return the deterministic visit order.

    ``limit`` truncates the visit order only; ``unique_documents`` always reports
    the full logical corpus, so a limited run can still state the real corpus size.
    """
    scan: ScanResult = scan_directory(Path(root))
    ordered = sorted(scan.documents, key=lambda document: document.filename)
    if limit > 0:
        ordered = ordered[:limit]
    return CorpusScan(
        root=str(root),
        unique_documents=scan.unique_documents,
        duplicate_copies=scan.duplicate_copies,
        ordered=ordered,
    )


def iter_corpus_documents(
    root: str | Path,
    *,
    options: ChunkingOptions | None = None,
    limit: int = 0,
    ocr: bool | None = None,
    scan: CorpusScan | None = None,
) -> Iterator[CorpusDocument]:
    """Yield every logical document in ``scan`` with its chunking outcome.

    Documents are visited in filename order. Unsupported formats are yielded too
    (with :data:`STATUS_UNSUPPORTED`) so callers can count them instead of losing
    them silently, and a parser or chunker failure never aborts the pass.
    """
    corpus_scan = scan or scan_corpus(root, limit=limit)
    chunking_options = options or ChunkingOptions()

    for document in corpus_scan.ordered:
        extension = document.extension.lower()
        if extension not in CHUNKABLE_EXTENSIONS:
            reason = (
                "unsupported for chunking in RAG-007"
                if extension in UNSUPPORTED_FOR_CHUNKING
                else "unsupported format"
            )
            yield CorpusDocument(document=document, status=STATUS_UNSUPPORTED, detail=reason)
            continue

        path = Path(document.canonical_path)
        try:
            parsed = parse_document(path, file_hash=document.file_hash, ocr=ocr)
        except (DocumentParseError, EncryptedDocumentError, UnsupportedDocumentFormat) as exc:
            yield CorpusDocument(
                document=document,
                status=STATUS_PARSE_ERROR,
                detail=f"parse error: {exc}",
            )
            continue
        except Exception as exc:  # defensive: a parser crash must not abort the pass
            yield CorpusDocument(
                document=document,
                status=STATUS_PARSE_ERROR,
                detail=f"parse error: {type(exc).__name__}: {exc}",
            )
            continue

        metadata = extract_metadata(
            path,
            file_hash=document.file_hash,
            physical_paths=[location.path for location in document.physical_locations],
        )
        try:
            result = chunk_document(parsed, metadata=metadata, options=chunking_options)
        except Exception as exc:  # defensive: chunking a bad document must not abort
            yield CorpusDocument(
                document=document,
                status=STATUS_CHUNKING_ERROR,
                detail=f"chunking error: {type(exc).__name__}: {exc}",
            )
            continue

        yield CorpusDocument(
            document=document,
            status=STATUS_CHUNKED if result.chunks else STATUS_WITHOUT_CHUNKS,
            chunked=result,
        )


def iter_corpus_chunks(
    root: str | Path,
    *,
    options: ChunkingOptions | None = None,
    limit: int = 0,
    ocr: bool | None = None,
    retrieval_eligible_only: bool = False,
    scan: CorpusScan | None = None,
) -> Iterator[tuple[CorpusDocument, Chunk]]:
    """Yield ``(document, chunk)`` pairs in deterministic corpus order."""
    for record in iter_corpus_documents(
        root, options=options, limit=limit, ocr=ocr, scan=scan
    ):
        for chunk in record.chunks:
            if retrieval_eligible_only and not chunk.retrieval_eligible:
                continue
            yield record, chunk


def build_corpus_pass(
    root: str | Path,
    *,
    options: ChunkingOptions | None = None,
    limit: int = 0,
    ocr: bool | None = None,
) -> CorpusPass:
    """Run the pass eagerly and keep every record (for dry runs and tests)."""
    scan = scan_corpus(root, limit=limit)
    documents = list(
        iter_corpus_documents(root, options=options, limit=limit, ocr=ocr, scan=scan)
    )
    return CorpusPass(
        root=scan.root,
        unique_documents=scan.unique_documents,
        duplicate_copies=scan.duplicate_copies,
        documents=documents,
    )
