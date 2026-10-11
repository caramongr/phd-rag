"""Shared fixtures for the RAG-009 retrieval tests.

No network, no model download, no Qdrant server: a controllable embedder, a
deterministic reranker double, and Qdrant's local/in-memory mode. Point vectors are
placed at known angles from the query vector so the *expected* dense order of a
test is arithmetic rather than guesswork.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from qdrant_client import QdrantClient, models

from tests.embedding_support import FAKE_DIMENSION, FakeEmbedder

#: Local Qdrant mode only accepts UUID point ids, so readable test chunk ids are
#: hashed into a stable UUID; the payload keeps the readable id.
POINT_ID_NAMESPACE = uuid.UUID("2f4c1c9e-6a1d-4d5f-9d5b-8f0b1a2c3d4e")


def point_id_for(chunk_id: str) -> str:
    """Deterministic UUID point id for a readable test chunk id."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


#: Collections used by the retrieval tests are tiny (8 dimensions).
TEST_COLLECTION = "test_retrieval_chunks"


def direction(angle_degrees: float, dimension: int = FAKE_DIMENSION) -> list[float]:
    """Unit vector at ``angle_degrees`` from the query direction (cosine = cos θ)."""
    radians = math.radians(angle_degrees)
    vector = [0.0] * dimension
    vector[0] = math.cos(radians)
    vector[1] = math.sin(radians)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class QueryEmbedder(FakeEmbedder):
    """Embedder double whose query vector is chosen by the test."""

    def __init__(
        self,
        *,
        query_vector: Sequence[float] | None = None,
        vectors: dict[str, Sequence[float]] | None = None,
        dimension: int = FAKE_DIMENSION,
    ) -> None:
        super().__init__(dimension=dimension, model_name="fake/bge-m3", revision="test-rev")
        self.query_vector = list(query_vector if query_vector is not None else direction(0.0))
        self.vectors = {key: list(value) for key, value in (vectors or {}).items()}
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        """Record the query and return its vector."""
        self.queries.append(text)
        return list(self.vectors.get(text, self.query_vector))

    def warm_up(self) -> float:
        self.embed_query("warm up")
        return 0.0


class FakeReranker:
    """Deterministic reranker double.

    ``score_fn`` receives ``(query, chunk_id, text)`` so a test can also assert
    *which text* the reranker was given (RAG-009 must pass ``retrieval_text``).
    """

    def __init__(
        self,
        *,
        score_fn: Callable[[str, str, str], float] | None = None,
        score: float = 0.5,
        batch_size: int = 8,
        device: str = "cpu",
        model_name: str = "fake/bge-reranker-v2-m3",
        revision: str | None = "test-rev",
    ) -> None:
        self.score_fn = score_fn
        self.score = score
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.revision = revision
        self.calls: list[tuple[str, list[str]]] = []
        self.seen_texts: dict[str, str] = {}
        self.last_seconds = 0.0
        self.warm_ups = 0

    @property
    def spec(self) -> Any:
        from app.reranking.models import RerankerSpec

        return RerankerSpec(
            model_name=self.model_name,
            model_revision=self.revision,
            device=self.device,
            batch_size=self.batch_size,
            library="fake",
            library_version="0",
            max_length=8192,
        )

    def rerank(self, query: str, candidates: Sequence[Any]) -> list[Any]:
        """Score candidates in input order."""
        from app.reranking.models import RerankScore

        self.calls.append((query, [candidate.chunk_id for candidate in candidates]))
        scores = []
        for candidate in candidates:
            self.seen_texts[candidate.chunk_id] = candidate.text
            value = (
                self.score_fn(query, candidate.chunk_id, candidate.text)
                if self.score_fn is not None
                else self.score
            )
            scores.append(RerankScore(chunk_id=candidate.chunk_id, score=float(value)))
        return scores

    def warm_up(self) -> float:
        self.warm_ups += 1
        return 0.0


def point(
    chunk_id: str,
    *,
    angle: float = 0.0,
    text: str | None = None,
    retrieval_text: str | None = None,
    document_id: str | None = None,
    filename: str | None = None,
    corpus_role: str = "literature",
    evidence_eligible: bool | None = None,
    content_type: str = "body",
    retrieval_eligible: bool = True,
    chunk_index: int = 0,
    year: int | None = 2024,
    title: str | None = None,
    first_author: str | None = "Author",
    doi: str | None = None,
    publication_type: str | None = "journal-article",
    page_start: int | None = 1,
    page_end: int | None = 2,
    section_title: str | None = "Introduction",
    heading_path: Sequence[str] = (),
    text_sources: Sequence[str] = ("native",),
    ocr_involved: bool = False,
    source_locators: Sequence[str] = (),
    dimension: int = FAKE_DIMENSION,
) -> dict[str, Any]:
    """Build one Qdrant point payload matching the RAG-008 schema."""
    if evidence_eligible is None:
        evidence_eligible = corpus_role == "literature"
    body = text if text is not None else f"{chunk_id} chunk text about artificial intelligence"
    return {
        "vector": direction(angle, dimension),
        "payload": {
            "chunk_id": chunk_id,
            "document_id": document_id or f"doc-{chunk_id}",
            "file_hash": f"hash-{chunk_id}",
            "filename": filename or f"{chunk_id}.pdf",
            "chunk_index": chunk_index,
            "page_start": page_start,
            "page_end": page_end,
            "section_title": section_title,
            "heading_path": list(heading_path),
            "content_type": content_type,
            "corpus_role": corpus_role,
            "authority": "primary" if corpus_role == "literature" else "own",
            "evidence_eligible": bool(evidence_eligible),
            "retrieval_eligible": bool(retrieval_eligible),
            "text_sources": list(text_sources),
            "ocr_involved": bool(ocr_involved),
            "title": title,
            "first_author": first_author,
            "year": year,
            "doi": doi,
            "publication_type": publication_type,
            "source_locators": list(source_locators),
            "sequence_start": 0,
            "sequence_end": 10,
            "token_count": 10,
            "text": body,
            "retrieval_text": retrieval_text if retrieval_text is not None else body,
        },
    }


def seed_collection(
    client: QdrantClient,
    points: Iterable[dict[str, Any]],
    *,
    collection: str = TEST_COLLECTION,
    dimension: int = FAKE_DIMENSION,
) -> str:
    """Create ``collection`` and upsert ``points``."""
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )
    built = [
        models.PointStruct(
            id=point_id_for(item["payload"]["chunk_id"]),
            vector=item["vector"],
            payload=item["payload"],
        )
        for item in points
    ]
    if built:
        client.upsert(collection_name=collection, points=built)
    return collection


def local_collection(tmp_path: Path, points: Iterable[dict[str, Any]], *, name: str = "q") -> QdrantClient:
    """A local persistent client with a seeded collection (flushes on close)."""
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    seed_collection(client, points)
    return client


def memory_collection(points: Iterable[dict[str, Any]]) -> QdrantClient:
    """An isolated in-memory client with a seeded collection."""
    client = QdrantClient(location=":memory:")
    seed_collection(client, points)
    return client


def make_service(
    client: QdrantClient,
    *,
    embedder: QueryEmbedder | None = None,
    reranker: Any | None = None,
    collection: str = TEST_COLLECTION,
    candidate_k: int = 25,
    final_k: int = 8,
    max_chunks_per_document: int = 3,
    suppress_duplicates: bool = True,
    duplicate_overlap_threshold: float = 0.85,
) -> Any:
    """Build a :class:`RetrievalService` wired to the test doubles."""
    from app.retrieval.service import RetrievalService

    return RetrievalService(
        embedder=embedder or QueryEmbedder(),
        reranker=reranker,
        client=client,
        collection=collection,
        candidate_k=candidate_k,
        final_k=final_k,
        max_chunks_per_document=max_chunks_per_document,
        suppress_duplicates=suppress_duplicates,
        duplicate_overlap_threshold=duplicate_overlap_threshold,
    )


def corpus_points() -> list[dict[str, Any]]:
    """A small mixed-role corpus: literature, my_work, future_chapters, references.

    Angles control the dense order: 5° ranks highest, then 10°, 15°, ...
    """
    return [
        point(
            "lit-a1",
            angle=5,
            document_id="doc-lit-a",
            filename="lit-a.pdf",
            title="Evidence A on AI in higher education",
            chunk_index=0,
            text="Artificial intelligence in higher education improves learning outcomes.",
        ),
        point(
            "lit-a2",
            angle=10,
            document_id="doc-lit-a",
            filename="lit-a.pdf",
            title="Evidence A on AI in higher education",
            chunk_index=1,
            text="A second, distinct passage from the same evidence paper about tutoring.",
        ),
        point(
            "lit-b1",
            angle=15,
            document_id="doc-lit-b",
            filename="lit-b.pdf",
            title="Evidence B on virtual reality",
            text="Virtual reality in higher education supports student engagement.",
        ),
        point(
            "own-1",
            angle=20,
            document_id="doc-own",
            filename="second-paper-draft-final.docx",
            corpus_role="my_work",
            title="My own unpublished draft",
            text="My own draft claims about artificial intelligence in higher education.",
        ),
        point(
            "own-2",
            angle=25,
            document_id="doc-own",
            filename="second-paper-draft-final.docx",
            corpus_role="my_work",
            title="My own unpublished draft",
            chunk_index=1,
            text="A different section of my own draft about virtual reality.",
        ),
        point(
            "future-1",
            angle=30,
            document_id="doc-future",
            filename="chapter-3-historical-development.docx",
            corpus_role="future_chapters",
            title="Chapter 3 Historical Development",
            text="Future chapter material on the historical development of AI.",
        ),
        point(
            "summary-1",
            angle=35,
            document_id="doc-summary",
            filename="summaries-27-09-2025.docx",
            corpus_role="summaries",
            title=None,
            text="My summary of the literature on immersive learning.",
        ),
        point(
            "rules-1",
            angle=40,
            document_id="doc-rules",
            filename="Reference checklist.docx",
            corpus_role="rules",
            title="Reference checklist",
            text="Harvard referencing rules for the thesis.",
        ),
        point(
            "guidance-1",
            angle=45,
            document_id="doc-guidance",
            filename="future_chapters_guidance_policy.md",
            corpus_role="guidance",
            title="Future Chapters Guidance Policy",
            text="Guidance policy for writing future chapters.",
        ),
        point(
            "sup-1",
            angle=50,
            document_id="doc-supervision",
            filename="APR-Form-2026.docx",
            corpus_role="supervision",
            title=None,
            text="Supervisor comments on methodology and research design.",
        ),
        point(
            "ref-1",
            angle=55,
            document_id="doc-lit-a",
            filename="lit-a.pdf",
            content_type="references",
            retrieval_eligible=False,
            title="Evidence A on AI in higher education",
            text="Adams, J. (2019). A reference entry that must never be retrieved.",
        ),
    ]
