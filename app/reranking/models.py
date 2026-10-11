"""Reranking-layer models and the reranker interface (RAG-009).

A reranker is a second, slower, *local* relevance model that scores a
``(query, passage)`` pair after dense retrieval has already narrowed the field.
It never replaces the dense score: the two live on different scales and
:class:`app.retrieval.models.Candidate` keeps both.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

#: BGE cross-encoder reranker, the sibling of the RAG-008 dense embedder.
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
#: Library used for loading and scoring.
DEFAULT_LIBRARY = "sentence-transformers"
#: Conservative default: the reranker is small next to BGE-M3, but a passage batch
#: costs far more attention than an embedding batch of the same size.
DEFAULT_BATCH_SIZE = 8


class RerankingError(RuntimeError):
    """Base class for reranking-layer failures."""


class RerankerLoadError(RerankingError):
    """The reranker could not be loaded (missing weights, bad device, ...)."""


class RerankerFailure(RerankingError):
    """A batch of query/passage pairs could not be scored."""


class RerankerSpec(BaseModel):
    """Exactly which reranker produced a score, and on what device."""

    model_config = ConfigDict(frozen=True)

    model_name: str = DEFAULT_RERANKER_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    batch_size: int = Field(default=DEFAULT_BATCH_SIZE, gt=0)
    library: str = DEFAULT_LIBRARY
    library_version: str | None = None
    max_length: int | None = None

    @property
    def identifier(self) -> str:
        """Stable identifier for this reranker configuration (no timestamps)."""
        library = f"{self.library}{self.library_version or ''}"
        return "|".join(
            (
                self.model_name,
                f"rev={self.model_revision or 'unknown'}",
                f"lib={library}",
            )
        )


class RerankInput(BaseModel):
    """One candidate handed to the reranker: identity plus the text to score."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str


class RerankScore(BaseModel):
    """One reranker score, tied to the candidate it belongs to."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    score: float


@runtime_checkable
class Reranker(Protocol):
    """Minimal interface every reranker must satisfy."""

    @property
    def spec(self) -> RerankerSpec:
        """Identity, device and batch size of the loaded reranker."""

    def rerank(self, query: str, candidates: Sequence[RerankInput]) -> list[RerankScore]:
        """Score ``candidates`` against ``query``, in input order."""
