"""Dense embedding inspection CLI (RAG-008).

Usage::

    python -m app.embedding.inspect [--model BAAI/bge-m3] [--device cpu]
        [--text "some text"] [--show-vector 8] [--local-files-only]

Prints the loaded model's identity, device, vector dimension, normalization and a
sample input with its vector norm. The vector itself is only printed when
``--show-vector N`` asks for it: dumping 1024 floats by default helps nobody.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from app.config import get_settings
from app.embedding.bge_m3 import BgeM3Embedder, resolve_device
from app.embedding.cache import unit_norm
from app.embedding.models import DEFAULT_MODEL_NAME, EmbeddingModelLoadError

_SAMPLE_TEXT = "Academic-aware chunking keeps page and section provenance."


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.embedding.inspect",
        description="Inspect the dense embedding model used for indexing.",
    )
    parser.add_argument("--model", default=settings.embedding_model or DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--text", default=_SAMPLE_TEXT, help="Sample input to embed.")
    parser.add_argument(
        "--show-vector",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N vector dimensions (default: none).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def render(embedder: Any, sample: str, *, show_vector: int = 0) -> str:
    """Render the embedding inspection for any embedder-like object."""
    spec = embedder.spec
    vector = embedder.embed_query(sample)
    lines = [
        "RAG-008 embedding inspection",
        "============================",
        "",
        f"model:                 {spec.model_name}",
        f"revision:              {spec.model_revision or 'unknown'}",
        f"identifier:            {spec.identifier}",
        f"library:               {spec.library} {spec.library_version or ''}".rstrip(),
        f"device:                {spec.device}",
        f"vector dimension:      {spec.dimension}",
        f"normalization:         {'l2 (unit norm)' if spec.normalized else 'none'}",
        f"max input tokens:      {spec.max_tokens or 'unknown'}",
        "",
        f"sample input:          {sample}",
        f"sample tokens:         {embedder.count_tokens(sample)}",
        f"sample vector norm:    {unit_norm(vector):.6f}",
        f"sample vector length:  {len(vector)}",
    ]
    if show_vector:
        preview = ", ".join(f"{value:.6f}" for value in vector[:show_vector])
        lines.append(f"sample vector (first {show_vector} dims): {preview}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, *, embedder_factory=None) -> int:
    """Run the embedding inspection CLI."""
    args = _build_parser().parse_args(argv)
    try:
        embedder = (embedder_factory or BgeM3Embedder)(
            model_name=args.model,
            revision=args.revision,
            device=resolve_device(args.device),
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
        )
    except EmbeddingModelLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    try:
        text = render(embedder, args.text, show_vector=args.show_vector)
    except EmbeddingModelLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 3
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
