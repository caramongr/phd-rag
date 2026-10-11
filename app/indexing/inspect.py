"""Inspection CLI for an indexed collection (RAG-008).

Usage::

    python -m app.indexing.inspect [--collection phd_rag_chunks_v1]
        [--qdrant-path .qdrant_data | --qdrant-url http://localhost:6333]
        [--limit 5] [--show-text] [--check]

Reports the point count, the role/evidence/content-type distribution, a few sample
payloads and - with ``--check`` - the pilot assertions of RAG-008 §26 (literature is
evidence, ``my_work`` and ``future_chapters`` are not, OCR chunks are marked, no
reference chunk is in the normal collection).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import Any, NamedTuple, Sequence

from app.config import get_settings
from app.indexing.collection import (
    QdrantConfigurationError,
    build_client,
    count_points,
    sample_payloads,
)
from app.indexing.payload import INDEXED_PAYLOAD_FIELDS

_PREVIEW_CHARS = 160
_TEXT_FIELDS = ("chunk_id", "document_id", "file_hash", "filename", "chunk_index", "content_type", "corpus_role", "evidence_eligible", "retrieval_eligible", "ocr_involved", "token_count")


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m app.indexing.inspect",
        description="Inspect an indexed RAG-008 collection.",
    )
    parser.add_argument("--collection", default=settings.qdrant_collection)
    parser.add_argument("--qdrant-url", default=settings.qdrant_url)
    parser.add_argument("--qdrant-api-key", default=settings.qdrant_api_key)
    parser.add_argument("--qdrant-path", default=settings.qdrant_path)
    parser.add_argument("--limit", type=int, default=5, help="Sample points to print.")
    parser.add_argument("--show-text", action="store_true", help="Print text previews.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the RAG-008 §26 pilot assertions over the whole collection.",
    )
    return parser


def _preview(text: str, width: int = _PREVIEW_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:width] + ("…" if len(collapsed) > width else "")


def collect_statistics(client, collection: str) -> dict[str, Any]:
    """Aggregate payload statistics over the whole collection."""
    roles: Counter[str] = Counter()
    content_types: Counter[str] = Counter()
    evidence: Counter[str] = Counter()
    retrieval: Counter[str] = Counter()
    ocr_chunks = 0
    missing_fields: Counter[str] = Counter()
    reference_points = 0
    total = 0
    my_work_not_evidence = True
    future_chapters_not_evidence = True
    literature_is_evidence = True
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            total += 1
            role = str(payload.get("corpus_role"))
            evidence_eligible = payload.get("evidence_eligible")
            roles[role] += 1
            content_types[str(payload.get("content_type"))] += 1
            evidence[str(evidence_eligible)] += 1
            retrieval[str(payload.get("retrieval_eligible"))] += 1
            if payload.get("ocr_involved"):
                ocr_chunks += 1
            if payload.get("content_type") == "references":
                reference_points += 1
            if role == "my_work" and evidence_eligible is not False:
                my_work_not_evidence = False
            if role == "future_chapters" and evidence_eligible is not False:
                future_chapters_not_evidence = False
            if role == "literature" and evidence_eligible is not True:
                literature_is_evidence = False
            for field in INDEXED_PAYLOAD_FIELDS:
                if field not in payload:
                    missing_fields[field] += 1
        if offset is None or not points:
            break
    return {
        "total": total,
        "by_corpus_role": dict(sorted(roles.items())),
        "by_content_type": dict(sorted(content_types.items())),
        "by_evidence_eligible": dict(sorted(evidence.items())),
        "by_retrieval_eligible": dict(sorted(retrieval.items())),
        "ocr_chunks": ocr_chunks,
        "reference_points": reference_points,
        "missing_payload_fields": dict(sorted(missing_fields.items())),
        "my_work_not_evidence": my_work_not_evidence,
        "future_chapters_not_evidence": future_chapters_not_evidence,
        "literature_is_evidence": literature_is_evidence,
    }


class PilotCheck(NamedTuple):
    """One inspection check.

    ``required`` distinguishes invariants (which must hold for the collection to be
    usable: no reference chunk, complete payload) from coverage observations (which
    report whether the sample happens to contain a role or an OCR chunk).
    """

    label: str
    ok: bool
    detail: str
    required: bool = True

    def render(self) -> str:
        """``[ok]`` / ``[FAIL]`` for invariants, ``[--]`` for missing coverage."""
        if self.ok:
            marker = "ok"
        else:
            marker = "FAIL" if self.required else "--"
        return f"  [{marker}] {self.label}: {self.detail}"


def check_pilot(stats: dict[str, Any]) -> list[PilotCheck]:
    """Evaluate the §26 pilot checks against collection statistics."""
    roles = stats["by_corpus_role"]
    content_types = stats["by_content_type"]
    evidence = stats["by_evidence_eligible"]
    checks: list[PilotCheck] = [
        PilotCheck(
            "no reference chunk indexed",
            stats["reference_points"] == 0,
            f"{stats['reference_points']} reference points in the normal collection",
        ),
        PilotCheck(
            "required payload fields present",
            not stats["missing_payload_fields"],
            ", ".join(stats["missing_payload_fields"]) or "all payload fields present",
        ),
        PilotCheck(
            "only literature is evidence-eligible",
            roles.get("literature", 0) == evidence.get("True", 0),
            f"evidence=true for {evidence.get('True', 0)} of {stats['total']} points "
            f"(literature points: {roles.get('literature', 0)})",
        ),
        PilotCheck(
            "my_work is retrievable but not evidence",
            roles.get("my_work", 0) == 0 or stats["my_work_not_evidence"],
            f"{roles.get('my_work', 0)} my_work points",
        ),
        PilotCheck(
            "future_chapters is retrievable but not evidence",
            roles.get("future_chapters", 0) == 0 or stats["future_chapters_not_evidence"],
            f"{roles.get('future_chapters', 0)} future_chapters points",
        ),
        PilotCheck(
            "literature is evidence-eligible",
            roles.get("literature", 0) == 0 or stats["literature_is_evidence"],
            f"{roles.get('literature', 0)} literature points",
        ),
        PilotCheck("my_work present", roles.get("my_work", 0) > 0, f"{roles.get('my_work', 0)} points", False),
        PilotCheck(
            "future_chapters present",
            roles.get("future_chapters", 0) > 0,
            f"{roles.get('future_chapters', 0)} points",
            False,
        ),
        PilotCheck(
            "OCR chunk present",
            stats["ocr_chunks"] > 0,
            f"{stats['ocr_chunks']} points with ocr_involved=true",
            False,
        ),
        PilotCheck(
            "content types indexed",
            "references" not in content_types,
            f"content types: {', '.join(content_types) or 'none'}",
        ),
    ]
    return checks


def render(collection: str, stats: dict[str, Any], samples: list[dict[str, Any]], *, show_text: bool) -> str:
    """Render the inspection output."""
    lines = [
        "RAG-008 collection inspection",
        "=============================",
        "",
        f"collection:            {collection}",
        f"points:                {stats['total']}",
        "",
        "by corpus role:",
    ]
    for name, count in stats["by_corpus_role"].items():
        lines.append(f"  {name:<16} {count}")
    lines.append("by content type:")
    for name, count in stats["by_content_type"].items():
        lines.append(f"  {name:<16} {count}")
    lines.append(f"evidence_eligible:     {stats['by_evidence_eligible']}")
    lines.append(f"retrieval_eligible:    {stats['by_retrieval_eligible']}")
    lines.append(f"ocr chunks:            {stats['ocr_chunks']}")
    lines.append(f"reference points:      {stats['reference_points']}")
    if stats["missing_payload_fields"]:
        lines.append(f"missing payload fields: {stats['missing_payload_fields']}")

    lines.append("")
    lines.append("sample points")
    lines.append("-------------")
    for payload in samples:
        lines.append(f"  chunk {payload.get('chunk_id')} ({payload.get('filename')})")
        lines.append(
            "    role "
            f"{payload.get('corpus_role')} | evidence {payload.get('evidence_eligible')} | "
            f"retrieval {payload.get('retrieval_eligible')} | type {payload.get('content_type')}"
        )
        lines.append(
            f"    pages {payload.get('page_start')}-{payload.get('page_end')} | "
            f"ocr {payload.get('ocr_involved')} | tokens {payload.get('token_count')} | "
            f"sources {payload.get('text_sources')}"
        )
        if show_text:
            lines.append(f"    text: {_preview(str(payload.get('text') or ''))}")
            lines.append(f"    retrieval_text: {_preview(str(payload.get('retrieval_text') or ''))}")
    checks = check_pilot(stats)
    lines.append("")
    lines.append("pilot checks")
    lines.append("------------")
    for check in checks:
        lines.append(check.render())
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, *, client_factory=None) -> int:
    """Run the inspection CLI."""
    args = _build_parser().parse_args(argv)
    if not args.qdrant_path and not args.qdrant_url:
        print("Error: configure --qdrant-url or --qdrant-path", file=sys.stderr)
        return 2
    try:
        client = (client_factory or build_client)(
            url=args.qdrant_url, api_key=args.qdrant_api_key, path=args.qdrant_path
        )
    except QdrantConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    try:
        count_points(client, args.collection)
    except Exception as exc:
        print(f"Error: collection {args.collection!r} is not readable: {exc}", file=sys.stderr)
        return 3

    stats = collect_statistics(client, args.collection)
    samples = sample_payloads(client, args.collection, limit=args.limit)
    text = render(args.collection, stats, samples, show_text=args.show_text)
    sys.stdout.write(text)
    checks = check_pilot(stats)
    exit_code = 0 if all(check.ok for check in checks if check.required) else 1
    close = getattr(client, "close", None)
    if callable(close):
        close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
