"""Retrieval evaluation and latency benchmark (RAG-009).

    python -m app.retrieval.evaluate --out docs/rag009-evaluation.txt --latency-out docs/rag009-latency.txt

Runs the manual evaluation set (``app.retrieval.evaluation_set``) twice - dense-only
and dense + reranker - and reports:

* Hit@5 / Hit@10 / MRR@10 for queries with a gold document;
* rank movement between the two stages, plus the worst reranking regressions;
* evidence-leakage checks (mode leakage, REFERENCES leakage);
* the required example queries verbatim;
* a cold/warm latency benchmark.

Pre-flight: every gold needle is verified to match at least one document in the
collection, so a mistyped expectation is reported instead of being scored as a
retrieval failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.config import get_settings
from app.retrieval.evaluation_set import EVAL_QUERIES, EvalQuery
from app.retrieval.models import Candidate, RetrievalMode, RetrievalResponse
from app.retrieval.service import RetrievalService

#: Retrieval depth used for metrics, so Hit@10 is meaningful.
EVAL_CANDIDATE_K = 25
EVAL_FINAL_K = 10

#: Queries the brief requires verbatim (§28).
REQUIRED_EXAMPLES: tuple[tuple[str, RetrievalMode], ...] = (
    ("artificial intelligence governance in higher education", RetrievalMode.EVIDENCE),
    ("virtual reality student engagement higher education", RetrievalMode.EVIDENCE),
    ("adaptive learning higher education", RetrievalMode.EVIDENCE),
    ("generative AI academic integrity", RetrievalMode.EVIDENCE),
    ("design science research methodology", RetrievalMode.EVIDENCE),
    ("what have I written about the historical development of AI?", RetrievalMode.MY_WORK),
    ("what did my supervisor say about methodology?", RetrievalMode.SUPERVISION),
)

#: Queries that deliberately tempt leakage from non-evidence material.
LEAKAGE_QUERIES: tuple[tuple[str, RetrievalMode], ...] = (
    ("my own draft arguments about AI and VR in higher education", RetrievalMode.EVIDENCE),
    ("the LATTICE framework for AI and VR", RetrievalMode.EVIDENCE),
    ("supervisor feedback and APR form guidance", RetrievalMode.EVIDENCE),
    ("academic writing rules and plagiarism policy", RetrievalMode.EVIDENCE),
    ("future chapters guidance policy for the thesis", RetrievalMode.EVIDENCE),
    ("my summaries of the literature", RetrievalMode.EVIDENCE),
)


def _fold(text: str | None) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFKC", str(text)).casefold()


@dataclass
class QueryOutcome:
    """Result of running one evaluation query in both configurations."""

    entry: EvalQuery
    dense: RetrievalResponse
    reranked: RetrievalResponse

    def relevant(self, candidate: Candidate) -> bool:
        """Whether a returned candidate satisfies the entry's gold expectations."""
        title = _fold(candidate.title)
        filename = _fold(candidate.filename)
        return any(needle in title for needle in self.entry.gold_titles) or any(
            needle in filename for needle in self.entry.gold_filenames
        )

    def first_relevant_rank(self, response: RetrievalResponse) -> int | None:
        """1-based rank of the first relevant result, or ``None``."""
        for rank, candidate in enumerate(response.results, start=1):
            if self.relevant(candidate):
                return rank
        return None


def hit_at(ranks: Sequence[int | None], k: int) -> float:
    """Share of queries whose first relevant result is within rank ``k``."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mrr_at(ranks: Sequence[int | None], k: int = 10) -> float:
    """Mean reciprocal rank of the first relevant result within rank ``k``."""
    if not ranks:
        return 0.0
    total = 0.0
    for rank in ranks:
        if rank is not None and rank <= k:
            total += 1.0 / rank
    return total / len(ranks)


def _document_needle_index(service: RetrievalService) -> dict[str, int]:
    """Count documents in the collection matching each gold needle."""
    payloads = service.retriever.client.scroll(
        collection_name=service.collection,
        limit=20000,
        with_payload=["title", "filename"],
        with_vectors=False,
    )[0]
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for point in payloads:
        payload = point.payload or {}
        title = _fold(payload.get("title"))
        filename = _fold(payload.get("filename"))
        key = (title, filename)
        if key in seen:
            continue
        seen.add(key)
        for entry in EVAL_QUERIES:
            for needle in entry.gold_titles:
                if needle in title:
                    counts[needle] += 1
            for needle in entry.gold_filenames:
                if needle in filename:
                    counts[needle] += 1
    return dict(counts)


def run_query(
    service: RetrievalService,
    entry: EvalQuery,
    *,
    candidate_k: int = EVAL_CANDIDATE_K,
    final_k: int = EVAL_FINAL_K,
) -> QueryOutcome:
    """Run one query dense-only and reranked, with identical settings."""
    dense = service.retrieve(
        entry.query,
        mode=entry.mode,
        candidate_k=candidate_k,
        final_k=final_k,
        rerank=False,
    )
    reranked = service.retrieve(
        entry.query,
        mode=entry.mode,
        candidate_k=candidate_k,
        final_k=final_k,
        rerank=True,
    )
    return QueryOutcome(entry=entry, dense=dense, reranked=reranked)


def _rank_map(response: RetrievalResponse) -> dict[str, int]:
    return {candidate.chunk_id: rank for rank, candidate in enumerate(response.results, start=1)}


def format_evaluation(
    outcomes: Sequence[QueryOutcome],
    *,
    needle_counts: dict[str, int],
    leakage: Sequence[tuple[str, RetrievalResponse, str]],
    latency_lines: Sequence[str],
    service: RetrievalService,
) -> str:
    """Render the whole evaluation report as plain text."""
    spec = service.embedder.spec
    reranker_spec = service.reranker.spec if service.reranker is not None else None
    lines: list[str] = []
    add = lines.append

    add("RAG-009 retrieval evaluation")
    add("=" * 78)
    add(f"collection        : {service.collection} ({service.retriever.point_count()} points)")
    add(f"embedding model   : {spec.model_name} rev={spec.model_revision} dim={spec.dimension} device={spec.device}")
    if reranker_spec is not None:
        add(
            f"reranker          : {reranker_spec.model_name} rev={reranker_spec.model_revision} "
            f"device={reranker_spec.device} batch={reranker_spec.batch_size} max_length={reranker_spec.max_length}"
        )
    else:
        add("reranker          : disabled")
    add(f"candidate_k       : {EVAL_CANDIDATE_K}   final_k: {EVAL_FINAL_K}")
    add(f"queries           : {len(outcomes)}")
    add("")

    # --- gold-needle pre-flight ---
    add("Gold-needle pre-flight (typo protection)")
    add("-" * 78)
    missing = [
        needle
        for entry in EVAL_QUERIES
        for needle in (*entry.gold_titles, *entry.gold_filenames)
        if needle_counts.get(needle, 0) == 0
    ]
    add(f"distinct gold needles : {len(gold_needles_all := {n for e in EVAL_QUERIES for n in (*e.gold_titles, *e.gold_filenames)})}")
    add(f"needles matching at least one document : {len(needle_counts)}")
    add(f"needles matching no document : {len(missing)}")
    for needle in missing:
        add(f"  UNMATCHED: {needle}")
    add("")

    # --- metrics ---
    gold = [outcome for outcome in outcomes if not outcome.entry.is_exploratory]
    dense_ranks = [outcome.first_relevant_rank(outcome.dense) for outcome in gold]
    rerank_ranks = [outcome.first_relevant_rank(outcome.reranked) for outcome in gold]
    add("Metrics on queries with a gold document")
    add("-" * 78)
    add(f"{'metric':<12}{'dense-only':>14}{'dense+rerank':>16}")
    for label, value_dense, value_rerank in (
        ("Hit@5", hit_at(dense_ranks, 5), hit_at(rerank_ranks, 5)),
        ("Hit@10", hit_at(dense_ranks, 10), hit_at(rerank_ranks, 10)),
        ("MRR@10", mrr_at(dense_ranks), mrr_at(rerank_ranks)),
    ):
        add(f"{label:<12}{value_dense:>14.3f}{value_rerank:>16.3f}")
    add(f"{'n_queries':<12}{len(gold):>14}{len(gold):>16}")
    add("")
    add("Note: document-level judgment (was a gold document in the top k), not passage-level.")
    add("")

    # --- per query ---
    add("Per-query comparison")
    add("-" * 78)
    improved = unchanged = worsened = 0
    for outcome in outcomes:
        dense_rank = outcome.first_relevant_rank(outcome.dense)
        rerank_rank = outcome.first_relevant_rank(outcome.reranked)
        if outcome.entry.is_exploratory:
            movement = "exploratory"
        elif dense_rank == rerank_rank:
            movement = "unchanged"
            unchanged += 1
        elif rerank_rank is not None and (dense_rank is None or rerank_rank < dense_rank):
            movement = f"improved {dense_rank}->{rerank_rank}"
            improved += 1
        else:
            movement = f"worsened {dense_rank}->{rerank_rank}"
            worsened += 1
        add(
            f"[{outcome.entry.mode.value:<11}] {outcome.entry.query[:70]:<70} "
            f"gold={'no ' if outcome.entry.is_exploratory else 'yes'} "
            f"dense_rank={str(dense_rank):>4} rerank_rank={str(rerank_rank):>4} {movement}"
        )
        add(f"    category: {outcome.entry.category}   {outcome.entry.note}")
    add("")
    add(f"rank movement on gold queries: improved={improved} unchanged={unchanged} worsened={worsened}")
    add("")

    # --- reranking regressions ---
    add("Reranking regressions (gold query where reranking pushed the gold document down)")
    add("-" * 78)
    regressions = []
    for outcome in gold:
        dense_rank = outcome.first_relevant_rank(outcome.dense)
        rerank_rank = outcome.first_relevant_rank(outcome.reranked)
        if rerank_rank is None or (dense_rank is not None and rerank_rank > dense_rank):
            regressions.append((outcome, dense_rank, rerank_rank))
    if not regressions:
        add("none")
    for outcome, dense_rank, rerank_rank in regressions:
        add(f"query: {outcome.entry.query}")
        add(f"  mode={outcome.entry.mode.value} dense_rank={dense_rank} rerank_rank={rerank_rank}")
        add("  dense top 5:")
        for candidate in outcome.dense.results[:5]:
            add(f"    {candidate.final_rank}. dense={candidate.dense_score:.4f} {(candidate.title or candidate.filename)[:64]}")
        add("  reranked top 5:")
        for candidate in outcome.reranked.results[:5]:
            add(
                f"    {candidate.final_rank}. dense={candidate.dense_score:.4f} "
                f"rerank={candidate.rerank_score if candidate.rerank_score is None else round(candidate.rerank_score, 4)} "
                f"{(candidate.title or candidate.filename)[:64]}"
            )
    add("")

    # --- rank churn across the whole list ---
    add("Rank churn (returned order, dense vs reranked)")
    add("-" * 78)
    churn_counts: Counter[str] = Counter()
    for outcome in outcomes:
        dense_map = _rank_map(outcome.dense)
        rerank_map = _rank_map(outcome.reranked)
        for chunk_id, rank in rerank_map.items():
            previous = dense_map.get(chunk_id)
            if previous is None:
                churn_counts["entered_top_k"] += 1
            elif previous == rank:
                churn_counts["same_rank"] += 1
            else:
                churn_counts["moved"] += 1
        for chunk_id in dense_map:
            if chunk_id not in rerank_map:
                churn_counts["dropped_out_of_top_k"] += 1
    for key in ("same_rank", "moved", "entered_top_k", "dropped_out_of_top_k"):
        add(f"  {key:<20} {churn_counts[key]}")
    add("")

    # --- evidence leakage ---
    add("Evidence-mode leakage checks")
    add("-" * 78)
    leaks = 0
    for query, response, note in leakage:
        bad = [
            candidate
            for candidate in response.results
            if not candidate.evidence_eligible or candidate.content_type == "references"
        ]
        references = [
            candidate for candidate in response.results if candidate.content_type == "references"
        ]
        leaks += len(bad)
        add(
            f"query: {query[:70]:<70} results={len(response.results)} "
            f"non_evidence={len(bad)} references={len(references)}"
        )
        add(f"    {note}")
        for candidate in bad[:3]:
            add(
                f"    LEAK role={candidate.corpus_role} evidence={candidate.evidence_eligible} "
                f"{candidate.filename}"
            )
    add("")
    add(f"total non-evidence results under EVIDENCE mode: {leaks}")
    add("")

    # --- required examples ---
    add("Required example queries (§28)")
    add("-" * 78)
    for query, mode in REQUIRED_EXAMPLES:
        response = service.retrieve(query, mode=mode, candidate_k=EVAL_CANDIDATE_K, final_k=8)
        add(f"query   : {query}")
        add(f"mode    : {mode.value}")
        add(
            f"audit   : candidate_k={response.audit.candidate_k} final_k={response.audit.final_k} "
            f"rerank={response.audit.rerank_enabled}"
        )
        if not response.results:
            add("  (no results)")
        for candidate in response.results:
            title = (candidate.title or candidate.filename or "(untitled)")[:60]
            add(
                f"  {candidate.final_rank}. dense={candidate.dense_score:.4f} "
                f"rerank={candidate.rerank_score if candidate.rerank_score is None else round(candidate.rerank_score, 4)} "
                f"[{candidate.corpus_role}] ev={int(candidate.evidence_eligible)} "
                f"{(candidate.first_author or '?')[:14]}/{candidate.year} p.{candidate.pages} {title}"
            )
        add("")

    # --- latency ---
    add("Latency")
    add("-" * 78)
    for line in latency_lines:
        add(line)

    # --- drop accounting ---
    add("")
    add("Drop accounting across the evaluation set")
    add("-" * 78)
    totals: Counter[str] = Counter()
    for outcome in outcomes:
        for item in outcome.reranked.dropped:
            totals[item.reason] += 1
    for reason, count in sorted(totals.items()):
        add(f"  {reason:<24} {count}")

    # --- duplicate-copy concentration ---
    add("")
    add("Duplicate-copy concentration in returned results")
    add("-" * 78)
    add("How many returned results share one document id, and how many distinct")
    add("copies of one work (same title) can occupy the top list.")
    worst: list[tuple[int, str, int, int]] = []
    for outcome in outcomes:
        per_document = Counter(candidate.document_id for candidate in outcome.reranked.results)
        per_title = Counter(
            _fold(candidate.title or candidate.filename) for candidate in outcome.reranked.results
        )
        worst.append(
            (
                max(per_title.values(), default=0),
                outcome.entry.query,
                max(per_document.values(), default=0),
                len(per_title),
            )
        )
    worst.sort(reverse=True)
    for title_max, query, document_max, distinct in worst[:5]:
        add(
            f"  max_per_title={title_max} max_per_document={document_max} "
            f"distinct_titles={distinct}  {query[:56]}"
        )
    return "\n".join(lines)


def latency_benchmark(
    service: RetrievalService,
    queries: Sequence[str],
    *,
    candidate_k: int = 25,
    final_k: int = 8,
) -> list[str]:
    """Measure cold and warm latency for embed/search/rerank/total."""
    lines: list[str] = []
    warm = service.warm_up()
    lines.append(f"warm-up: embedder={warm['embedder_seconds']:.3f}s reranker={warm.get('reranker_seconds', 0.0):.3f}s")
    lines.append("")

    # Cold: one query with the reranker model evicted is not practical without
    # reloading weights, so "cold" = first query after the process started, before
    # any caching of tokenizer/allocator state; "warm" = median of the rest.
    per_query: list[dict[str, float]] = []
    for index, query in enumerate(queries):
        response = service.retrieve(query, mode=RetrievalMode.EVIDENCE, candidate_k=candidate_k, final_k=final_k)
        per_query.append(
            {
                "index": index,
                "embed": response.stats.embedding_seconds,
                "search": response.stats.search_seconds,
                "rerank": response.stats.rerank_seconds,
                "total": response.stats.total_seconds,
            }
        )

    def median(values: Sequence[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2

    rest = per_query[1:] or per_query
    lines.append(f"queries measured: {len(per_query)}  (candidate_k={candidate_k} final_k={final_k})")
    lines.append(f"{'stage':<10}{'first query (s)':>18}{'warm median (s)':>18}{'warm max (s)':>14}")
    for key, label in (
        ("embed", "embed"),
        ("search", "search"),
        ("rerank", "rerank"),
        ("total", "total"),
    ):
        lines.append(
            f"{label:<10}{per_query[0][key]:>18.4f}{median([row[key] for row in rest]):>18.4f}"
            f"{max(row[key] for row in rest):>14.4f}"
        )
    lines.append("")
    lines.append("Per-query detail (reranked):")
    for row, query in zip(per_query, queries):
        lines.append(
            f"  {row['index'] + 1:>2}. embed={row['embed']:.4f}s search={row['search']:.4f}s "
            f"rerank={row['rerank']:.4f}s total={row['total']:.4f}s  {query[:60]}"
        )

    dense_only = None
    if queries:
        started = time.perf_counter()
        service.retrieve(queries[0], mode=RetrievalMode.EVIDENCE, candidate_k=candidate_k, final_k=final_k, rerank=False)
        dense_only = time.perf_counter() - started
    if dense_only is not None:
        lines.append("")
        lines.append(f"dense-only total for one query (no rerank): {dense_only:.4f}s")
    return lines


def build_service(device: str | None, *, qdrant_path: str | None = None, collection: str | None = None) -> RetrievalService:
    """Build the real service (BGE-M3 + BGE reranker) for evaluation."""
    from app.embedding.bge_m3 import BgeM3Embedder
    from app.reranking.bge_reranker import BgeReranker

    settings = get_settings()
    embedder = BgeM3Embedder(
        model_name=settings.embedding_model,
        device=device or settings.embedding_device,
        local_files_only=settings.embedding_local_files_only,
    )
    reranker = BgeReranker(
        model_name=settings.reranker_model,
        device=device or settings.reranker_device,
        batch_size=settings.reranker_batch_size,
        local_files_only=settings.embedding_local_files_only,
    )
    return RetrievalService(
        embedder=embedder,
        reranker=reranker,
        collection=collection or settings.qdrant_collection,
        qdrant_path=qdrant_path or (settings.qdrant_path or None),
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        candidate_k=EVAL_CANDIDATE_K,
        final_k=EVAL_FINAL_K,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.retrieval.evaluate",
        description="Run the RAG-009 retrieval evaluation and latency benchmark.",
    )
    parser.add_argument("--out", default="docs/rag009-evaluation.txt")
    parser.add_argument("--latency-out", default="docs/rag009-latency.txt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--qdrant-path", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--json-out", default=None, help="optional JSON dump of every query/result")
    parser.add_argument("--skip-latency", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    service = build_service(args.device, qdrant_path=args.qdrant_path, collection=args.collection)
    try:
        needle_counts = _document_needle_index(service)
        outcomes = [run_query(service, entry) for entry in EVAL_QUERIES]

        leakage: list[tuple[str, RetrievalResponse, str]] = []
        for query, mode in LEAKAGE_QUERIES:
            response = service.retrieve(query, mode=mode, candidate_k=EVAL_CANDIDATE_K, final_k=8, rerank=True)
            leakage.append((query, response, f"mode={mode.value}"))

        latency_lines: list[str] = []
        if not args.skip_latency:
            latency_queries = [entry.query for entry in EVAL_QUERIES[:10]]
            latency_lines = latency_benchmark(service, latency_queries)

        report = format_evaluation(
            outcomes,
            needle_counts=needle_counts,
            leakage=leakage,
            latency_lines=latency_lines,
            service=service,
        )
    finally:
        service.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")
    if latency_lines:
        Path(args.latency_out).write_text("\n".join(latency_lines) + "\n", encoding="utf-8")
        print(f"wrote {args.latency_out}")
    if args.json_out:
        payload = [
            {
                "query": outcome.entry.query,
                "mode": outcome.entry.mode.value,
                "category": outcome.entry.category,
                "exploratory": outcome.entry.is_exploratory,
                "dense": [
                    {
                        "rank": candidate.final_rank,
                        "chunk_id": candidate.chunk_id,
                        "document_id": candidate.document_id,
                        "filename": candidate.filename,
                        "title": candidate.title,
                        "corpus_role": candidate.corpus_role,
                        "evidence_eligible": candidate.evidence_eligible,
                        "dense_score": candidate.dense_score,
                        "pages": candidate.pages,
                        "section": candidate.section_title,
                        "relevant": outcome.relevant(candidate),
                    }
                    for candidate in outcome.dense.results
                ],
                "reranked": [
                    {
                        "rank": candidate.final_rank,
                        "chunk_id": candidate.chunk_id,
                        "document_id": candidate.document_id,
                        "filename": candidate.filename,
                        "title": candidate.title,
                        "corpus_role": candidate.corpus_role,
                        "evidence_eligible": candidate.evidence_eligible,
                        "dense_score": candidate.dense_score,
                        "rerank_score": candidate.rerank_score,
                        "pages": candidate.pages,
                        "section": candidate.section_title,
                        "relevant": outcome.relevant(candidate),
                    }
                    for candidate in outcome.reranked.results
                ],
            }
            for outcome in outcomes
        ]
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
