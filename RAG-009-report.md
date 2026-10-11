# RAG-009 — Retrieval and reranking

Status: **complete** — two-stage retrieval over `phd_rag_chunks_v1`, evidence policy
enforced at Qdrant level and re-verified locally, local BGE cross-encoder rerank,
deterministic diversity, a retrieval-time bibliography guard (§25), 98 new tests, real
evaluation on 36 corpus queries.

Scope guard: RAG-009 returns ranked chunks with provenance. It does not generate
prose, call Qwen, synthesise claims or render citations, and it does not begin
RAG-010.

---

## 1. Files created

| Path | Purpose |
|------|---------|
| [app/model_loading.py](../app/model_loading.py) | Shared local-model policy: `resolve_device`, `resolve_snapshot`, `library_version` (used by the RAG-008 embedder and the RAG-009 reranker) |
| [app/reranking/__init__.py](../app/reranking/__init__.py) | Reranking package exports |
| [app/reranking/models.py](../app/reranking/models.py) | `RerankerSpec`, `RerankInput`, `RerankScore`, `Reranker` protocol, `RerankerLoadError` |
| [app/reranking/bge_reranker.py](../app/reranking/bge_reranker.py) | `BgeReranker` — lazy `CrossEncoder` wrapper for `BAAI/bge-reranker-v2-m3` |
| [app/retrieval/__init__.py](../app/retrieval/__init__.py) | Retrieval package exports |
| [app/retrieval/models.py](../app/retrieval/models.py) | `RetrievalMode`, `RetrievalFilters`, `Candidate`, `DroppedCandidate`, `QueryAudit`, `RetrievalStats`, `RetrievalResponse` |
| [app/retrieval/query.py](../app/retrieval/query.py) | Conservative `normalize_query` (NFKC, whitespace collapse, case preserved) |
| [app/retrieval/filters.py](../app/retrieval/filters.py) | Qdrant filter construction + the defensive payload predicate |
| [app/retrieval/dense.py](../app/retrieval/dense.py) | `DenseRetriever` (query embedding, geometry checks, `query_points`) and payload → `Candidate` conversion |
| [app/retrieval/diversity.py](../app/retrieval/diversity.py) | Near-duplicate suppression, work-level result cap, deterministic text/work keys |
| [app/retrieval/service.py](../app/retrieval/service.py) | `RetrievalService.retrieve(...)` — the pipeline |
| [app/retrieval/inspect.py](../app/retrieval/inspect.py) | `python -m app.retrieval.inspect` CLI |
| [app/retrieval/evaluate.py](../app/retrieval/evaluate.py) | Evaluation + leakage + latency harness |
| [app/retrieval/evaluation_set.py](../app/retrieval/evaluation_set.py) | The manual 36-query evaluation set with gold needles |
| [tests/retrieval_support.py](../tests/retrieval_support.py) | Controllable embedder, deterministic reranker double, seeded local/in-memory collections |
| [tests/test_retrieval_query.py](../tests/test_retrieval_query.py) | 7 tests — query handling and query embedding |
| [tests/test_retrieval_filters.py](../tests/test_retrieval_filters.py) | 12 tests — modes and explicit filters |
| [tests/test_retrieval_safety.py](../tests/test_retrieval_safety.py) | 7 tests — evidence-mode safety and defensive rejection |
| [tests/test_retrieval_rerank.py](../tests/test_retrieval_rerank.py) | 9 tests — reranking, score handling, audit |
| [tests/test_retrieval_diversity.py](../tests/test_retrieval_diversity.py) | 10 tests — duplicates, work cap, work keys |
| [tests/test_retrieval_determinism.py](../tests/test_retrieval_determinism.py) | 9 tests — determinism, filter order, CLI |
| [tests/test_retrieval_rounds.py](../tests/test_retrieval_rounds.py) | 6 tests — widening rounds when duplicates starve the list |
| [tests/test_retrieval_evaluation.py](../tests/test_retrieval_evaluation.py) | 7 tests — evaluation set and metrics |
| [app/retrieval/bibliography.py](../app/retrieval/bibliography.py) | High-confidence bibliography-like detector used as a retrieval-time guard (§27) |
| [tests/test_retrieval_bibliography.py](../tests/test_retrieval_bibliography.py) | 31 tests — detector rules, mode policy, audit trail, opt-in |
| [docs/rag009-bibliography-scan.txt](../docs/rag009-bibliography-scan.txt) | Corpus-wide detection scan: the known 142, the misses, the prose sweep |
| [docs/rag009-bibliography-manual-review.txt](../docs/rag009-bibliography-manual-review.txt) | 25 detected + 25 not-detected chunks with reviewer verdicts |
| [docs/rag009-bibliography-impact.txt](../docs/rag009-bibliography-impact.txt) | The MY_WORK query and the five required queries, guard on vs off |
| [docs/rag009-evaluation.txt](../docs/rag009-evaluation.txt) | Real evaluation, leakage checks, required examples, latency |
| [docs/rag009-evaluation.json](../docs/rag009-evaluation.json) | Machine-readable per-query results (both stages) |
| [docs/rag009-latency.txt](../docs/rag009-latency.txt) | Warm/cold latency table |
| [docs/rag009-reranker-benchmark.txt](../docs/rag009-reranker-benchmark.txt) | Reranker batch sizes, VRAM, Qdrant local-mode filter latency |
| [docs/rag009-required-example-texts.txt](../docs/rag009-required-example-texts.txt) | Top-8 texts of the five required evidence queries (classification input) |
| [docs/rag009-non-evidence-examples.txt](../docs/rag009-non-evidence-examples.txt) | MY_WORK / SUPERVISION / GUIDANCE example queries |
| [docs/rag009-document-inventory.txt](../docs/rag009-document-inventory.txt) | 465 indexed documents: role, year, author, title, filename, document id |

## 2. Files modified

| Path | Change |
|------|--------|
| [app/config.py](../app/config.py) | RAG-009 settings: `reranker_model`, `reranker_device`, `reranker_batch_size`, `retrieval_mode`, `retrieval_candidate_k`, `retrieval_final_k`, `retrieval_max_chunks_per_document`, `retrieval_duplicate_overlap`, `retrieval_max_rounds`, `retrieval_exclude_bibliography` (+ `PHD_RAG_*` parsing and defaults) |
| [app/retrieval/models.py](../app/retrieval/models.py) | Candidate/audit/stats fields for the bibliography guard (§27) |
| [app/retrieval/filters.py](../app/retrieval/filters.py) | `REASON_BIBLIOGRAPHY` + the per-mode bibliography policy |
| [app/retrieval/dense.py](../app/retrieval/dense.py) | The verdict is computed once, when a point becomes a candidate |
| [app/retrieval/service.py](../app/retrieval/service.py) | Guard wiring, `bibliography_like` drop reason and counters |
| [app/retrieval/inspect.py](../app/retrieval/inspect.py) | `--include-bibliography`, `biblio=` in the result block, drop count in the header |
| [app/embedding/bge_m3.py](../app/embedding/bge_m3.py) | `resolve_device` / `resolve_snapshot` now delegate to `app.model_loading`; the public names, signatures and error types are unchanged, so RAG-008 behaviour and tests are untouched |
| [README.md](../README.md) | Retrieval section + CLI examples |
| [tests/test_retrieval_determinism.py](../tests/test_retrieval_determinism.py) | Covers the `--max-rounds` flag |

Nothing under `data/` was read-modify-written, and `app/models/`, `app/chunking/`,
`app/indexing/` and the RAG-008 report were **not** changed.

## 3. Dependencies added

None. The reranker runs on the already-declared `sentence-transformers` (+`torch`,
`transformers`, `huggingface-hub`); `FlagEmbedding` was deliberately not added.
`pyproject.toml` is unchanged. The only new artifact is the 2.29 GB model snapshot in
the Hugging Face cache.

## 4. Reranker model/revision

| Item | Value |
|------|-------|
| Model | `BAAI/bge-reranker-v2-m3` |
| Revision | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (resolved from the snapshot directory) |
| Files | `model.safetensors` (2.29 GB), `sentencepiece.bpe.model`, `tokenizer.json`, configs |
| Library | `sentence-transformers` 6.1.0 (`CrossEncoder`) |
| `max_length` | 8192 (model default; **no truncation** of any benchmarked candidate — measured, see §22) |
| Output | one sigmoid relevance score per `(query, passage)` pair |

`RerankerSpec` records model, revision, device, batch size, library version and
`max_length`; the resolution error class is `RerankerLoadError`, and an explicitly
requested CUDA device that is unavailable fails instead of silently falling back.

## 5. Device and batch size

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5090 (32 GiB), driver 616.56 |
| torch | 2.14.1+cu130, CUDA runtime 13.0, cuDNN 92400 |
| Device policy | `auto` → `cuda`; `cpu` also supported and tested |
| Production batch size | **16** (see §22 for the measurement that selects it) |
| Peak VRAM (reranker, batch 16) | 4.93 GiB allocated / 5.36 GiB reserved |

## 6. Retrieval architecture

```
query
  ↓  normalize_query()               NFKC + whitespace collapse, case preserved
  ↓  BgeM3Embedder.embed_query()      the RAG-008 wrapper, revision pinned, L2-normalized
  ↓  Qdrant query_points()            candidate_k=25, mode/filter pushed into the search
  ↓  local payload re-check           mode + filters + REFERENCES + retrieval_eligible
  ↓  BgeReranker.rerank()             retrieval_text scored against the query
  ↓  optional min score floors        off by default
  ↓  near-duplicate suppression       identical text (incl. copies), adjacent overlap
  ↓  work cap (3 chunks)              title-based work key, not file-based
  ↓  bibliography guard (§27)         high-confidence reference lists withheld
  ↓  optional widening round          only when duplicates starve the list (≤3 rounds)
  ↓  final_k results                  both scores, full provenance, audit record
```

* Query embedding is never re-implemented: `DenseRetriever.embed_query` calls
  `BgeM3Embedder.embed_query`, then checks the dimension against the collection and
  the L2 norm (`|norm-1| ≤ 1e-3`), failing loudly on a mismatch.
* Qdrant is queried with `query_points` (qdrant-client 1.19.1 has no `search`),
  `with_payload=True`, `with_vectors=False`.
* The reranker consumes `retrieval_text` (the same text that was embedded, synthetic
  `[Document: …]`/`[Section: …]` labels included), never `text`.
* Rerank scores from different widening rounds come from the same model and are
  therefore comparable; scores are never fused with dense scores.

## 7. Retrieval modes

| Mode | Filter pushed to Qdrant | Notes |
|------|-------------------------|-------|
| `EVIDENCE` | `evidence_eligible = true` | effectively published literature only |
| `GENERAL` | `retrieval_eligible = true` | every role; the role is preserved in each result |
| `MY_WORK` | `corpus_role = my_work` | the author's own drafts — never evidence |
| `SUPERVISION` | `corpus_role = supervision` | APR forms |
| `GUIDANCE` | `corpus_role ∈ {rules, guidance, future_chapters, summaries}` | scaffolding material |

Modes are filters only — no mode changes scoring, thresholds or result shape.

## 8. Qdrant filter policy

Every query, in every mode, always carries:

* `retrieval_eligible = true` (defensive; RAG-008 only indexed eligible chunks), and
* `content_type != references` via `must_not` (a filter that relies on "the indexer
  did not do X" is an assumption, not a guarantee).

Optional filters, each individually testable: `year_min`, `year_max`, `document_id`,
`file_hash`, `doi`, `corpus_role`, `evidence_eligible`, `ocr_involved`,
`publication_type`, `content_type` (repeated). Year ranges use a Qdrant `Range`,
content types a `MatchAny`, everything else a `MatchValue`.

The bibliography guard (§27) is deliberately *not* a Qdrant filter: it depends on the
chunk text, which is not indexed for that purpose, and adding a payload field would
mean rebuilding the collection. It runs on the candidates the search returned, and its
outcome is recorded as a drop reason.

## 9. Evidence-safety rules

1. The mode condition is sent **into** the vector search, so non-evidence chunks are
   never candidates (asserted by a test that records the `query_filter` handed to
   Qdrant).
2. Everything Qdrant returns is re-checked locally (`rejection_reason`) with recorded
   reasons: `mode_mismatch`, `references_content_type`, `not_retrieval_eligible`,
   `filter_mismatch`.
3. `second-paper-draft-final.docx` (SHA-256 identical, present under `literature/`
   and `my_work/`, resolved to `my_work` by RAG-007) is asserted to be unreachable in
   EVIDENCE mode and reachable in MY_WORK mode.
4. A filter cannot silently change ranking, and a mode cannot be overridden by a
   caller-supplied filter that contradicts it.

## 10. Candidate-k / final-k

Defaults `candidate_k = 25`, `final_k = 8`, configurable per call, per CLI flag and
via `PHD_RAG_RETRIEVAL_CANDIDATE_K` / `PHD_RAG_RETRIEVAL_FINAL_K`. They are not
buried in the implementation: `RetrievalService.__init__` takes both, `retrieve()`
can override both, and the audit record stores the values actually used.

## 11. Diversity rules

* Cap: `max_chunks_per_document = 3` (configurable, `--max-per-document`, `0`/`None`
  disables), applied **after** reranking.
* The cap is applied to a **work**, not a file: the corpus holds several physical
  copies of the same paper under different filenames (and therefore different
  SHA-256 hashes and document ids), so a per-file cap let a single paper fill the
  whole result list. A work key is `title:<NFKC-folded tokens>` when the title has at
  least 3 tokens and 15 characters, otherwise `document:<document_id>` — so short
  generic titles (`Form`, `Book Reviews`, `PRIVACY POLICY`) never merge unrelated
  documents.
* It is a cap, not a quota: several distinct passages of one strong paper are kept
  (up to three), and one chunk per document is never forced.
* Drops are reported: `document_limit` with the work key and the count that hit the
  limit.

## 12. Duplicate suppression

Two deterministic signals, no model, no learned threshold:

| Signal | Scope | Reason label |
|--------|-------|--------------|
| Identical normalised text (NFKC + case-folded alphanumeric tokens, SHA-256) | any document (copies of one work), toggleable | `identical_text` |
| Same document, `|Δchunk_index| ≤ 1`, token containment ≥ 0.85 | one document | `overlapping_neighbour` |

The higher-ranked candidate always wins, the loser is reported with the id it lost to,
and `--no-dedup` disables the pass. Extending identical-text suppression across
documents was a deliberate correction during this task: the corpus really does contain
several copies of some papers, and before the change one query's top 8 contained five
identical passages from five copies of one paper (see §19).

## 13. Score handling

* `dense_score` (cosine, from Qdrant) and `rerank_score` (cross-encoder sigmoid) are
  stored separately and never combined; the report states that they are not on the
  same scale and no pseudo-fusion score exists.
* Ordering uses the rerank score when reranking is enabled, with `dense_rank` then
  `chunk_id` as deterministic tie-breakers; with `--no-rerank` the dense score orders
  the list and `rerank_score` stays `None`.
* If no reranker is configured, `rerank=True` degrades to dense ordering and the audit
  records `rerank_enabled = false` (it does not pretend the reranker ran).
* `min_dense_score` / `min_rerank_score` exist and default to `None`; no threshold was
  invented before evaluation (see §26 for what the evaluation suggests).

## 14. CLI usage

```powershell
python -m app.retrieval.inspect "artificial intelligence governance in higher education"
python -m app.retrieval.inspect "what did my supervisor say about methodology?" --mode supervision
python -m app.retrieval.inspect "what have I written about the historical development of AI?" --mode my_work
python -m app.retrieval.inspect "virtual reality in higher education" --no-rerank --year-min 2019 \
    --content-type body --show-text --show-locators
python -m app.retrieval.inspect "..." --json
```

Flags: `--mode {evidence,general,my_work,supervision,guidance}`, `--candidate-k`,
`--final-k`, `--no-rerank`, `--year-min`, `--year-max`, `--content-type` (repeatable),
`--corpus-role`, `--document-id`, `--file-hash`, `--doi`, `--publication-type`,
`--ocr-only`, `--no-dedup`, `--include-bibliography`, `--max-per-document`,
`--max-rounds`, `--min-dense-score`, `--min-rerank-score`, `--show-text`,
`--show-retrieval-text`, `--show-locators`, `--json`, `--qdrant-path`, `--collection`,
`--device`, `--width`. Default output is a readable block per result (rank, both
scores, title, author/year, pages, section, role, evidence, OCR, `biblio=` when a chunk
is bibliography-like, preview); `--json` is for scripting.

```powershell
# citation/bibliography analysis: return reference lists on purpose
python -m app.retrieval.inspect "references on adaptive learning" --mode my_work --include-bibliography
# environment switch with the same meaning
$env:PHD_RAG_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY = "false"
```

## 15. Evaluation-query set

36 queries (`app/retrieval/evaluation_set.py`), 28 with a gold document and 8 marked
`exploratory`. Gold needles are lowercase title/filename substrings taken from the
real corpus inventory, and the harness verifies every needle matches at least one
document (26/26 matched, 0 unmatched), so a typo cannot masquerade as a retrieval
failure.

| Group | Queries |
|-------|---------|
| EVIDENCE with gold | VR and student engagement, adaptive learning, generative-AI academic integrity, DSR methodology, AI literacy, intelligent tutoring, learning analytics (×2), AI-in-HE discourses/bibliometrics/state of the field, formative and peer assessment, experiential learning, immersive VR (×2 + design framework), generative AI assessments, personalization, VR dental education, GANs, AI for VR review, self-evolving AI tutors |
| EVIDENCE, exploratory | AI governance in higher education, ethical use of AI in universities, implementation barriers, generative-AI policy, personalized learning with knowledge tracing, student engagement online |
| MY_WORK | historical development of AI (gold = filename), own paper on AI/VR tools |
| SUPERVISION | methodology (two wordings; no gold — forms are unstructured) |
| GUIDANCE | referencing rules (two wordings; gold = Reference checklist) |
| GENERAL | VR and AI in higher education (checks role is preserved) |

Where no corpus document is clearly on the topic, the query is marked `exploratory`
rather than given an invented target.

## 16. Dense-only metrics

`candidate_k=25`, `final_k=10`, document-level judgment (was a gold document in the
top k), 28 gold queries, with the bibliography guard on — from
[docs/rag009-evaluation.txt](../docs/rag009-evaluation.txt):

| Metric | Dense-only | Dense-only before §27 |
|--------|-----------|----------------------|
| Hit@5 | 0.857 | 0.893 |
| Hit@10 | **0.964** | 0.929 |
| MRR@10 | 0.843 | 0.869 |

## 17. Reranked metrics

| Metric | Dense-only | Dense + rerank | (before §27: dense / reranked) |
|--------|-----------|----------------|-------------------------------|
| Hit@5 | 0.857 | 0.857 | 0.893 / 0.857 |
| Hit@10 | 0.964 | **0.964** | 0.929 / 0.964 |
| MRR@10 | 0.843 | **0.871** | 0.869 / 0.871 |

Reranking improves Hit@10 and MRR@10 slightly and loses 0.036 on Hit@5: it makes fewer
queries fail outright but occasionally demotes a gold document from within the top 5 to
rank 6–8. With 28 gold queries this is an engineering observation, not a significance
claim.

**Why the dense-only numbers moved once the bibliography guard was added.** The dense
Hit@5 drop is not a retrieval regression: the gold "hit" that disappeared was itself a
reference list. For *artificial intelligence in higher education: the state of the
field* the dense rank-1 result was a bibliography chunk of the correct document
(chunk `2a3b8e76`, `biblio=1`, text starting "test system. Advances in Multimedia.
https://doi.org/10.1155/2022/2156"): reference lists are dense in topical phrases, so
they outrank the paper's prose. With the guard on, that chunk is withheld, the paper's
real prose chunk appears in the dense list (rank 6) and the reranker promotes it to
rank 1 — which is why the same query shows `improved 6->1` and dense Hit@10 rose to
0.964. Dense-only Hit@5/MRR fell because two "hits" were bibliography fragments; the
reranked pipeline, which decides the order that is actually returned, improved.

## 18. Per-query comparison

24 of the 28 gold queries have the gold document at dense rank 1, so the reranker is
mostly confirming the dense order there. Rank movement over the gold queries (with the
bibliography guard on): **improved 2, unchanged 25, worsened 1**.

* `generative AI academic integrity` → gold at rank 1 (dense and reranked).
* `design science research methodology` → gold at rank 1 (dense and reranked).
* `virtual reality student engagement higher education` → gold at rank 1 (three
  distinct passages of the same paper fill ranks 1–3, within the work cap).
* `adaptive learning higher education` → gold at rank 7/8 in both stages.
* `research design framework for classifying virtual reality studies` → rank 1 in both
  stages, and the passage's five copies were reduced to one result.
* `what have I written about the historical development of AI?` (MY_WORK) → the own
  draft at rank 1 (gold needle is the filename, because the draft has no extracted
  title).
* `virtual reality and artificial intelligence in higher education` (GENERAL) → gold
  only at reranked rank 9 (dense: outside the top 10): unfiltered retrieval over all
  roles is visibly harder.
* `what are the rules for referencing and citation style in my thesis?` (GUIDANCE) →
  gold never retrieved (see §19); the second wording “Harvard referencing style
  guidelines” finds it at rank 7.

## 19. Reranking regressions

Only one gold query worsened: `generative adversarial networks image synthesis`
(dense rank 5 → reranked rank 8).

Inspection shows this is not a false match: every candidate in the reranked top 5 is a
GAN / image-synthesis paper (three chunks of *Generative Adversarial Networks (GANs)
for Medical Image Synthesis*, one *Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks*, one AI-for-VR review), and the reranker scores
them within 0.977–0.999 while the seminal *Generative Adversarial Nets* sits just
behind. The gold needle is the seminal paper, so this counts as a regression even
though the returned material is topically correct. The most precise statement is that
the reranker compresses the top of this query into near-identical scores.

Two non-regression issues were found by this evaluation and **fixed**:

1. **Duplicate copies flooding results.** For `design science research methodology` the
   top 25 dense candidates were 19 chunks of one work plus 6 identical passages from
   six copies of another; 5 of the top 8 were the same passage from different copies.
   Fixed by extending identical-text suppression across documents (§12) and by the
   work-level cap (§11). After the fix, no evaluated query returns more than 3 results
   from one work (before: up to 6).
2. **Short result lists when one work dominates.** After those drops, the same query
   returned only 4 of 8 requested results, because the candidate pool held just two
   works. Fixed with bounded widening rounds (§6); the DSR query now returns 8 results
   from 4 works (used 2 rounds), and healthy queries still use exactly one round
   (asserted by a test).

A GUIDANCE-mode gold miss remains and is reported as a genuine limitation:
*Reference checklist* is a table-shaped document with a single chunk, and the phrasing
“rules for referencing and citation style in my thesis” does not bring it into the
first 60 candidates of the 897 GUIDANCE chunks (it appears at rank 32 of the 60
retrieved when the search is widened); a second, more specific phrasing finds it at
rank 7. Nothing was tuned to hide this.

## 20. Manual relevance assessment

The five required evidence queries, top 8 each, classified from the retrieved texts
([docs/rag009-required-example-texts.txt](../docs/rag009-required-example-texts.txt)):

| Query | directly relevant | relevant | weakly relevant | irrelevant |
|-------|------------------:|---------:|----------------:|-----------:|
| artificial intelligence governance in higher education | 3 | 2 | 3 | 0 |
| virtual reality student engagement higher education | 3 | 4 | 1 | 0 |
| adaptive learning higher education | 5 | 3 | 0 | 0 |
| generative AI academic integrity | 5 | 2 | 1 | 0 |
| design science research methodology | 3 | 0 | 3 | 2 |
| **total (40 results)** | **19** | **11** | **8** | **2** |

The classification stands unchanged after the bibliography guard (§27): for all five
queries the returned top 8 are *identical* with the guard on and off
([docs/rag009-bibliography-impact.txt](../docs/rag009-bibliography-impact.txt)), because
the withheld chunks were ranked below the visible results for these queries.

* *Governance* is the query with no gold document, and retrieval is good anyway: ranks
  3, 4 and 8 are explicitly about AI policy, ethics and governance frameworks, rank 5
  quotes AI-governance standards and regulation inside a wider AI-in-HE paper, and rank
  7 is the abstract of the paper whose policy-and-ethics section is at rank 4; ranks 1,
  2 and 6 are about AI in higher-education management.
* *Adaptive learning* and *generative-AI academic integrity* are excellent: every
  result is on topic.
* *DSR methodology* is the weakest: the corpus contains essentially one dedicated DSR
  paper, so ranks 1–3 are its three best chunks and the remaining five slots go to
  weaker material (Radianti's "research design framework" section ×3, a VR game design
  paper, a Turnitin evaluation). Those last two are also exactly where the reranker
  gives near-zero scores (0.059 and 0.047), i.e. the system itself signals that the
  list is exhausted — the argument for a score floor in RAG-010 rather than a
  threshold invented now.
* No obvious irrelevant material dominates any query: 2 of 40 results (5%) are
  irrelevant, both in the thinnest topic.

### Non-evidence modes (§28)

From [docs/rag009-non-evidence-examples.txt](../docs/rag009-non-evidence-examples.txt):

* **MY_WORK**, “what have I written about the historical development of AI?” — rank 1 is
  the correct draft (`Historical-Development-24-02-2025.docx`, the section on the
  historical development and current role of AI in higher education), rank 2 the
  introduction draft. Ranks 3–5, however, are *reference-list* text from the author's own
  drafts (a Radianti citation, a bibliography fragment): still `my_work`, never evidence,
  but not useful as content. See the limitation below.
* **SUPERVISION**, “what did my supervisor say about methodology?” — all five results are
  from the two `supervision` role documents, but those documents are APR *forms*, so the
  hits are form boilerplate and the author's own progress statement rather than supervisor
  commentary on methodology. The reranker scores agree (0.30, 0.09, 0.007, 0.006, 0.004).
  Retrieval behaves correctly; the supervision material itself is form-shaped, which is a
  fact about the corpus, not about filtering.
* **GUIDANCE**, “how should I structure my references and citations?” — writing-rule and
  dissertation-guidance chunks at ranks 1–3, and the *Reference checklist* table at rank
  4 (it is found when the query is about referencing *rules*, and missed when the query
  also asks about “my thesis”).

### Reference lists typed as body (RAG-007 classification, inherited)

A scan of the index for chunks containing at least three citation/DOI-URL markers
(`Available at: http…`, `https://doi.org/…`) finds **142 indexed chunks (1.2% of
11,469)** that are reference lists but are typed `body` (120), `summary_entry` (12),
`conclusion` (9) or `table` (1) — 100 of them `literature`, 23 `my_work`. RAG-007
excluded 3,317 chunks as REFERENCES, so the mechanism works in general; it does not fire
for these. RAG-009 does not change chunking (that would invalidate the RAG-008 index):
the correction is a retrieval-time guard (§25), which detects 133 of these 142 plus 140
more outside the marker scan and withholds them in EVIDENCE/MY_WORK/GENERAL. The
underlying classification gap remains and a RAG-007 follow-up should re-run the
reference-heading detection over the affected documents.

## 21. Evidence leakage tests

Six deliberately tempting queries (own-draft phrasing, the LATTICE framework,
supervisor/APR wording, writing rules and plagiarism, future-chapters guidance, “my
summaries of the literature”) were run in EVIDENCE mode with reranking on:

| Check | Result |
|-------|--------|
| results returned | 8 per query |
| non-evidence results | **0** |
| REFERENCES results | **0** |

Plus unit-level guarantees: the mode condition is present in the Qdrant filter handed
to the client; `second-paper-draft-final.docx` is unreachable in EVIDENCE and reachable
in MY_WORK; every non-evidence role is excluded from EVIDENCE across eight queries; a
synthetic payload with `content_type=references` is rejected with
`references_content_type` even when it claims `evidence_eligible=true`.

Re-run with the bibliography guard on: still **0 non-evidence results** and **0
references**, and the guard itself dropped 112 bibliography-like candidates across the
36 evaluation queries (its own drop reason, counted separately).

## 22. Reranker benchmark (§9 of the brief)

25 real query/passage pairs for `artificial intelligence governance in higher
education` (median 360 words per pair), peak memory reset before each measurement,
[5 runs per batch](docs/rag009-reranker-benchmark.txt):

| Batch | median | pairs/s | peak allocated | peak reserved | truncated pairs |
|------:|-------:|--------:|---------------:|--------------:|----------------:|
| 8 | 0.3194 s | 78.3 | 4.60 GiB | 4.82 GiB | 0 |
| **16** | **0.2905 s** | **86.1** | **4.93 GiB** | **5.36 GiB** | 0 |
| 32 | 0.3781 s | 66.1 | 5.30 GiB | 6.11 GiB | 0 |

No OOM at any size (the largest pairs are ~600 words, far below `max_length=8192`, so
nothing is truncated). Batch 16 is selected as the production default: highest
throughput at ~5 GiB reserved, i.e. ~27 GiB of headroom. In the full pipeline the
in-process rerank latency is ~0.35 s at batch 8 and ~0.37 s at batch 16, because the
second forward pass covers only the remaining 9 of 25 candidates — the isolated
throughput advantage of 16 shows up when many candidates are scored.

## 23. Latency benchmark

From [docs/rag009-latency.txt](../docs/rag009-latency.txt) (10 queries,
`candidate_k=25`, `final_k=8`, reranking on, warm models):

| Stage | first query | warm median | warm max |
|-------|------------:|------------:|---------:|
| query embedding | 0.014 s | 0.011 s | 0.017 s |
| Qdrant search | 0.152 s | 0.156 s | 0.311 s |
| rerank (25 candidates) | 0.306 s | 0.372 s | 0.876 s |
| **total** | **0.476 s** | **0.543 s** | **1.208 s** |

* Dense-only total for one warm query: **0.164 s**.
* The warm maximum (1.21 s, adaptive learning) and the second-slowest (1.04 s, DSR)
  are the two queries that needed a second widening round — the cost of not returning a
  short list, paid only when it is needed.
* Cold start (new process, weights not in VRAM): ~10.7 s for the first query, dominated
  by loading BGE-M3 and the reranker; a `--no-rerank` cold run took 8.0 s with 7.7 s in
  the first embedding call.
* Interactive use is practical: ~0.5 s warm per query, ~0.16 s without reranking.

## 24. Local Qdrant performance

`phd_rag_chunks_v1`, 11,469 points, 1024-d cosine, local embedded mode, median of 20
runs of the same query vector:

| Query filter | median | min | max |
|--------------|-------:|----:|----:|
| none | 32.6 ms | 26.7 ms | 34.3 ms |
| `EVIDENCE` (`evidence_eligible=true`) | 147.1 ms | 145.5 ms | 150.2 ms |
| `GENERAL` (`retrieval_eligible=true`) | 121.3 ms | 119.9 ms | 127.3 ms |
| `EVIDENCE` + year range | 167.9 ms | — | — |

Filtering costs ~90–115 ms locally because RAG-008's payload indexes do not accelerate
filtering in embedded mode (a known Qdrant limitation, unchanged by RAG-009). At this
corpus size that is acceptable: a full warm query is ~0.5 s and the filter overhead is
~20% of it. **No migration to a Qdrant server is needed**; if the corpus grows by an
order of magnitude, re-measure first.

## 25. Bibliography-like retrieval guard (RAG-009 final correction)

RAG-009 §20 found 142 indexed chunks that are reference lists but whose upstream
`content_type` is not `references`. Rebuilding RAG-007/RAG-008 was explicitly out of
scope, so the correction is a **retrieval-time guard**: nothing in Qdrant, the chunks,
the embeddings or the source files is touched (the collection still holds exactly
11,469 points with the RAG-008 fingerprint `3b1f63cb…`; `data/` is unmodified).

### Detector

[app/retrieval/bibliography.py](../app/retrieval/bibliography.py) measures each chunk's
own text and flags it only when **several independent bibliographic signals agree**.
Features: DOI count, URL count, `Available at:` count, `Accessed/Retrieved` count,
parenthetical publication years, `et al.`, journal furniture (`vol. 45`, `pp. 12`,
`Journal, 30(1), 1`), reference-entry line starts (`Surname, A.`), numbered entries,
number-immediately-followed-by-year (`Nat. Commun. 5, 4006 (2014)`), a
`References`/`Bibliography` heading, and marker density per 100 words.

Rules (first match wins, each with a recorded reason):

| Reason | Fires when |
|--------|-----------|
| `repeated_available_at_entries` | ≥2 `Available at:` **and** ≥2 URLs **and** density ≥ 4 |
| `citation_year_dense_list` | ≥8 parenthetical years **and** ≥2 DOI/URLs |
| `author_initials_entry_lines` | ≥5 entry-like lines **and** density ≥ 4 |
| `numbered_entry_list` | ≥6 numbered entries **and** ≥6 citation markers of any kind |
| `labelled_reference_section` | the chunk *is* a references section (heading) with ≥3 bibliographic markers and list content |
| `doi_dense_entry_list` | ≥4 DOIs in ≥60 words |
| `dense_reference_markers` | density ≥ 6, ≥3 non-URL bibliographic markers, ≥3 furniture markers, and no long prose line (or a short chunk, or a numbered/entry list) |

Ordinary academic prose is protected three times over: by the density thresholds, by
the **furniture** requirement (a number followed by a year, journal markers, DOIs - not
in-text citations), and by the **non-URL marker** requirement, so a list of bare links
(ORCIDs, resource lists, UNESCO endnotes) is never mistaken for a bibliography. "Do not
classify a chunk merely because it contains several normal academic citations" is
therefore enforced by construction, and pinned by tests
([tests/test_retrieval_bibliography.py](../tests/test_retrieval_bibliography.py), 31
tests, including a paragraph with 15 in-text citations that must stay retrievable).

### Mode policy and opt-in

| Mode | Bibliography-like chunks |
|------|--------------------------|
| `EVIDENCE`, `MY_WORK`, `GENERAL` | withheld by default |
| `GUIDANCE`, `SUPERVISION` | returned (these modes surface the author's own scaffolding material, where a reference-format table - e.g. the *Reference checklist* rules document - is legitimately the content) |

The three modes the correction names are the ones that withhold. Opt back in for
citation/bibliography analysis with `--include-bibliography`,
`RetrievalService(exclude_bibliography=False)`, `retrieve(..., exclude_bibliography=False)`
or `PHD_RAG_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY=false`. Every call records
`audit.exclude_bibliography`, `stats.dropped_as_bibliography`, a `bibliography_like`
drop reason with the rule name as detail, and `bibliography_like` / `bibliography_reason`
on each candidate.

### Corpus results

From [docs/rag009-bibliography-scan.txt](../docs/rag009-bibliography-scan.txt):

| Measurement | Value |
|-------------|-------|
| indexed chunks scanned | 11,469 |
| **known 142 candidates detected** | **133** |
| known 142 not detected | 9 |
| **additional bibliography-like chunks found outside the known set** | **140** |
| **total detected corpus-wide** | **273** (2.4% of the index) |

Reasons across the 273: `citation_year_dense_list` 125, `doi_dense_entry_list` 41,
`dense_reference_markers` 41, `repeated_available_at_entries` 35, `numbered_entry_list`
21, `author_initials_entry_lines` 8, `labelled_reference_section` 2. By role: literature
237, my_work 23, summaries 7, future_chapters 4, supervision 1, rules 1. By upstream
type: body 245, conclusion 17, summary_entry 7, table 3, appendix 1 — i.e. the guard
catches material the content-type field does not.

### Manual review

[docs/rag009-bibliography-manual-review.txt](../docs/rag009-bibliography-manual-review.txt)
contains 25 detected and 25 not-detected chunks with full text and a verdict per item:

* **Detected: 25/25 correct.** All are reference lists or reference-list-dominated
  chunks (Harvard, APA, Nature-numbered, IEEE-numbered, one-line PDF lists). The only
  mixed case (A04) is ~90 words of closing prose plus boilerplate plus a full A–Z
  reference list, i.e. ~75% bibliography.
* **Not detected: 20 correct skips, 4 borderline, 1 genuine false negative.**
  * correct skips (20): author's summary entries with one citation each, Chapter-7
    governance prose, a postprint cover page, article mastheads, ORCID lists, UNESCO
    endnote-link lists, comparison tables, and prose with dense in-text citation
    clusters (36 year-parentheses in 519 words - precisely the material that must stay
    retrievable);
  * borderline (4): citation-key tables ("claim → sources" columns) and AI-literacy
    tables with a "sample references" column - not reference lists, so leaving them is
    correct;
  * false negative (1): B05, a 105-word slice of three reference entries with three
    DOIs. No conservative rule distinguishes it from a paragraph citing three sources;
    missing it is the deliberate direction of the trade-off.
* Of the 9 undetected members of the known 142, eight are the same prose/summary/table
  material and one is B05.

### Effect on the reported problem and on retrieval quality

From [docs/rag009-bibliography-impact.txt](../docs/rag009-bibliography-impact.txt):

* **The MY_WORK query “what have I written about the historical development of AI?”** —
  with the guard off, ranks 4–8 of the top 8 were bibliography fragments (Radianti
  citation, Ouyang bibliography, second-paper reference entries, a "Reference List"
  chunk); with the guard on, **17 bibliography chunks are dropped and all eight results
  are prose** from the author's own drafts and paper. The measured problem is fixed.
* **The five required EVIDENCE queries** — the returned top 8 are **identical** with the
  guard on and off (drops: governance 6, VR engagement 1, adaptive learning 0,
  generative-AI integrity 7, DSR 0). Retrieval quality does not deteriorate, because the
  withheld chunks ranked below the returned ones for these queries.
* **Across the 36-query evaluation**: 112 bibliography-like candidates dropped, evidence
  leakage still 0, reranked Hit@5 0.857, Hit@10 0.964, MRR@10 0.871 (see §16/§17 for the
  dense-only movement and its explanation).

## 26. Final pytest result

```
514 passed, 2 warnings in 7.77s
```

416 pre-existing tests (RAG-001 … RAG-008) still pass unchanged, plus **98 new
RAG-009 tests**: query 7, filters 12, safety 7, reranking 9, diversity 10, determinism
9, widening rounds 6, evaluation 7, bibliography guard 31. No model download, no
network, no GPU is required by the suite: the embedder, the reranker and Qdrant are all
doubles or local-mode.

## 27. Known limitations and recommendation for RAG-010

Known limitations (recorded, not fixed here):

* **Corpus duplication.** Several papers exist as multiple physical copies with
  different hashes (e.g. one work contributes 30 chunks to the DSR query's top 100, and
  the whole top 25 can be two works). Retrieval compensates (identical-text
  suppression, work-level cap, widening rounds), but the underlying corpus contains
  genuine duplicates that a future ingestion pass could report.
* **Local metadata noise leaks into results.** First authors such as `ssrn`,
  `Windows User`, `Personal`, `reviewer`, `Form` and `How to Cite`, missing years, and
  mojibake titles (`060624-28.å¼ å–„æ–‡+…`) are RAG-006 extraction artifacts visible in
  output. They do not affect ranking, but they will affect citation rendering.
* **Two corpus documents in GUIDANCE mode with no titles** make work grouping fall back
  to the document id (correct, but it means untitled drafts cannot be grouped by work).
* **Document-level gold judgments.** Gold is asserted per document, not per passage, so
  a query scores a hit even when only a marginal chunk of the right paper returned; and
  duplicate copies of a work make a hit easier. Metrics are therefore optimistic.
* **GUIDANCE phrasing sensitivity** (`Reference checklist`, §19).
* **No threshold.** Rerank scores are compressed in the 0.85–0.9999 range on this
  corpus, and thin-topics produce near-zero tail scores (0.047). Retrieval returns
  `final_k` items regardless.
* **142 reference-list chunks were typed as body** (see §20). They are now withheld by
  the retrieval-time guard (§27), but the underlying classification gap remains: a
  RAG-007 follow-up should reclassify them (the guard is a filter, not a fix).
* **The bibliography guard misses small fragments.** One reviewed borderline case (a
  105-word, three-entry reference slice) is not detected, and citation-key tables are
  deliberately left retrievable. Precision was chosen over recall, per the brief.

Recommendation for RAG-010:

1. Answer only from EVIDENCE-mode results by default, and treat non-evidence modes as
   explicitly labelled context (never as citable literature).
2. Pass `dense_score` and `rerank_score` through unchanged, and decide a
   `min_rerank_score` **at generation time** with the numbers in §20 (the DSR tail at
   ~0.05 is the clearest signal); do not lower it in retrieval.
3. Cite from `title`, `first_author`, `year`, `doi`, `filename` and `page_start/end`
   exactly as returned; where `type=table` or `content_type=table`, prefer rendering the
   chunk as a table rather than prose.
4. Surface the OCR flag and locators in citations: 104 indexed chunks are OCR-derived,
   and `page:x:block:y` locators are the only page-accurate pointer for OCRed pages.
5. Keep the bibliography guard on (§27) so a generation step cannot quote a citation
   list as prose; if citation analysis is ever wanted, run it with
   `--include-bibliography` rather than by weakening the default.
6. Do not add query rewriting or HyDE before measuring: the plain dense + rerank
   pipeline already reaches Hit@10 0.964 on this set, and its failures are dominated by
   corpus gaps (a topic with one paper) rather than by query phrasing.
7. Keep `candidate_k=25` / `final_k=8` unless generation needs more context; raising
   `candidate_k` mostly buys more copies of the same works for these queries.

**RAG-009 COMPLETE** — no RAG-010 work has begun.
