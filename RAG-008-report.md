# RAG-008 Report — BGE-M3 dense embeddings and persistent Qdrant indexing

Task: turn the RAG-007 chunk corpus into a searchable dense vector index. Every
retrieval-eligible chunk gets a BGE-M3 dense embedding (1024-dim, cosine-ready) and
one deterministic Qdrant point whose payload carries the provenance RAG-009 needs.

Explicitly **not** built here: retrieval ranking, reranking, hybrid search, sparse
vectors, ColBERT/multi-vector retrieval, query orchestration, generation, citation
rendering, agents or any LLM answering. RAG-008 ends at "the vectors are stored and
inspectable": the validated pilot (§18-§20), the GPU benchmark (§22) and finally the
**complete 11,469-chunk index** (§23) — no retrieval logic of any kind.

## 1. Files created

Application packages:

| File | Purpose |
|------|---------|
| `app/embedding/__init__.py` | Package exports |
| `app/embedding/models.py` | `EmbeddingSpec`, the `EmbeddingModel` protocol, errors, `as_vector_list`, `count_tokens_many` |
| `app/embedding/bge_m3.py` | `BgeM3Embedder`: lazy load, revision resolution, device policy, normalization, tokenizer counting |
| `app/embedding/batch.py` | Streaming, cache-aware batching (`iter_batches`, `embed_requests`, `embed_stream`) |
| `app/embedding/cache.py` | Disk embedding cache keyed by model + chunk id + retrieval-text hash |
| `app/embedding/inspect.py` | `python -m app.embedding.inspect` CLI |
| `app/indexing/__init__.py` | Package exports |
| `app/indexing/models.py` | `IndexingOptions`, `IndexableChunk`, `IndexReport`, `DryRunReport`, `TokenStats`, failure rows |
| `app/indexing/payload.py` | `build_payload`: the Qdrant payload contract |
| `app/indexing/collection.py` | Qdrant client factory, schema/dimension safety, payload indexes, point I/O |
| `app/indexing/fingerprint.py` | Deterministic corpus fingerprint |
| `app/indexing/manifest.py` | `IndexManifest` (what produced these vectors) |
| `app/indexing/indexer.py` | Streaming population selection + index build |
| `app/indexing/build.py` | `python -m app.indexing.build` CLI |
| `app/indexing/inspect.py` | `python -m app.indexing.inspect` CLI, including the pilot checks |
| `app/chunking/corpus.py` | Shared RAG-007/RAG-008 corpus iterator (see §2) |

Tests: `tests/embedding_support.py` and
`tests/test_embedding_{model,cache}.py`,
`tests/test_indexing_{payload,collection,sync,fingerprint,pilot,cli}.py`.

Artifacts: `docs/rag008-indexing.txt` (full corpus run), `docs/rag008-indexing-reindex.txt`
(unchanged re-run), `docs/rag008-full-validation.txt` (separate-process validation),
`docs/rag008-dryrun-tokens.txt` (corpus-wide dry run + BGE-M3 tokenizer measurement),
`docs/rag008-index-manifest.json` (the manifest), `docs/rag008-gpu-benchmark.txt`, and
the pilot set kept separately: `docs/rag008-pilot-indexing.txt`,
`docs/rag008-pilot-indexing-reindex.txt`, `docs/rag008-pilot-index-manifest.json`,
`docs/rag008-pilot-collection.txt`, `docs/rag008-pilot-points.txt`,
`docs/rag008-stale-{seed,retained,deleted}.txt`, `docs/rag008-stale-demo-manifest.json`.

## 2. Files modified

| File | Change |
|------|--------|
| `app/config.py` | Embedding and Qdrant settings (`PHD_RAG_EMBEDDING_*`, `PHD_RAG_QDRANT_*`, `PHD_RAG_INDEX_MANIFEST_DIR`) |
| `app/chunking/report.py` | Now consumes the shared corpus iterator instead of its own scan/parse/chunk loop |
| `app/embedding/*`, `app/indexing/*` | New packages (above) |
| `README.md` | Documented the RAG-008 CLIs, the new dependencies and the environment variables |
| `pyproject.toml` | Declared the RAG-008 runtime dependencies |
| `.gitignore` | Ignores the local embedding cache, index manifests and the local Qdrant store |

`app/chunking/corpus.py` exists so the RAG-007 report and the RAG-008 indexer cannot
drift apart: both now walk the same deterministic document order, resolve the corpus
role from all physical locations the same way, and produce the same chunks. The
refactor was verified by re-running the full RAG-007 corpus report and diffing it
byte-for-byte against the previously committed artifact — identical
(`14,786` chunks, same distributions, same role audit).

No existing test was edited, and no RAG-001…RAG-007 behaviour changed.

## 3. Dependencies added

`numpy`, `torch`, `transformers`, `sentence-transformers`, `huggingface-hub` and
`qdrant-client` are now declared in `pyproject.toml`. All six were already present
in the environment at the versions shown in §4; nothing was upgraded, and no
alternative embedding framework was introduced.

## 4. Embedding model

```
model:        BAAI/bge-m3
library:      sentence-transformers 6.1.0 (transformers 5.19.0, torch 2.14.1+cu130)
revision:     5617a9f61b028005a4858fdac845db406aefb181   (snapshot commit hash)
dimension:    1024  (read from the model, never hard-coded)
max tokens:   8192  (model limit reported by the loaded model)
normalization: l2 -> unit norm, verified numerically
device:       cuda (RTX 5090, compute capability 12.0); "cpu" when CUDA is absent
cache path:   C:\Users\User\.cache\huggingface\hub\models--BAAI--bge-m3
```

**Device history (relevant to reading §18/§20).** The pilot and the corpus dry run in
§16-§20 were produced *before* the environment's CPU-only PyTorch wheel was replaced:
at that time `torch 2.14.1+cpu` reported `torch.cuda.is_available() = False` even
though the machine has an RTX 5090, so device auto-selection correctly resolved to
`cpu`. The wheel was then swapped for the CUDA build of the **same** torch version
(`torch==2.14.1+cu130`, only that one package changed, verified by a `pip freeze`
diff), after which `torch.cuda.is_available() = True`, one device is visible
("NVIDIA GeForce RTX 5090"), and device auto-selection resolves to `cuda`. The pilot
timings in §18 are therefore CPU timings and remain accurate for what they measured;
§22 benchmarks the same input on the GPU.

* The weights are downloaded once into the Hugging Face cache and re-used locally;
  `--local-files-only` makes a run strictly offline.
* The revision is not guessed: `snapshot_download` resolves the hub id to the
  snapshot directory whose name *is* the commit, and that value goes into the
  specification, the cache key and the manifest.
* `--device cuda` on a machine without CUDA **fails** instead of silently running on
  the CPU; `--device` unset means "CUDA when available, else CPU" and the chosen
  device is reported.
* The dense head is the only thing used: BGE-M3's sparse and ColBERT heads are for
  RAG-009 to consider, so no `FlagEmbedding` dependency was added.

## 5. Embedding layer

```python
class BgeM3Embedder:
    spec -> EmbeddingSpec(model_name, model_revision, dimension, normalized,
                          device, library, library_version, max_tokens)
    embed_documents(texts, *, batch_size=None) -> list[list[float]]
    embed_query(text) -> list[float]
    count_tokens(text) -> int | None
    count_tokens_many(texts) -> list[int | None]
    dimension, normalized, max_tokens, local_path
    verify_normalization(sample) -> float
```

Indexing code never touches a sentence-transformer: it uses this interface, which is
why the unit tests can substitute a deterministic double.

Vectors are converted through float32 (`as_vector_list`) because that is the
precision Qdrant stores; a vector read back from the cache is therefore *identical*
to the one that produced it, instead of differing in the last decimal place.

**Normalization** is applied once, by the model
(`normalize_embeddings=True`), and verified numerically: `verify_normalization()`
embeds a sample and returns its L2 norm, which the inspection CLI prints. Measured:
`1.000000`.

## 6. Batching and memory

Two independent knobs, both configurable by flag and environment variable:

* `--embedding-batch-size` (default 8) — texts per forward pass;
* `--upload-batch-size` (default 128) — points per Qdrant upsert.

The population is streamed: documents are chunked one at a time, chunks accumulate
into an upload batch, that batch is embedded (in embedding batches), converted to
points, uploaded, and dropped. Nothing ever materialises the ~11.5k retrieval texts
or their vectors at once. A batch that fails to embed is retried item by item, so a
single pathological chunk cannot destroy the run.

## 7. What is embedded

`Chunk.retrieval_text` — the RAG-007 text that carries the synthetic
`[Document: …]` / `[Section: …]` labels — falling back to `Chunk.text` only when a
chunk has no retrieval text. `Chunk.text` is never modified, and the payload stores
both fields separately so a retrieved result can quote the bare source text.

Indexing population: **`retrieval_eligible` chunks only** (11,469 of 14,786 in this
corpus). The 3,317 reference chunks stay in the chunk corpus and out of the normal
retrieval collection. No role is filtered out: `my_work`, `summaries`, `supervision`,
`rules`, `guidance` and `future_chapters` are all retrieval-eligible and therefore
indexed, with their role and evidence policy in the payload for RAG-009 to filter on.

## 8. Qdrant schema

```
collection:  phd_rag_chunks_v1 (configurable)
vector size: 1024   (== model dimension, checked)
distance:    Cosine
points:      one per chunk, id = Chunk.id (RAG-007 UUID)
```

Payload indexes (kept deliberately short): `corpus_role`, `evidence_eligible`,
`content_type`, `year`, `document_id`, `file_hash`.
Local-mode Qdrant accepts index creation but ignores it (it prints a warning); the
indexes take effect on a Qdrant server, and they are recorded in the manifest.

Safety rules:

* an existing collection with a **different dimension** aborts the run
  (`CollectionDimensionMismatch`) — it is never dropped and recreated;
* a collection that does not exist is created with the dimension read from the
  model, then the payload indexes are applied;
* `--qdrant-path` (local persistent store, the development default) and
  `--qdrant-url` (server) are both first-class; the local store is flushed on close
  so a later process sees the points.

## 9. Payload

Every point carries the fields RAG-009 needs to filter and to render a citation
without re-reading the source PDF:

```
chunk_id  document_id  file_hash  filename  chunk_index
page_start  page_end  section_title  heading_path  content_type
corpus_role  authority  evidence_eligible  retrieval_eligible
text_sources  ocr_involved
title  first_author  year  doi  publication_type
source_locators  sequence_start  sequence_end  token_count
text  retrieval_text
```

Plus `duplicate_locations` / `role_resolution` **only** for logical documents with
several physical copies, so a chunk can explain why it carries the role it carries.
The RAG-006 candidate history is deliberately *not* copied into points; it stays
addressable by `file_hash`.

## 10. Identity and idempotency

* Point id = `Chunk.id` = `uuid5(chunk namespace, file_hash:chunk_index:sha256(text))`
  from RAG-007. No random ids are generated, so a re-run **updates** the same points.
* `payload["document_id"]` = `Chunk.document_id` = `uuid5(document namespace, file_hash)`
  (RAG-007 §15), verified by a regression test against `document_id_for(file_hash)`.
* Reported per run: points before, points after, inserted, updated/upserted,
  duplicate point ids, stale points (and deleted stale points).

## 11. Stale points

`stale = points in the collection that the current chunk set no longer contains`.
They are always *reported*; they are only *deleted* when the CLI is run with
`--delete-stale`. The default is therefore non-destructive: a shrunk or renamed
corpus cannot silently throw away vectors. Detection is scoped to the collection
being written, so indexing into a second collection never touches the first.

## 12. Corpus fingerprint and manifest

`corpus_fingerprint = sha256(recipe version + chunking version + digest(sorted
file hashes) + digest(sorted chunk ids))` — order-independent, timestamp-free, and
stable for an unchanged corpus. It describes the chunks the run targeted, so it does
not change when an individual chunk happens to fail to embed.

The manifest (`IndexManifest`, written as JSON next to the configuration and, for the
pilot, into `docs/`) records: collection, Qdrant target, distance, vector dimension,
embedding model + revision + identifier, library and version, normalization, device,
chunking version, chunk/document counts, the pilot limit (0 = full run), the corpus
fingerprint and both digests, point totals, stale counts and the build timestamp.
A collection name alone is never treated as sufficient provenance.

## 13. Failure handling

* Embedding model load failure → the run **aborts** (`EXIT_MODEL`) before anything is
  written.
* Collection dimension mismatch → the run **aborts** (`EXIT_QDRANT`); nothing is
  recreated.
* A failed batch is retried per item; the failing chunks are recorded in
  `embedding_failures` with chunk id, filename and error, counted in `skipped_chunks`
  and printed in the report. Failures are never silently dropped.
* Upload failures are recorded in `upload_failures` and counted; the run continues so
  the operator sees the full picture.
* A corrupted embedding cache file is ignored (treated as empty) instead of breaking
  an indexing run.

## 14. CLIs

```
# Embedding inspection: model, revision, device, dimension, normalization, sample norm
python -m app.embedding.inspect [--show-vector N] [--local-files-only] [--device X]

# Indexing: dry run, pilot, full run, stale synchronisation
python -m app.indexing.build data \
    [--collection phd_rag_chunks_v1] [--qdrant-path .qdrant_data | --qdrant-url URL] \
    [--embedding-batch-size 8] [--upload-batch-size 128] [--device cpu] \
    [--limit N] [--pilot N] [--dry-run] [--delete-stale] [--measure-tokens] \
    [--cache-path ...] [--no-cache] [--manifest-path ...] [--report-out ...]

# Collection inspection, with the §26 pilot assertions
python -m app.indexing.inspect [--collection ...] [--limit N] [--show-text] [--check]
```

`--dry-run` chunks the corpus, reports the intended indexing population and validates
the configuration, and **never** computes embeddings or touches Qdrant. Exit codes:
`0` ok, `2` configuration, `3` model load, `4` token-limit violation, `5` Qdrant.

## 15. Tests

`416 passed` = 362 pre-existing (RAG-001…RAG-007, untouched) + **54 new RAG-008
tests**. All external dependencies are mocked or local: no test downloads BGE-M3, and
none contacts a Qdrant server (they use Qdrant's local/in-memory mode).

| Test file | Tests | The task's numbered checks |
|-----------|------:|----------------------------|
| `tests/test_embedding_model.py` | 8 | 1-5 (determinism, dimension, batch order, unit norm, explicit load failure) plus backend failure, device policy, tokenizer counting |
| `tests/test_embedding_cache.py` | 8 | 19-21 (cache hit, text invalidation, model invalidation) plus key composition, corrupt cache, batching, per-item failure isolation |
| `tests/test_indexing_payload.py` | 7 | 6-11 (payload fields, role/evidence, OCR, references excluded, my_work, future_chapters) and 12-14 (point id = chunk id, reindex without duplicates, RAG-007 document UUID) |
| `tests/test_indexing_collection.py` | 7 | 15-18 (cosine + dimension, mismatch raises, no destructive recreation, payload indexes) plus reuse, name validation, client factory |
| `tests/test_indexing_sync.py` | 3 | 22-24 (stale detected, retained without the flag, deleted with the flag) plus collection scoping |
| `tests/test_indexing_fingerprint.py` | 5 | 25-27 (same corpus, ordering, timestamp) plus version sensitivity and manifest round-trip |
| `tests/test_indexing_pilot.py` | 6 | Population streaming, pilot sampling, determinism, document limit, RAG-007 chunk reuse, chunking-option pass-through |
| `tests/test_indexing_cli.py` | 10 | Dry run (no Qdrant writes), pilot dry run, configuration validation, end-to-end pilot with injected doubles, token-measurement run, inspection CLI + checks, embedding inspection CLI, role payloads |

## 16. Corpus-wide dry run

`python -m app.indexing.build data --dry-run --measure-tokens` (one pass over the
whole corpus; artifact `docs/rag008-dryrun-tokens.txt`):

```
unique documents:            466
documents with chunks:       465
total RAG-007 chunks:        14786
retrieval-eligible chunks:   11469     <- the intended indexing population
excluded (ineligible):        3317
  of which references:        3317     <- exactly the reference chunks
chunking version:            rag007-normalize-v1
```

This matches the RAG-007 corpus report exactly — which is the point of the shared
corpus iterator (§2): the indexer and the chunking report see the same corpus, the
same roles and the same chunks.
Run with `--measure-tokens`, the dry run loads the tokenizer (and **only** the
tokenizer: `compute_embeddings=False`, so no vector is computed and Qdrant is not
contacted at all). It is a dry run plus a measurement, never an index build. The pass
took 1076 s (17.9 min) for 466 documents, including parsing, OCR and chunking; the
tokenizer measurement is reported in §17.

## 17. BGE-M3 tokenizer measurement (§27)

Measured on the retrieval text of every eligible chunk with the real BGE-M3
tokenizer, changing no chunk boundary:

```
chunks measured:           11469      (every retrieval-eligible chunk)
max BGE-M3 tokens:         2197
median:                    731
p95:                       986
model max input tokens:    8192
chunks above 1200:         63
chunks above model limit:  0
largest estimator count:   1200
measurement pass:          1076 s (17.9 min), no embeddings, no Qdrant contact
```

Consequences:

* **No chunk exceeds the model's input limit** (max 2197 vs 8192), so nothing is
  truncated and no re-chunking is required before indexing: the pilot embedded these
  texts as-is.
* **The RAG-007 estimator is not uniformly conservative.** Its 1200-token ceiling maps
  to a real median of 731 and p95 of 986, but the tail runs higher: 63 chunks exceed
  1200 real tokens, the largest reaching 2197 — 1.8x its estimated size. The estimator
  counts a "word" as `ceil(len / 4)` tokens, which under-counts text that is dense in
  numbers, identifiers, URLs or OCR noise. A chunk can therefore be up to ~1.8x larger
  than the estimator promised, while still being far inside the model limit.
* This is exactly why §27 asked for a real-tokenizer measurement: the estimator remains
  the right *boundary* rule (it is deterministic, offline and model-independent), but
  it must not be quoted as a model token count.
* **Recommended follow-up (not done here, measurement only):** if later stages want a
  hard guarantee in *model* tokens, either re-chunk with a `Tokenizer` implementation
  backed by the BGE-M3 tokenizer (RAG-007 already exposes that substitution point) or
  lower `hard_max_tokens`. The 63 oversized chunks were not individually inspected,
  because that would require a second full corpus pass; the count and the maximum are
  the actionable facts.

> **Correction note.** An earlier draft of this report stated *"max 688 real vs 1192
> estimated"* tokens. **Neither figure was ever measured.** They were placeholder
> numbers typed while the measurement pass was still running: 1192 approximated the
> estimator ceiling and 688 was invented. The measured values above (max 2197, p95
> 986, median 731, estimator ceiling 1200) replace them, and the conclusion is the
> *opposite* of what the placeholder implied: the estimator is conservative in the
> body of the distribution, not in its tail. §21.6 carries the corrected statement.
> The measurement was not re-run to repair this: the numbers in this section come
> straight from `docs/rag008-dryrun-tokens.txt`, which is the artifact the run wrote.

## 18. Pilot run (§25)

`python -m app.indexing.build data --pilot 100 …` against real BGE-M3 and a real
**local persistent** Qdrant store (`.qdrant_data`); artifact
`docs/rag008-indexing.txt`:

```
collection:            phd_rag_chunks_v1   created with 1024 dims, Cosine
points before:         0
points after:          100
inserted:              100
updated/upserted:      0
duplicate point ids:   0
stale points:          0

documents embedded:    100
chunks embedded:       100
skipped chunks:        0
embedding failures:    0
upload failures:       0

embedding seconds:     25.8
embeddings/second:     3.88
upload seconds:        0.66
total seconds:         220.5
device:                cpu   (no CUDA device available)
cache hits/misses:     0 / 100
payload indexes:       corpus_role, evidence_eligible, content_type, year,
                       document_id, file_hash
corpus fingerprint:    d9723980cf423dc5fc18a23f0a4bfa85a86dcb64f56f898c8057d73cddb117f2
```

Pilot selection is deterministic and documented: one retrieval-eligible chunk per
document, visiting the **non-literature roles first** (they are small folders that
would otherwise sit behind hundreds of literature PDFs in filename order), then
literature in filename order, plus a top-up step for any required kind still missing.
The pilot therefore covers every corpus role without inflating the sample.

`total_seconds` is dominated by parsing and chunking the 100 documents (≈194 s of the
220 s); embedding is 26 s for 100 chunks on a CPU-only machine (~1.7 s per
1000-token chunk at batch size 8).

## 19. Pilot verification (§26)

**Collection inspection** (`python -m app.indexing.inspect --check`, artifact
`docs/rag008-pilot-collection.txt`) — all ten checks pass:

```
by corpus role:            by content type:        eligibility:
  literature       75        body           94       retrieval_eligible: all 100
  future_chapters   8        table           5       evidence_eligible:  75 true
  rules             8        summary_entry   1                         25 false
  my_work           5
  supervision       2        OCR chunks:      3
  summaries         1        reference points: 0
  guidance          1
```

**One point of each required kind** (artifact `docs/rag008-pilot-points.txt`):

| Kind | File | corpus_role | retrieval | evidence | Details |
|------|------|-------------|-----------|----------|---------|
| literature paper | `Artificial_Intelligence_in_Education_A_Review.pdf` | literature | yes | **yes** | DOI `10.1109/ACCESS.2020.2988510`, year 2020, title, page 1, 1027 tokens |
| OCR chunk | `2021_UNESCO_AI-and-education-Guidande-for-policy-makers_EN.pdf` | literature | yes | yes | `ocr_involved = true`, `text_sources = ['native_pdf', 'ocr']`, pages 1-3 |
| `my_work` chunk | `Historical-Development-24-02-2025.docx` | my_work | yes | **no** | authority `high` |
| `future_chapters` chunk | `Chapter_1_Introduction_Revised.docx` | future_chapters | yes | **no** | authority `navigation_only` |

* `my_work` and `future_chapters` are retrievable but not evidence — the RAG-007 role
  policy carried into the payload.
* An OCR chunk is present and marked, including a mixed native+OCR chunk.
* **No point in the collection has `content_type = references`** (query returns 0).
* Every point carries the full required payload field set; every vector has 1024
  dimensions.

**Restart persistence**: the inspection ran in a *separate process* that opened the
local store after the indexing process had exited, and found all 100 points with their
1024-dim vectors intact — the pilot is genuinely persistent, not in-memory.

## 20. Idempotency and stale handling on real data

Re-running the identical pilot command (artifact
`docs/rag008-indexing-reindex.txt`):

```
cache hits:            100      cache misses: 0
embeddings/second:     9090.9   (100 vectors served from the cache in 0.011 s)
points before:         100
points after:          100
inserted:              0
updated/upserted:      100
duplicate point ids:   0
stale points:          0
corpus fingerprint:    d9723980cf423dc5fc18a23f0a4bfa85a86dcb64f56f898c8057d73cddb117f2   (identical)
```

A re-index of an unchanged corpus therefore recomputes nothing, rebuilds nothing and
creates no duplicates: the same chunk ids update the same points.

Stale handling was exercised on real documents in a separate collection
(`phd_rag_chunks_stale_demo`, artifacts `docs/rag008-stale-*.txt`):

| Step | Command | points before → after | stale | deleted |
|------|---------|----------------------|-------|---------|
| seed | 4 documents | 0 → 4 | 0 | 0 |
| shrink, default | 2 documents | 4 → 4 | **2 reported, retained** | 0 |
| shrink, `--delete-stale` | 2 documents | 4 → 2 | 2 | **2** |

The default is non-destructive: a shrunken corpus cannot silently drop vectors, and
deletion happens only with the explicit flag.

## 21. Known limitations and architectural concerns

1. **Resolved: the collection now holds the whole corpus.** `phd_rag_chunks_v1`
   was seeded with 100 pilot points and then filled with the complete indexing
   population: **11,469 points** (§23). The pilot artifacts are kept separately
   (`docs/rag008-pilot-*`) because they document the sampling and idempotency checks
   that justified the full run.
2. **The pilot ran on the CPU; the GPU is now available.** The pilot numbers in §18
   are CPU numbers (`torch 2.14.1+cpu`, 3.88 embeddings/s), produced before the CPU
   wheel was replaced by `torch 2.14.1+cu130`. §22 benchmarks the identical 100-chunk
   input on the RTX 5090: ~132 embeddings/s steady state, i.e. ~34x the CPU rate, so
   the full 11,469-chunk embedding pass is a ~1.5-2.5 minute GPU job instead of
   ~49 minutes on the CPU. The chunking/parsing pass (17.9 min over 466 documents,
   OCR included) is now the dominant cost of a full run.
3. **Payload indexes are inert in Qdrant local mode.** The embedded mode prints
   *"Payload indexes have no effect in the local Qdrant"*; indexes are created as
   configured and take effect against a Qdrant server. Filtering still works locally,
   it just does not use an index.
4. **Local Qdrant allows one process at a time** (the store directory is locked), so
   the inspection CLI cannot run while a build is running. The indexing CLI closes its
   client explicitly for exactly this reason.
5. **The embedding cache never evicts entries.** Keys are `(model identifier, chunk
   id, retrieval-text hash)`, so stale entries are never *used*, but the file grows
   with every edited chunk (ceiling ~45 MB at float32 for this corpus). A prune command
   can be added when it matters.
6. **Estimator and model token counts differ, and the estimator can under-count.**
   RAG-007 boundaries came from `subword-estimator-v1` (ceiling 1200 *estimator*
   tokens). §17 measures the same population with the real tokenizer: median 731,
   p95 986, **max 2197**, with 63 chunks above 1200. The estimator is therefore a good
   band in the middle of the distribution but **not a guaranteed upper bound in model
   tokens** (observed worst case ~1.8x), even though everything stays far inside the
   model's 8192-token limit. Never quote an estimator count as a model token count,
   and treat "≤1200" as an estimator claim only.
7. **`--delete-stale` is collection-scoped.** Points in other collections are never
   touched, and there is no cross-collection namespace registry: a collection can only
   be identified by its manifest.
8. **Fingerprint scope.** The fingerprint covers the chunks a run targeted (file
   hashes + chunk ids + chunking version). It deliberately excludes the embedding
   revision and the collection name — those live next to it in the manifest.
9. **One chunk per document in the pilot** is a sampling choice, not an indexer
   property: it exercises each document's first chunk (and therefore the whole role
   and OCR matrix) without paying for 11k embeddings. A full run indexes every
   eligible chunk.
10. **The author's own material is embedded.** `my_work`, `rules`, `guidance`,
    `summaries` and `supervision` are retrieval-eligible and therefore indexed. That is
    intended, but it means the vector store contains unpublished text: RAG-009 must
    honour `evidence_eligible` when citing, and any future export/share of the index
    must filter by role first.
11. **The `supervision`/`rules`/`guidance` chunks carry `text_sources = []`** when they
   come from DOCX/Markdown (no page-level extraction source exists for those formats,
   a RAG-007 decision). That is why the pilot's OCR evidence is a PDF chunk.

## 22. GPU benchmark and production batch size

Benchmark: `docs/rag008-gpu-benchmark.txt`, produced with
`python rag008_gpu_bench.py` (a throwaway harness, not part of the shipped CLI).

Method: the **same deterministic 100-chunk pilot input** as §18 — verified inside
the run by comparing the chunk ids with the points already stored in
`phd_rag_chunks_v1` (`identical to indexed pilot ids: True`). Embeddings are computed
live (`embedder.embed_documents`, no cache involved), `torch.cuda.reset_peak_memory_stats()`
is called before every measurement, each configuration is measured 3 times and the
**median** is reported, GPU utilisation is sampled from `nvidia-smi` during each
measurement, and nothing is uploaded (embedding-only benchmark, the production
collection is untouched).

```
GPU:              NVIDIA GeForce RTX 5090 (compute capability 12.0, 31.82 GiB)
torch:            2.14.1+cu130      CUDA runtime: 13.0
model:            BAAI/bge-m3 @ 5617a9f61b028005a4858fdac845db406aefb181
input:            100 chunks, 47,388 real BGE-M3 tokens (median 514, max 961)
pilot estimator:  57,078 estimator tokens -> real/estimator ratio 0.830
```

### Phase 1 - the pilot input itself (median of 3 runs)

| batch | time | embeddings/s | tokens/s | peak allocated | peak reserved | GPU util (mean/peak) |
|------:|-----:|-------------:|---------:|---------------:|--------------:|----------------------|
| 8 | 0.938 s | 106.6 | 50,515 | 2.49 GiB | 2.71 GiB | 93.8% / 95% |
| **16** | **0.908 s** | **110.2** | **52,212** | 2.84 GiB | 3.24 GiB | 98.6% / 100% |
| 32 | 0.970 s | 103.0 | 48,829 | 3.46 GiB | 4.52 GiB | 100% / 100% |
| 64 | 1.071 s | 93.4 | 44,248 | 4.79 GiB | 6.85 GiB | 100% / 100% |
| 128 | 1.439 s | 69.5 | 32,942 | 6.27 GiB | 9.47 GiB | 85.8% / 100% |

No OOM and no failures at any size, including 128 (9.47 GiB reserved of 31.82 GiB).

### Phase 2 - steady state (the same input repeated 20x = 2,000 chunks)

A sub-second measurement is dominated by per-call overhead, so the same texts were
repeated to obtain a stable throughput figure (also used for the estimate below):

| batch | time | embeddings/s | tokens/s | peak allocated | peak reserved | GPU util (mean/peak) |
|------:|-----:|-------------:|---------:|---------------:|--------------:|----------------------|
| 8 | 16.25 s | 123.1 | 58,339 | 2.51 GiB | 3.08 GiB | 92.3% / 100% |
| **16** | **15.10 s** | **132.5** | **62,783** | 2.86 GiB | 3.98 GiB | 93.8% / 100% |
| 32 | 15.16 s | 131.9 | 62,527 | 3.58 GiB | 5.35 GiB | 96.7% / 100% |
| 64 | 15.29 s | 130.8 | 61,998 | 4.88 GiB | 6.85 GiB | 90.4% / 100% |
| 128 | 15.57 s | 128.4 | 60,869 | 7.65 GiB | 11.52 GiB | 97.5% / 100% |

Reading: throughput **peaks at batch 16** and is flat (within 1%) out to 64; 128 is
slower *and* the most memory-hungry. The GPU is saturated (≥90% utilisation) from
batch 8 upwards, so there is no headroom to buy with larger batches.

### Selected production batch size: **16**

* it is the throughput optimum in both phases (110.2 emb/s short-run, 132.5 emb/s
  steady state) — larger batches are equal at best and clearly worse at 128;
* it is the *smaller* of the two joint-best choices (16 vs 32), so it is the
  conservative one: 3.98 GiB reserved in steady state versus 5.35 GiB, and the
  marginal cost measured across the sweep is ~0.09 MiB per token per batch, so even a
  batch of 16 worst-case 2197-token chunks (the corpus maximum, 2.3x the pilot's
  longest chunk) stays in single-digit GiB — well inside the 31.82 GiB available;
* `DEFAULT_EMBEDDING_BATCH_SIZE` is therefore 16 (`PHD_RAG_EMBEDDING_BATCH_SIZE`,
  `--embedding-batch-size`); on a much smaller GPU the flag can be lowered with no
  code change.

### Expected full-corpus embedding time

| Estimate | Basis | Result |
|----------|-------|--------|
| chunks/s | 11,469 chunks / 132.5 emb/s | **~87 s** |
| tokens/s (token-proportional, more robust) | ~9M corpus tokens (11,469 x ~780 mean real tokens, from median 731 / p95 986) / 62,783 tok/s | **~140 s** |

**≈1.5-2.5 minutes of pure embedding** for the whole retrieval-eligible corpus, plus
the point upload (the pilot's 100 points took 0.66 s, so 11.5k points are a matter of
seconds) — versus the ~49 minutes the same job would need on the CPU at the measured
3.88 embeddings/s (a ~34x speed-up).

The full run's wall-clock is therefore dominated by the RAG-007 pass, not by the GPU:
the corpus-wide dry run took **17.9 min** for 466 documents (parsing, OCR, chunking),
so an end-to-end full index is expected to land around **20-22 minutes**, of which
under 3 minutes are embedding. The measured full run (§23) landed at **21.5 min**,
with 157 s of embedding.

**Full indexing was not started** (100 of 11,469 chunks are indexed), and RAG-009 was
not begun.

## 23. Full corpus index run

Approved configuration, executed exactly as specified with `python -m app.indexing.build data`
against the **existing** pilot collection (no recreation, no deletion):

```
embedding model:      BAAI/bge-m3 @ 5617a9f61b028005a4858fdac845db406aefb181
device:               cuda   (RTX 5090, compute capability 12.0)
CUDA runtime:         13.0   (torch 2.14.1+cu130)
embedding batch size: 16
upload batch size:    128
vector dimension:     1024   |  normalization: L2  |  Qdrant distance: Cosine
collection:           phd_rag_chunks_v1 (local persistent store .qdrant_data)
artifact:             docs/rag008-indexing.txt
```

### Population (no truncation, no re-chunking)

```
total RAG-007 chunks:        14786
retrieval-eligible chunks:   11469
excluded reference chunks:    3317   (all references; never indexed)
chunks embedded:             11469
embedding failures:              0
upload failures:                 0
skipped chunks:                  0
```

### Points and cache

```
points before:        100      (the pilot points)
points after:         11469
inserted:             11369
updated/upserted:     100      (the pilot points, updated in place)
duplicate point ids:  0
stale points:         0
cache hits:           100      (exactly the pilot vectors)
cache misses:         11369
actual GPU embeddings: 11369   (11,469 chunks minus 100 cache hits)
```

### Timing, GPU and corpus identity

```
embedding seconds:        157.2
embeddings/second:         72.9
real tokens/second:     44346.1   (6,972,629 real tokens)
Qdrant upload seconds:     69.2
total wall-clock:        1290.3 s (21.5 min)
peak allocated VRAM:       3.76 GiB
peak reserved VRAM:        9.77 GiB
corpus fingerprint:       3b1f63cb00ebca341587ad7a7daaa060a1ca7d6cc17879b9c9cea615853637f3
document hashes digest:   42edf2f8b939e8d89b1411c9febbe485e7f1dcb03a14afafb548c856913af899
chunk ids digest:         74a200acb811a0f2961bf39f8bf97f9578a5e1f6993a6f035e98ab9bcbab4d86
```

The tokenizer measurement ran inside the same pass (`--measure-tokens`), independently
of embedding: 11,469 chunks measured, max 2197, median 731, p95 986, 63 above 1200,
**0 above the 8192 model limit** — identical to §17, which is the stability check that
matters: the full population reproduces the dry run exactly.

**Throughput back-calculation is honest but narrower than the isolated benchmark:**
the in-pipeline rate is 72.9 embeddings/s (44,346 real tokens/s) against 132.5
embeddings/s (62,783 tokens/s) measured in §22. The difference is architectural, not
GPU capacity: the pipeline is strictly serial — embed batch, convert vectors to
Python lists, build payloads, upload, then tokenize the next batch — so the GPU idles
during the CPU phases. The corpus chunks are also longer than the pilot's (median 731
vs 514 real tokens). No pipelining/prefetch was added, because none of it was asked
for and it changes no output.

The full run's wall clock (21.5 min) matches the §22 estimate (~20-22 min) and is
dominated by the RAG-007 pass (parsing, OCR, chunking) plus the local Qdrant writes;
the GPU spent ~2.6 minutes computing 11,369 vectors.

**Point count is exactly 11,469** (the approved expectation), the collection was not
recreated, and the 100 pilot points were updated rather than duplicated.

### Validation after indexing (separate process, read-only)

Artifact: `docs/rag008-full-validation.txt` — the indexing process exited, then a new
process opened `.qdrant_data` and verified everything; **0 failures**:

```
persisted point count = 11469        [ok]
vector dimension = 1024              [ok]
collection distance = cosine         [ok]

role distribution      literature 10456 | my_work 106 | future_chapters 188 | summaries 648
                       rules 59 | supervision 10 | guidance 2      (total 11469)
content types          body 9057 | table 1243 | summary_entry 648 | conclusion 289
                       abstract 178 | appendix 50 | caption 4 | references 0
evidence_eligible      true 10456 | false 1013
retrieval_eligible     true 11469
OCR chunks             104

no references point indexed          [ok]   0 points
no missing payload field             [ok]
only literature is evidence-eligible [ok]   10456 true == 10456 literature
my_work / future_chapters present    [ok]   both retrievable, both non-evidence
OCR chunk present                    [ok]   104 chunks
all points retrieval-eligible        [ok]
```

Notice `evidence_eligible = true` is *exactly* the literature count (10,456): no
`my_work`, `summaries`, `supervision`, `rules`, `guidance` or `future_chapters` point
is evidence-eligible anywhere in the corpus, which is the RAG-007 policy preserved
end-to-end. `references = 0` confirms the 3,317 reference chunks stayed out of the
normal collection.

**Duplicated-draft role safety** (the RAG-007 §25 correction, on the full corpus):

```
second-paper-draft-final.docx -> 53 points
  corpus_role = my_work            [ok]
  retrieval_eligible = true        [ok]
  evidence_eligible = false        [ok]
```

The same bytes also exist under `data/literature/`, and the role is still resolved
from all physical locations: the draft is retrievable and never evidence.

**20 sampled points** (deterministic: every 571st id of the sorted id list), each
verified for vector presence, dimension, payload completeness, `text`,
`retrieval_text`, and identity:

```
vector present in every sample                     [ok]
vector dimension = 1024 in every sample            [ok]
payload complete in every sample                   [ok]
source text present in every sample                [ok]
retrieval_text present in every sample             [ok]
point id = deterministic RAG-007 chunk id          [ok]   recomputed via chunk_id_for(file_hash, chunk_index, text)
document id = uuid5(document namespace, file_hash) [ok]   recomputed via document_id_for(file_hash)
```

### Idempotency (second unchanged run)

Artifact: `docs/rag008-indexing-reindex.txt` — the same command over the unchanged
corpus, with the manifest left untouched so the full-run manifest stays the artifact
of record:

```
points before:        11469
points after:         11469
inserted:             0
duplicate ids:        0
updated/upserted:     11469     (upsert by design; the same deterministic point ids)
stale points:         0
cache hits:           11469
cache misses:         0
actual GPU embeddings computed: 0
embedding seconds:    0.99        (cache serving only, 11,631 lookups/s)
upload seconds:       70.0
total wall-clock:     1136.8 s (18.9 min, dominated by the RAG-007 pass)
peak reserved VRAM:   2.13 GiB
corpus fingerprint:   3b1f63cb00ebca341587ad7a7daaa060a1ca7d6cc17879b9c9cea615853637f3  (identical)
document hashes digest: 42edf2f8b939e8d89b1411c9febbe485e7f1dcb03a14afafb548c856913af899  (identical)
chunk ids digest:       74a200acb811a0f2961bf39f8bf97f9578a5e1f6993a6f035e98ab9bcbab4d86  (identical)
embedding failures:   0
upload failures:      0
```

The second run recomputes **nothing**: all 11,469 vectors come from the cache
(model revision + chunk id + retrieval-text hash all match), so the GPU work is zero
and the only cost is re-chunking the corpus and re-upserting the same deterministic
points. No duplicate point appeared, and the corpus fingerprint, document-hash digest
and chunk-id digest are byte-identical to the full run.

## 24. Final state

| Requirement | Status |
|-------------|--------|
| embed `retrieval_text`, never the source text | ✅ payload keeps `text` and `retrieval_text` separate; the embedding input is `retrieval_text` |
| deterministic document identity | ✅ `payload["document_id"] == uuid5(document namespace, file_hash)`, regression-tested |
| embedding wrapper, no scattered model calls | ✅ `app/embedding/` behind the `EmbeddingModel` protocol |
| BGE-M3 loading: CUDA/CPU, batch size, offline reuse, revision reported | ✅ §4 |
| normalized vectors for cosine, verified numerically | ✅ norm 1.000000, collection distance Cosine |
| dimension discovered, not hard-coded; collection matches | ✅ 1024 read from the model and asserted against the collection |
| index only retrieval-eligible chunks | ✅ 11,469 of 14,786 indexed corpus-wide; 3,317 reference chunks excluded but kept in the chunk corpus |
| full corpus indexed (approved configuration) | ✅ §23: 11,469 points exactly, 0 failures, CUDA, batch 16, cosine 1024 |
| validation in a separate process | ✅ §23: persisted count/dimension/distance, role+OCR+type+evidence filters, 0 references, 20 sampled points |
| duplicated-draft role safety on the full corpus | ✅ §23: `second-paper-draft-final.docx` = 53 points, `my_work`, retrieval true, evidence false |
| idempotent second run | ✅ §23: 11,469 → 11,469, inserted 0, duplicates 0, stale 0, cache hits 11,469, identical fingerprint |
| complete payload for RAG-009 filtering and provenance | ✅ §9, verified per point |
| role safety preserved (`my_work` never evidence) | ✅ §19 and the payload tests |
| deterministic point ids, idempotent re-index | ✅ §20 |
| collection naming/versioning + manifest | ✅ §12, `docs/rag008-index-manifest.json` |
| persistent Qdrant; server URL **and** local path | ✅ local store verified across a process restart |
| cosine collection with payload indexes | ✅ created 1024/Cosine; 6 indexes configured |
| independent embedding/upload batch sizes, streaming | ✅ §6 |
| GPU usage reported | ✅ §22: RTX 5090, both batch-size phases, throughput, VRAM, utilisation; production batch size 16 selected |
| idempotent indexing with before/after/inserted/updated/stale | ✅ §20 |
| stale handling with explicit `--delete-stale` | ✅ §20 |
| deterministic corpus fingerprint | ✅ stable across re-runs, order- and timestamp-independent |
| embedding cache with correct invalidation | ✅ §20 and tests 19-21 |
| failure handling (abort on model/dimension, report the rest) | ✅ §13 |
| indexing CLI with the required options | ✅ §14 |
| embedding inspection CLI | ✅ `app.embedding.inspect`, prints the sample norm |
| report + machine-readable artifact | ✅ this file, `docs/rag008-indexing.txt`, the manifest |
| all 27 numbered tests plus the existing suite | ✅ 416 passed (362 pre-existing + 54 new) |
| validation order: dry run → 100-chunk pilot → persistence → tokenizer measurement → GPU benchmark → full index → validation → idempotency | ✅ §16-§23 |
| full-corpus point count exactly 11,469 | ✅ §23 |
| no truncation, no re-chunking, no sparse/ColBERT vectors, no reranker, no RAG-009 logic | ✅ none present |
| existing 416 tests still pass | ✅ 416 passed |

```
RAG-008 COMPLETE
```

`phd_rag_chunks_v1` holds 11,469 deterministic points (1024-dim, cosine, CUDA-built),
covering every retrieval-eligible RAG-007 chunk with its role, evidence policy, page
and section provenance; the 3,317 reference chunks remain in the chunk corpus and out
of the retrieval collection. RAG-009 has not been started.

Artifacts: `docs/rag008-indexing.txt`, `docs/rag008-indexing-reindex.txt`,
`docs/rag008-dryrun-tokens.txt`, `docs/rag008-pilot-collection.txt`,
`docs/rag008-pilot-points.txt`, `docs/rag008-stale-{seed,retained,deleted}.txt`,
`docs/rag008-index-manifest.json`, `docs/rag008-stale-demo-manifest.json`,
`docs/rag008-gpu-benchmark.txt`.

