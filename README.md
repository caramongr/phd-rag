# phd-rag

PhD Academic RAG — backend for ingesting and parsing a personal academic corpus.

## Layout

| Path | Contents |
|------|----------|
| `app/` | FastAPI backend: ingestion (scanning, hashing, folder/topic hints), parsing (PDF via PyMuPDF with a selective Tesseract OCR fallback, DOCX, Markdown), metadata (local academic-metadata extraction with provenance, plus controlled Crossref/OpenAlex enrichment), chunking (academic-aware normalization and structure-preserving chunks with page/section/OCR provenance), embedding (dense BGE-M3 vectors with a local cache), indexing (deterministic Qdrant points for the retrieval-eligible chunks), retrieval (evidence-policy-filtered dense search plus a local BGE cross-encoder rerank), discovery (read-only external scholarly search, kept separate from trusted evidence) and generation (grounded answers from retrieved evidence through a local Ollama model, with source markers and a grounding audit) |
| `tests/` | pytest suite (external metadata APIs, the OCR engine, the embedding model and Qdrant are all mocked or run in Qdrant's local mode, so the suite needs no network access, no GPU and no Tesseract) |
| `RAG-00N.md` | Task briefs |
| `RAG-00N-report.md` | End-of-task reports |
| `docs/` | Project documentation and validation reports |

## Setup

Requires Python 3.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install pymupdf python-docx pytesseract Pillow
.\.venv\Scripts\python.exe -m pytest -q
```
`pyproject.toml` declares the runtime dependencies for RAG-001…RAG-008:
`fastapi`, `pydantic`, `requests`, `pymupdf`, `python-docx`, `pytesseract`,
`Pillow`, `numpy`, `torch`, `transformers`, `sentence-transformers`,
`huggingface-hub` and `qdrant-client`. The test-only tools (`pytest`, `httpx` via
the FastAPI test client) are still installed separately.

The OCR fallback needs a local Tesseract installation (never downloaded
automatically, no cloud OCR). Configuration lives in `app/config.py` and is
overridable through `PHD_RAG_*` environment variables:

```
PHD_RAG_OCR_ENABLED=true
PHD_RAG_OCR_LANGUAGE=eng
PHD_RAG_OCR_DPI=300
PHD_RAG_OCR_MIN_NATIVE_CHARS=50
```

Metadata enrichment (RAG-006) is configured the same way. Set
`PHD_RAG_METADATA_ENRICH_EXTERNAL=false` for a fully offline run:

```
PHD_RAG_METADATA_ENRICH_EXTERNAL=true
PHD_RAG_METADATA_CACHE_PATH=.metadata_cache/cache.json
PHD_RAG_METADATA_REQUEST_TIMEOUT=10.0
PHD_RAG_METADATA_MAX_RETRIES=2
PHD_RAG_METADATA_MIN_REQUEST_INTERVAL=1.0
PHD_RAG_METADATA_CROSSREF_MAILTO=you@example.org
```

`PHD_RAG_METADATA_CROSSREF_MAILTO` is the contact identity sent to the bibliographic
APIs: Crossref receives it as a `User-Agent` suffix, OpenAlex as the same suffix plus the
`mailto` parameter. When it is unset, no address is invented or sent. Note that OpenAlex
additionally enforces a free daily budget shared by everyone on the same IP address: once
it is used up the API answers `429` and asks for an API key, so such a run simply loses
its OpenAlex fallback lookups (Crossref is always queried first, so no identity is
affected); the budget resets at midnight UTC.

Embeddings and the vector index (RAG-008) are configured the same way:

```
PHD_RAG_EMBEDDING_MODEL=BAAI/bge-m3
PHD_RAG_EMBEDDING_DEVICE=cpu            # unset = CUDA when available, else CPU
`PHD_RAG_EMBEDDING_BATCH_SIZE=16        # measured optimum for BGE-M3 on this corpus
PHD_RAG_EMBEDDING_CACHE_PATH=.embedding_cache/embeddings.npz
PHD_RAG_EMBEDDING_LOCAL_FILES_ONLY=false
PHD_RAG_QDRANT_COLLECTION=phd_rag_chunks_v1
PHD_RAG_QDRANT_URL=                      # set for a Qdrant server
PHD_RAG_QDRANT_API_KEY=
PHD_RAG_QDRANT_PATH=.qdrant_data         # local persistent index (default)
PHD_RAG_QDRANT_UPLOAD_BATCH_SIZE=128
PHD_RAG_INDEX_MANIFEST_DIR=.index_cache/manifests
```

The embedding model is downloaded once into the Hugging Face cache and is re-used
locally afterwards (`PHD_RAG_EMBEDDING_LOCAL_FILES_ONLY=true` makes a run strictly
offline). An explicitly requested device is never silently swapped: `--device cuda`
on a machine without CUDA fails instead of quietly running on the CPU. The default
device is "CUDA when available, else CPU", so a plain run uses the GPU on this
machine.

## CLI

```powershell
# Parse one document and print a summary
.\.venv\Scripts\python.exe -m app.parsing.parse "paper.pdf"

# Native extraction only
.\.venv\Scripts\python.exe -m app.parsing.parse "paper.pdf" --no-ocr

# Report local OCR availability, version and installed languages
.\.venv\Scripts\python.exe -m app.parsing.parse --ocr-info

# Inspect academic metadata and provenance for one document (offline-safe)
.\.venv\Scripts\python.exe -m app.metadata.inspect "paper.pdf" --offline
.\.venv\Scripts\python.exe -m app.metadata.inspect "paper.pdf" --show-candidates
# Resolve the corpus role across every physical copy found under a corpus root
.\.venv\Scripts\python.exe -m app.metadata.inspect "paper.pdf" --corpus "data"

# Offline metadata coverage across the corpus; optional external sample of N docs
.\.venv\Scripts\python.exe -m app.metadata.report "data"
.\.venv\Scripts\python.exe -m app.metadata.report "data" --sample 20

# Stage 1: DOI-only corpus enrichment (no title-based searches)
.\.venv\Scripts\python.exe -m app.metadata.doi_stage "data" --out "docs\rag006-doi-stage.txt"

# Stage 2A: title-based enrichment *validation sample* (no DOI lookups)
.\.venv\Scripts\python.exe -m app.metadata.title_stage "data" --size 30 --out "docs\rag006-stage2a-sample.txt"

# Stage 2B: title-based enrichment for the whole eligible DOI-less corpus
.\.venv\Scripts\python.exe -m app.metadata.title_stage "data" --all --out "docs\rag006-stage2b.txt"

# RAG-007: inspect the chunks (and provenance) of one document
.\.venv\Scripts\python.exe -m app.chunking.inspect "paper.pdf" --show-text --show-locators
# ... with the role resolved across all physical copies under a corpus root
.\.venv\Scripts\python.exe -m app.chunking.inspect "paper.pdf" --corpus "data"

# RAG-007: full-corpus chunking report (read-only)
.\.venv\Scripts\python.exe -m app.chunking.report "data" --out "docs\rag007-chunking.txt"

# RAG-008: inspect the embedding model (identity, dimension, normalization)
.\.venv\Scripts\python.exe -m app.embedding.inspect --show-vector 8

# RAG-008: dry run - intended indexing population, no embeddings, no Qdrant writes
.\.venv\Scripts\python.exe -m app.indexing.build "data" --dry-run

# RAG-008: pilot indexing (about N chunks) into a local persistent Qdrant index
.\.venv\Scripts\python.exe -m app.indexing.build "data" --pilot 100 --report-out "docs\rag008-indexing.txt"

# RAG-008: inspect the indexed collection, including the pilot assertions
.\.venv\Scripts\python.exe -m app.indexing.inspect --check --show-text

# RAG-009: retrieve ranked chunks (evidence mode by default, reranked)
.\.venv\Scripts\python.exe -m app.retrieval.inspect "artificial intelligence governance in higher education"

# RAG-009: other modes - general, my_work, supervision, guidance (non-evidence)
.\.venv\Scripts\python.exe -m app.retrieval.inspect "what did my supervisor say about methodology?" --mode supervision
.\.venv\Scripts\python.exe -m app.retrieval.inspect "what have I written about the historical development of AI?" --mode my_work

# RAG-009: dense-only ranking, explicit filters and full text/quote output
.\.venv\Scripts\python.exe -m app.retrieval.inspect "virtual reality in higher education" --no-rerank --year-min 2019 --content-type body --show-text --show-locators

# RAG-009: reference lists are withheld in evidence/my_work/general mode by default;
# this returns them on purpose (citation analysis only)
.\.venv\Scripts\python.exe -m app.retrieval.inspect "references on adaptive learning" --mode my_work --include-bibliography

# RAG-009: the evaluation set (Hit@5/Hit@10/MRR@10, leakage checks) and latency benchmark
.\.venv\Scripts\python.exe -m app.retrieval.evaluate --out "docs\rag009-evaluation.txt" --latency-out "docs\rag009-latency.txt"

# RAG-010: external scholarly discovery (Crossref + OpenAlex). Read-only: nothing is
# downloaded into data/, embedded, or added to Qdrant. The default output groups the
# results into ALREADY IN LOCAL CORPUS / NEW-VERIFIED / DISCOVERY ONLY / CONFLICTS.
.\.venv\Scripts\python.exe -m app.discovery.search "AI governance in higher education"

# RAG-010: recent work only, hiding what the corpus already has
.\.venv\Scripts\python.exe -m app.discovery.search "generative AI academic integrity" --year-min 2025 --new-only

# RAG-010: identity lookups, and optional open-access full-text verification
.\.venv\Scripts\python.exe -m app.discovery.search --doi 10.1016/j.compedu.2019.103778
.\.venv\Scripts\python.exe -m app.discovery.search --title "Artificial intelligence in higher education"
.\.venv\Scripts\python.exe -m app.discovery.search --doi 10.1016/j.compedu.2019.103778 --verify-fulltext

# RAG-010: local evidence plus external discovery for one query, as two separate
# sections (never one fused ranking, never a generated answer)
.\.venv\Scripts\python.exe -m app.discovery.research "AI governance in higher education" --local-k 8 --year-min 2025

# RAG-011: answer a question from the retrieved evidence with the local Qwen model
# (qwen3.5:35b through Ollama). Evidence mode by default; every claim sentence carries
# an [S1] marker and the answer is audited afterwards.
.\.venv\Scripts\python.exe -m app.generation.inspect "What is AI governance in higher education?"

# RAG-011: show exactly what the model saw and the prompts that were used
.\.venv\Scripts\python.exe -m app.generation.inspect "..." --show-context --show-prompt --show-audit

# RAG-011: add the author's own drafts (or supervision guidance) as non-citable
# continuity context - never as evidence - and control the answer length
.\.venv\Scripts\python.exe -m app.generation.inspect "What have I written about AI history?" --include-my-work
.\.venv\Scripts\python.exe -m app.generation.inspect "..." --include-my-work --include-guidance --length detailed

# RAG-011: tighter context budget, an evidence score floor, machine-readable output
.\.venv\Scripts\python.exe -m app.generation.inspect "..." --context-tokens 2000 --min-rerank-score 0.3
.\.venv\Scripts\python.exe -m app.generation.inspect "..." --json
```

Generation in RAG-011 (`app/generation/`) answers *only* from what RAG-009 retrieved,
with a source marker on every claim sentence. `app/generation/ollama.py` is the single
place that talks to Ollama; failures are typed (`model_not_found`, `timeout`,
`empty_response`, ...) and there is no silent retry, so a degraded model shows up as
`model_error` rather than as a plausible answer. The context is built from whole
chunks only (never truncated mid-chunk) under a deterministic two-phase budget: the
evidence is capped at `budget - 1500` when the caller explicitly asks for
`my_work`/`guidance`, so an explicit request cannot be starved by a large evidence set,
and if nothing citable fits the service refuses without calling the model at all.

Grounding is checked after generation, never by fixing the text: `invalid_markers`
(a cited id that was never supplied), `unmarked_claims` (a finding-shaped sentence with
no marker), `non_evidence_citations` (a claim resting only on the author's own work or
guidance), `external_finding_claims` (findings attributed to a metadata-only external
record) and unfindable quotations. Any of the first, third or fourth makes the answer
`grounding_warnings` instead of `ok`. Non-evidence material is labelled and stays
non-citable: `my_work` and `guidance` blocks can be reported ("your draft states ...")
but never cited as evidence about the world, and RAG-010 discovery records stay external
(metadata-only records carry no abstract precisely so their findings cannot leak in).
Source text is delimited and declared untrusted, because a document may contain
instructions; the model must treat them as data.

Retrieval in RAG-009 is two-stage and evidence-safe: a BGE-M3 dense search over
`phd_rag_chunks_v1` returns candidates that are filtered **at the vector-search
level** (and re-checked afterwards) so non-evidence material - the author's own
drafts, summaries, supervision notes, writing rules, future chapters - can never
appear in `--mode evidence`, followed by a local `BAAI/bge-reranker-v2-m3`
cross-encoder rerank, near-duplicate suppression and a per-work result cap (the
corpus holds several copies of some papers, so if the first round is consumed by
duplicates the search widens by bounded rounds instead of returning a short list).
The dense score and the rerank score are kept separately (they are not the same
scale) and no threshold, query rewriting or generation is applied: RAG-009 returns
ranked chunks with their provenance, and stops there.

Reference lists that RAG-007 did not classify as references are withheld at retrieval
time by a deterministic, high-confidence detector (`app/retrieval/bibliography.py`):
it needs several agreeing bibliographic signals (repeated `Available at:`, DOIs,
journal furniture, numbered entries, a `References` heading, very high marker density
with reference furniture) and never fires on ordinary prose that cites sources, so it
can be turned off with `--include-bibliography` for citation analysis.

External discovery in RAG-010 (`app/discovery/`) is deliberately kept apart from that
evidence path. It searches Crossref and OpenAlex over HTTP, normalizes each provider
response into one `ExternalWork` record with field-level provenance, merges the same
paper found twice, marks material conflicts instead of silently choosing a winner,
compares every result against the local corpus and ranks deterministically (query
relevance, abstract relevance, metadata completeness and a bounded recency nudge -
never citation count). External records are **not evidence and are never imported**:
no download into `data/`, no embeddings, no Qdrant writes, no promotion into
`--mode evidence`. The combined command prints the trusted local chunks and the
external discoveries as two separate sections and generates no prose answer. Every
request is cached (TTL plus an explicit `--refresh`) and audited, provider failures are
reported as provider failures rather than as "no results", and the optional
`--verify-fulltext` fetch only follows public `http(s)` open-access URLs that were
re-validated at every redirect hop, with a size cap and a content-type allowlist.

Provider etiquette: no personal address is hard-coded. Set `PHD_RAG_CROSSREF_MAILTO`
(and/or `PHD_RAG_OPENALEX_EMAIL`, `PHD_RAG_OPENALEX_API_KEY`) to identify this client
to the APIs' polite pools; without a configured contact the tools say so instead of
inventing one.

Chunking targets roughly 700–1000 tokens per chunk with a hard maximum of 1200
(1500 for an intact table), splits on academic structure rather than fixed
windows, and records page, section, source-locator and OCR provenance on every
chunk. A document that exists in several corpus folders (identical SHA-256 bytes)
stays one logical document: its corpus role is resolved across *all* physical
locations, and when those disagree the most conservative role wins, so a copy under
`literature/` can never turn the author's own work, a summary or a supervision note
into scholarly evidence. Spreadsheets (`.xlsx`) are **not** chunked in RAG-007: flattening cells
into prose would be wrong, so they are reported as unsupported and counted
rather than silently dropped.

## Not tracked

`data/` is excluded from git: it holds the document corpus (third-party published
PDFs plus personal drafts and supervision material), which is not redistributed.
