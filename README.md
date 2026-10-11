# phd-rag

PhD Academic RAG — backend for ingesting and parsing a personal academic corpus.

## Layout

| Path | Contents |
|------|----------|
| `app/` | FastAPI backend: ingestion (scanning, hashing, folder/topic hints), parsing (PDF via PyMuPDF with a selective Tesseract OCR fallback, DOCX, Markdown), metadata (local academic-metadata extraction with provenance, plus controlled Crossref/OpenAlex enrichment), chunking (academic-aware normalization and structure-preserving chunks with page/section/OCR provenance), embedding (dense BGE-M3 vectors with a local cache), indexing (deterministic Qdrant points for the retrieval-eligible chunks) and retrieval (evidence-policy-filtered dense search plus a local BGE cross-encoder rerank; no generation) |
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
```

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
