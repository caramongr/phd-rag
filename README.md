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

### DeepSeek as a second model for the chat (RAG-016)

The simple chat can be answered by the local model or by the DeepSeek API, chosen per message
in the composer's **Model** list. DeepSeek is optional: without a key the chat is local-only
and the cloud entry is shown greyed out. Nothing is uploaded and there is no automatic
fallback — the cloud is used only when you select it, and only the context already selected for
that one message is sent.

```
# Temporary (this shell only)
$env:DEEPSEEK_API_KEY="sk-…"

# Persistent (Windows). Affects NEW processes and terminals only: close and reopen
# the terminal, then start the interface again.
setx DEEPSEEK_API_KEY "sk-…"
```

| Variable | Default | What it is |
|---|---|---|
| `DEEPSEEK_API_KEY` | *(unset)* | the key; unset means the cloud provider is unavailable. Server-side only |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | the API endpoint |
| `DEEPSEEK_MODEL` | `deepseek-flash` | the default DeepSeek model (shown as *DeepSeek V4.1 Flash*) |
| `DEEPSEEK_MODELS` | `deepseek-v4-pro` | further models the selector offers (*DeepSeek V4 Pro*), comma-separated |
| `PHD_RAG_DEEPSEEK_TIMEOUT` | `300` | seconds for one cloud answer |
| `PHD_RAG_DEEPSEEK_NUM_CTX` | `1000000` | the provider window the context builder budgets against |
| `PHD_RAG_DEEPSEEK_CONTEXT_TOKENS` | `96000` | input budget for one cloud turn |
| `PHD_RAG_DEEPSEEK_MAX_OUTPUT_TOKENS` | `4096` | room kept for the cloud answer |

The selector shows the default model first and the rest after it, so the two ids above appear as
*DeepSeek V4.1 Flash* and *DeepSeek V4 Pro*. The window is a ceiling, not a target: a turn sends
the input budget (96,000 tokens), never a million.

Each variable is also read with the project's `PHD_RAG_` prefix (`PHD_RAG_DEEPSEEK_API_KEY`
wins over `DEEPSEEK_API_KEY`). The key is never written to the database, returned by any
endpoint, included in an error message or reported in the logs; the model ids and the
provider's own `/models` answer are what the selector shows, so no name is hard-coded.

## Web interface

One command starts the local interface — a FastAPI backend serving a static HTML/CSS/JS page,
with no build step and no framework:

```powershell
.\.venv\Scripts\python.exe -m app.web
# PhD RAG local interface
#   open: http://127.0.0.1:8000/
#   model: qwen3.5:35b
#   stop: Ctrl+C
```

Then open <http://127.0.0.1:8000/>: choose a mode, type a question or paste a paragraph of
your thesis, and read the answer. Sources (with the role each block was allowed to play, its
scores, chunk id and provenance), validation details, external discoveries, warnings and the
raw audit are all expandable, and `--host 0.0.0.0` is required to leave the machine (the
default is `127.0.0.1`, and the banner says so when it is changed). `python -m app.web --help`
lists the two other options (`--port`, `--model`).

The interface is a renderer over the mode service: `POST /api/mode` runs one mode through
RAG-013 and returns its `ModeResult` (plus a transport-only `notice` with a friendly message
when a run needs explaining), `GET /api/modes` serves the mode copy the selector shows, and
`GET /api/status` reports Ollama and index availability without loading a model.

The default screen is the simple chat (`GET`/`POST /api/chats…`), whose **Model** list chooses
the provider for the next message — the local Ollama model or DeepSeek — while the conversation
itself always lives in the local SQLite database. `GET /api/chat/info` serves that list.

For day-to-day use there is a manual with screenshots: [docs/manual.md](docs/manual.md) — what
each mode is for, how to read a result, the advanced options, everyday recipes and a
troubleshooting table. Keyboard shortcuts: `Ctrl+Enter` runs, `Esc` clears.

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

# RAG-012: validate every claim against the source it cites and convert the internal
# [Sx] markers into Harvard in-text citations plus a deduplicated reference list.
# Unsupported claims are withheld from the validated answer; partial and uncertain
# claims keep their citation with a visible annotation.
.\.venv\Scripts\python.exe -m app.citations.inspect "What is AI governance in higher education?"

# RAG-012: the claim-by-claim audit, the source map, warnings only, the untouched
# RAG-011 answer, and machine-readable output
.\.venv\Scripts\python.exe -m app.citations.inspect "..." --show-claims --show-source-map
.\.venv\Scripts\python.exe -m app.citations.inspect "..." --show-warnings --show-original
.\.venv\Scripts\python.exe -m app.citations.inspect "..." --json

# RAG-012: escalate only the claims the deterministic checks cannot decide to the local
# BGE-M3 model, or tighten the support band for a stricter run
.\.venv\Scripts\python.exe -m app.citations.inspect "..." --semantic --support-coverage 0.7

# RAG-013: the five academic working modes. A mode is an orchestration preset over the
# pipeline above: EVIDENCE and WRITING retrieve, generate and validate; CONSISTENCY,
# SUPERVISOR and RESEARCH never call the model at all.
.\.venv\Scripts\python.exe -m app.modes.inspect evidence "What is AI governance in higher education?"
.\.venv\Scripts\python.exe -m app.modes.inspect writing "Write a short PhD paragraph on AI governance in higher education."
.\.venv\Scripts\python.exe -m app.modes.inspect consistency "Compare my existing treatment of AI governance with the literature."
.\.venv\Scripts\python.exe -m app.modes.inspect supervisor "What guidance do I have about methodology?"
.\.venv\Scripts\python.exe -m app.modes.inspect research "Find recent papers on AI governance in universities."

# RAG-013: sources with the role each block was allowed to play, the full audit trail
# (preset, policy note, claim verdicts in the generating modes), and machine-readable output
.\.venv\Scripts\python.exe -m app.modes.inspect evidence "..." --show-sources --show-audit
.\.venv\Scripts\python.exe -m app.modes.inspect research "..." --json
```

The web interface in RAG-014 (`app/web/`) is a renderer over that service, not a second
pipeline: it owns one `ModeService` for the process (built on the first request, used one run
at a time, closed on shutdown), exposes it as two JSON endpoints, and renders the result in
static HTML/CSS/JavaScript. Every value a document, a model or a provider supplies is written
with `textContent`, so markup inside a source is displayed as text; the page contains no
server-side templating at all. Nothing about a mode is decided in the frontend — the mode
list, the sections, the findings, the labels and the verdicts all come from the `ModeResult`,
so the UI and the CLI cannot disagree. Failures arrive as JSON with a readable message
(`invalid_request` for a rejected request, `internal_error` for anything unexpected) and a
`notice` that explains, in one sentence, what a person should do about a missing model, an
unreachable Ollama, a locked index, a provider outage, no results, insufficient evidence or a
withheld claim — with the technical detail kept in a collapsed debug panel.

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

Citation validation in RAG-012 (`app/citations/`) answers the question RAG-011 left open:
does the cited source actually contain the claim? No model is regenerated and no text is
rewritten by a language model. The deterministic checks run first and are never
overridden — the cited id exists, the chunk is evidence-eligible and not a reference
list, its text is present, every direct quotation occurs verbatim in the cited source, and
every numeral (percentage, p-value, sample size, range, year, count) must appear in the
cited evidence or be directly derivable from it. Only then does the support layer decide,
from how much of the claim the evidence contains and whether quantified findings from
different passages were merged into one sentence (the RAG-011 failure mode). Verdicts are
`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED` or `UNCERTAIN`, each recorded with its
method, confidence and reason, and `UNCERTAIN` is never silently upgraded. Claims resting
on scanned text or a table are flagged `elevated_validation_risk` and checked against
stricter thresholds rather than rejected. An optional local BGE-M3 layer can be enabled
for the claims the deterministic checks cannot decide, and it can never turn partial
support into full support.

The stage is a *policy*, applied consistently and visible in the output: an unsupported
claim is withheld (replaced by `[claim withheld: the cited source does not support it]`
and listed in the warnings), a partial or uncertain claim keeps its citation with a
visible `[partially supported]` / `[unverified]` annotation, a report about the author's
own draft keeps its wording and gets a provenance note instead of a citation, and the
RAG-011 answer is preserved unchanged in `original_answer`. Harvard output is built only
from metadata that is present: `(Smith, 2024)`, `(Smith and Jones, 2024)`,
`(Smith et al., 2024)`, with `n.d.` for a missing year and a quoted short title for a
missing (or unusable) author, each fallback recorded per citation. Markers of one
sentence are ordered alphabetically by author then year, and references are one
deduplicated entry per *work* — DOI first, then normalized title, then document — never
per chunk, so eight chunks of one paper produce one bibliography entry. Nothing is
invented: no volume, issue, publisher, page range, URL or DOI that the metadata does not
carry, and any gap is reported as a `metadata_warning` on the entry.

The five academic working modes in RAG-013 (`app/modes/`) are orchestration presets over
that pipeline, not a new pipeline. Each mode is one frozen `ModeConfig` — retrieval mode,
generation mode, context flags, output style and a `context_policy` mapping every corpus
role to the role it may play — and `ModeService.run_mode` calls the existing services and
nothing else. The safety rules are data: an undeclared corpus role resolves to
`INSTRUCTIONS_ONLY`, a `ModeSource` counts as evidence only when its role is `EVIDENCE`
**and** the corpus marked it `evidence_eligible`, and every external record is typed
`usable_as_evidence=False`. EVIDENCE retrieves evidence only and ends in a validated
Harvard-cited answer; WRITING adds the author's own draft and the project rules as
non-citable context, which RAG-012 then refuses to cite; CONSISTENCY compares the draft
with the literature and the rules using a deterministic engine (terminology, duplication,
unsupported claims, numeric discrepancies, style) and never asks the model, because a model
asked to judge the author's text would invent findings; SUPERVISOR labels supervision and
guidance excerpts, cites nothing and generates nothing; RESEARCH runs RAG-010 discovery
beside local retrieval and keeps local evidence, new external papers, works already in the
corpus and possible matches in four separate sections. The common `ModeResult` is the same
shape for every mode, and the policy that produced a run travels inside it
(`audits["policy_note"]`), so a saved result can be checked without the documentation. CLI
exit codes: `1` model error, `3` at least one unsupported claim was withheld.

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
