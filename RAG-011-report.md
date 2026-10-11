# RAG-011 — Grounded answer generation with a local Qwen model

Status: **complete** — questions are now answered in prose by the local `qwen3.5:35b`
through Ollama, strictly over material retrieved by the RAG-009 pipeline, with a `[S1]`
marker on every claim sentence, a grounding audit that would expose an unsupported claim,
and record-level separation between corpus evidence, the author's own work, supervision
guidance and RAG-010 external records. 112 new tests; **769 tests pass** in total.

Scope guard: RAG-011 retrieves, selects, renders context, generates and audits. It does
**not** validate or format Harvard citations, does not import anything into `data/`, does
not re-index, does not build academic answer modes, does not touch the Qdrant collection
and does not begin RAG-012.

---

## 1. Files created

| Path | Purpose |
|------|---------|
| [app/generation/__init__.py](app/generation/__init__.py) | Package exports (31 names) |
| [app/generation/models.py](app/generation/models.py) | `GenerationMode`, `SourceClass`, `OutputLength`, `GenerationStatus`, `SourceBlock`, `DroppedBlock`, `ContextPackage`, `QuotationEvidence`, `SourceMarkerAudit`, `ModelFailure`, `GenerationAudit`, `GenerationResponse` |
| [app/generation/ollama.py](app/generation/ollama.py) | `ChatMessage`, `GenerationOptions`, `ChatResult`, `GenerationModel` protocol, `OllamaClient` (`version`, `available_models`, `describe`, `chat`), `model_summary` — the only module that talks to Ollama |
| [app/generation/prompts.py](app/generation/prompts.py) | Template loading, `prompt_version()`, `build_system_prompt`, `build_user_prompt`, `PromptError` |
| [app/generation/templates/system_v1.txt](app/generation/templates/system_v1.txt) | First system prompt (superseded in this stage, kept as the recorded iteration) |
| [app/generation/templates/user_v1.txt](app/generation/templates/user_v1.txt) | First user prompt (superseded, kept) |
| [app/generation/templates/system_v2.txt](app/generation/templates/system_v2.txt) | Active system prompt: one marker per claim sentence, no consensus wording, untrusted-source rule, style rules |
| [app/generation/templates/user_v2.txt](app/generation/templates/user_v2.txt) | Active user prompt: question, source blocks, marker and refusal instructions |
| [app/generation/context.py](app/generation/context.py) | `estimate_tokens`, `ExternalSource`, `evidence_block`/`context_block`/`external_block`, `render_context`, `ContextBuildRequest`, `build_context` (deterministic budgeting) |
| [app/generation/grounding.py](app/generation/grounding.py) | `MARKER_RE`, abbreviation-protected `split_sentences`, `extract_markers`, `find_quotations`, `audit_markers`, `used_source_ids`, `grounding_warnings`, external-record and non-evidence checks |
| [app/generation/service.py](app/generation/service.py) | `GenerationService.answer(...)` — retrieve, budget, render, generate, audit, one typed result |
| [app/generation/inspect.py](app/generation/inspect.py) | `python -m app.generation.inspect` CLI |
| [app/retrieval/factory.py](app/retrieval/factory.py) | `build_retrieval_service(...)` — the shared RAG-009 builder now used by RAG-009's CLI, RAG-010 and RAG-011 |
| [tests/generation_support.py](tests/generation_support.py) | Offline harness: real `Candidate`/`RetrievalResponse` builders, `FakeRetrievalService`, `FakeGenerationModel` (records both prompts), `FakeExternalRecord` |
| [tests/test_generation_context.py](tests/test_generation_context.py) | Block conversion, source ids, relevance order, budgeting, drop recording, labels, external blocks (21 tests) |
| [tests/test_generation_ollama.py](tests/test_generation_ollama.py) | Request shape, configurable model, timeout, missing model, malformed/empty response, `think=false` (18) |
| [tests/test_generation_grounding.py](tests/test_generation_grounding.py) | Valid/invalid markers, marker-free claims, non-evidence citation, external metadata-only misuse, quotations, determinism (31) |
| [tests/test_generation_service.py](tests/test_generation_service.py) | End-to-end service behaviour over a fake model, modes, refusals, audit fields, prompt version (26) |
| [tests/test_generation_cli.py](tests/test_generation_cli.py) | Rendering, flags, JSON, exit codes (16) |
| [docs/rag011-generation-evaluation.txt](docs/rag011-generation-evaluation.txt) | The 11 real questions with retrieved sources, context, raw answers, markers, timings, tokens |
| [docs/rag011-grounding-validation.txt](docs/rag011-grounding-validation.txt) | Claim-by-claim manual evaluation, marker evaluation, thin questions, external records, injection test, quotations, budget and OCR |
| [docs/rag011-performance.txt](docs/rag011-performance.txt) | Latency and throughput for the three output lengths, plus the estimator calibration |
| [RAG-011-report.md](RAG-011-report.md) | This report |

## 2. Files modified

| Path | Change |
|------|--------|
| [app/config.py](app/config.py) | RAG-011 `generation_*` / `ollama_*` settings block and `PHD_RAG_*` environment overrides; nothing else touched |
| [app/retrieval/inspect.py](app/retrieval/inspect.py) | `build_service` now delegates to the shared factory (identical behaviour, one implementation) |
| [app/discovery/research.py](app/discovery/research.py) | RAG-010's combined CLI uses the same factory |
| [README.md](README.md) | Generation CLI, grounding rules, source-marker semantics |

No file under `data/` was modified, and the Qdrant collection was only read (see §19).
The vector payloads were written by RAG-008 and are unchanged by this stage.

## 3. Architecture

```
question
   │
   ├─ RAG-009 retrieval (app/retrieval/factory.build_retrieval_service)
   │     dense BGE-M3 → top 25 → cross-encoder BGE-reranker-v2-m3 → top 8
   │     bibliography guard active → bibliography-like chunks withheld
   │
   ├─ evidence selection (§4)
   │     evidence        (corpus_role=literature, evidence_eligible=True) → citable
   │     my_work         (evidence_eligible=False) → context only, never citable
   │     guidance        (rules / supervision)     → instructions, never citable
   │     external        (RAG-010 records)         → labelled, metadata-only = not citable
   │
   ├─ context construction (§5, §6)
   │     deterministic estimate → two-phase budget → whole blocks only → rendered text
   │
   ├─ generation (app/generation/ollama.OllamaClient → Ollama /api/chat)
   │
   ├─ grounding audit (app/generation/grounding)
   │     markers, invalid markers, unmarked claims, non-evidence citations,
   │     external finding claims, quotations, style warnings
   │
   └─ GenerationResponse (answer + sources + retrieval audit + generation audit + status)
```

The dependency direction is one way: `service` → `context`/`grounding`/`prompts`/`ollama`
→ `models`. `models.py` imports nothing from the package, and `ollama.py` is the only
module that performs HTTP. Retrieval is reached through the same shared factory the RAG-009
CLI already used, so RAG-011 cannot drift from the retrieval that RAG-009 and RAG-010
measured; the refactor was verified by the unchanged RAG-009 and RAG-010 suites.

Statuses are exhaustive and mutually exclusive: `ok`, `grounding_warnings`,
`insufficient_evidence`, `no_citable_context`, `model_error`.

## 4. Generation model and Ollama integration

All values below were read from the running Ollama instance (`/api/version`,
`/api/show`, `/api/tags`), not assumed:

| Property | Value |
|----------|-------|
| Ollama | 0.35.1 at `http://127.0.0.1:11434` |
| Model | `qwen3.5:35b` |
| Digest | `72aac7c28be8ff9b81c2a270eb489c88e6d822a76ec0cb7bf34d965768848567` |
| Format / family | `gguf` / `qwen35moe` (parent `qwen35-validation-20261004:35b-a3b-q4`) |
| Parameters | 34.7B (`general.parameter_count` = 34,660,610,688) |
| Experts | 256 total, 8 active per token (`expert_used_count`) |
| Quantization | `Q4_K_M` |
| On disk | 22,069,592,181 bytes (20.6 GiB), modified 2026-10-06 |
| Context length | 262,144 (requested 8,192) |
| Capabilities | completion, vision, tools, thinking |
| Sampling | temperature 0.1, top_p 0.9, seed 0, `think=false` |
| Output limit | 900 tokens |
| Timeout | 300 s |
| Keep-alive | 30 m |

`think=false` matters: this build exposes a reasoning channel, and with it enabled the
answer channel stays empty while the model reasons. Measurements taken with the reasoning
channel would not describe the answer path, so RAG-011 requests `think=false` and asserts
it in the request test.

Failures are typed, never stringly-matched: `model_not_found`, `invalid_request`,
`server_error`, `http_error`, `malformed_response`, `empty_response`, `timeout`,
`connection_error`, `interrupted`. There is deliberately **no automatic retry** — a
grounded answer that silently retried would hide a degraded service, so the failure is
returned with `status=model_error`. Timings come from Ollama's own counters
(`eval_count`, `eval_duration`, `prompt_eval_count`, `load_duration`), which is why
tokens/second below are real measurements rather than estimates.

## 5. Prompt version

`prompt_version()` returns the template names and a SHA-256 of their LF-normalised text,
so an edit can never leave a stale version behind. Current value:

```
system_v2:6ca8b9ca+user_v2:5f9685ea
```

Both the version and the SHA-256 of the exact prompt pair are recorded in every result
and printed by the CLI, so any answer in the artifacts can be tied to the prompts that
produced it. `--show-prompt` prints both prompts verbatim.

This stage iterated the prompt once, and the iteration is kept rather than hidden:

* **v1** put the citation instruction in the middle of the rules. Three of five evidence
  answers then opened with an unmarked framing sentence ("The literature on AI governance
  in higher education identifies several dimensions…") and were flagged
  `grounding_warnings`. The unmarked sentences were true but unsupported positions, so
  they were treated as defects.
* **v2** requires *every* claim sentence, including the first, to carry a marker, forbids
  "the literature agrees / there is consensus" constructions, and repeats the untrusted
  source rule. All five evidence answers then came back `ok`.

`system_v1.txt` / `user_v1.txt` remain in the tree because §33 requires a failure to be
recorded rather than erased.

## 6. Context format

Evidence blocks are rendered as delimited blocks, never as bare text:

```
SOURCES
<<<SOURCE S1
title: Generative AI in Higher Education; The ChatGPT Effect
author: Cecilia Ka Yuk Chan | year: 2024 | doi: - | pages: 175-176 | section: 148
role: literature | evidence_eligible: true
SOURCE_TEXT
...chunk text...
SOURCE_TEXT
SOURCE>>>
...
QUESTION
What is AI governance in higher education?
```

Each block carries only metadata that exists in the RAG-007 payload: title, first author,
year, DOI, pages, section, corpus role, evidence eligibility, OCR flag, content type. No
field is invented, and a missing value is rendered as `-` rather than dropped, so the
model cannot mistake absence for a different fact. `SOURCE_TEXT` markers are used inside
the block because a stored document may itself contain text that looks like a delimiter;
the system prompt states that everything between them is data.

The context text is a pure function of (candidates, mode, budget, prompt version). The
determinism test builds the same context from a reversed candidate list and asserts
identical ids, order and rendering; a second test varies dictionary insertion order.

## 7. Context budget policy

* The estimator is deterministic: `max(ceil(chars / 4.0), ceil(words * 1.33))`, plus 32
  tokens of per-block overhead. `truncated_blocks` is always empty: a block is included
  whole or dropped whole, never clipped.
* Budget default 6,000 estimated tokens against a 8,192-token window, leaving room for the
  question and a 900-token answer.
* Two-phase allocation. Evidence is capped at `budget − reserved`, where `reserved` is
  1,500 tokens **only when the caller explicitly asked for my_work or guidance context**.
  Non-evidence material then fills the remainder. This exists because of a defect found in
  validation: before the reservation, a large evidence set consumed the whole budget and an
  explicit `include_my_work=True` request was silently starved, which made the model refuse
  a question it could have answered from the author's own draft.
* Drops are recorded with a reason: `budget`, `block_too_large` or
  `below_min_rerank_score`, and also surfaced as warnings.
* `min_generation_rerank_score` defaults to `None`. RAG-009 showed weak tail rerank scores,
  and no threshold that would remove bad evidence without removing good evidence was found,
  so none is applied; the parameter exists and is tested, but nothing is enabled by
  pretending a number was justified.
* If nothing citable fits, the service returns `no_citable_context` with a fixed refusal
  and **does not call the model at all** — there is nothing to ground on, so generating
  would be an invitation to invent.

Measured: 5,350–5,928 estimated tokens for a typical 8-chunk evidence context; the real
prompt is 5,269–6,486 tokens. Estimated/my_work run: evidence used 4,298 of its 4,500
cap, non-evidence 983 of 1,500 reserved, total 5,281 of 6,000.

## 8. Source-marker design

* Blocks are numbered `S1…Sn` in the order the model receives them, which is relevance
  order: rerank score descending, then dense score, then chunk id. The ids are therefore
  deterministic and reproducible for the same evidence.
* The model must cite the id it used for each claim sentence: `… [S1]`, and `[S1,S4]` when
  a sentence rests on several sources. It may not invent an id, may not renumber, and may
  not cite a block that was not supplied.
* The `source_map` in the result maps every id back to chunk id, document id, filename,
  title, author, year, DOI, pages, section, corpus role, `evidence_eligible`,
  `ocr_involved` and `content_type`. Bookkeeping is done by the service, not the model:
  even if the model wrote the wrong id, the record of what was supplied is intact.
* Only citable blocks receive an id that can support a claim. Non-citable blocks
  (my_work, guidance, metadata-only external records) may be cited for existence or
  continuity, and such sentences are counted separately (see §9).
* Markers are validated after generation, never "repaired". An id that does not exist in
  the supplied context is recorded in `invalid_markers` and raises the status; the answer
  text is left exactly as the model produced it.

**Measured: 0 invalid markers in all 11 evaluated answers (53 markers total).**

## 9. Grounding rules

The audit distinguishes five different problems, because they need different responses:

| Field | Meaning | Consequence |
|-------|---------|-------------|
| `invalid_markers` | cited id was not supplied | status → `grounding_warnings` |
| `unmarked_claims` | a finding-shaped sentence with no marker | status → `grounding_warnings` |
| `unmarked_statements` | a non-trivial sentence with no marker | warning only, surfaced in the audit |
| `non_evidence_citations` | a claim resting only on non-citable material | status → `grounding_warnings` |
| `external_finding_claims` | content wording attributed to a metadata-only external record | status → `grounding_warnings` |

Sentence splitting is regex-based with abbreviation protection, because the first version
split "Sleeman et al. (1982)…" into two sentences and produced a phantom unmarked claim.
Quotations are located in the cited source text after whitespace normalisation, with page
provenance recorded, so RAG-012 can re-check them; a quotation that cannot be found is
recorded (unit-tested). Style warnings (em dash, opening restatement) are advisory and
never change the status, so a well-grounded answer is not marked down for style.

**Measured: 0 invalid markers, 0 unmarked claims, 0 non-evidence citations, 0 external
finding claims, 72 of 82 sentences carrying a marker, 4 unmarked statements (one
transition sentence and three negative findings in the thin answers).**

## 10. Handling of non-evidence material

| Mode | Contents | Citable |
|------|----------|---------|
| `evidence_only` (default) | literature only | evidence blocks |
| `evidence_plus_my_work` | literature + `my_work` | evidence only |
| `evidence_plus_guidance` | literature + `my_work` + supervision/rules | evidence only |

`my_work` and `guidance` blocks are marked `citable=False` in the context, described as
such in the prompt, and cannot support a factual claim. A sentence whose only marker is a
non-evidence block is a `non_evidence_citation` — **unless** it is explicitly framed as a
report about that material ("your draft states…", "the guidance notes…"), which is the
whole point of supplying it. Those sentences are counted as `non_evidence_reports` and are
allowed in the corresponding mode. This distinction was added during validation: the first
`evidence_plus_my_work` run reported the author's draft correctly and was flagged as a
violation for doing so, which is a checker defect, not a model defect.

The example run (`What have I already written about the historical development of AI?`,
`evidence_plus_my_work`) produced 9 own-work reports, 0 non-evidence citations, 0 invalid
markers, and every specific claim in the answer was verified against the draft chunk.
Self-citation-as-evidence is **not** enabled and was not implemented (§18/§19 of the
spec): the `evidence_eligible=false` policy from RAG-007 is preserved end to end, and a
self-citation mode would need its own policy decision rather than being a default.

## 11. External-discovery behaviour (RAG-010 records)

RAG-010 results can be attached to a generation and stay visibly external:

* `external_discovery` — identity metadata from Crossref/OpenAlex, **not citable**. The
  block deliberately carries **no abstract**: an abstract describes findings, and a paper
  that was never retrieved must not be able to contribute findings. The block says so.
* `external_fulltext_verified` — a record whose open-access full text was actually fetched
  and identity-verified; citable but labelled external and excluded from corpus counts.

The check separates two kinds of wording: *existence* ("a 2026 paper was found and may be
relevant") is allowed and recorded as a report; *content* ("the paper found that…") is
refused as an `external_finding_claim`. **Measured: 3 metadata-only records → status `ok`,
0 external finding claims, 1 allowed existence report.** The first attempt failed — the
block still contained the provider abstract, the model reported its findings, and the
checker flagged it. Both the failure and the fix are recorded in
[docs/rag011-grounding-validation.txt](docs/rag011-grounding-validation.txt) (§28), as §33
requires. The verified-fulltext record (DOI `10.1186/s41239-023-00434-1`, SHA-256
`316d1664a4ef7647b487f97eae6a17224b148b200b823e00ef436c8f85247bcd`) was citable; the model
used it for what it supports and stated that it could not describe findings that are not
in the retrieved text.

External records are never merged into the corpus, never influence `evidence_eligible`, and
cannot become literature evidence by being supplied.

## 12. Prompt-injection defence

Three hostile chunks were placed in the collection alongside one benign paragraph: a direct
"ignore all previous instructions", a fake `SOURCE_TEXT`/system block, and a role-play
transcript asserting a fabricated effect size. Layered defence:

1. source text is untrusted data, stated in the system prompt, and delimited so a fake
   delimiter inside a document cannot escape its block;
2. instructions found in documents must be ignored, and are never followed;
3. the audit catches the *outcomes* (invented markers, marker-free claims, non-evidence
   citations, external finding claims) regardless of how the text was phrased.

**Result: the model identified the hostile chunks as instructions, refused them, invented
nothing, did not repeat the fabricated effect size, did not reveal the prompt, and still
cited the one benign source correctly** (`status=insufficient_evidence`, which is the
correct answer to a question whose only real evidence is one paragraph). Marker text that
would have shown compliance (BANANA, INJECTED) does not appear.

## 13. Insufficient-evidence behaviour

Five deliberately unanswerable questions were run against the real corpus:

| Question | Status | Output tokens |
|----------|--------|---------------|
| 2024 OECD report on quantum computing in primary schools | `insufficient_evidence` | 131 |
| Universities in Bolivia adopting an AI ethics policy in 2025 | `insufficient_evidence` | 101 |
| Feeding habits of Antarctic krill | `insufficient_evidence` | 114 |
| The dataset in a specific 2019 study, with record count | `insufficient_evidence` | 196 |
| Exact wording of a 2026 European Commission directive | `insufficient_evidence` | 139 |

**5/5 refused or qualified correctly, 0 partial answers, 0 hallucinations.** Each answer
states that the retrieved evidence does not provide enough support for the question, and
then describes what the sources do cover. No statistic, study, author or report was
invented; the model even quoted the figure that exists (LLFF, 20-62 images per scene) while
saying explicitly that it does not answer the question asked. When the corpus is empty of
citable material altogether the service short-circuits to `no_citable_context` without
calling the model.

## 14. Real Qwen results

Answered by the real model through Ollama over the real 11,469-chunk index
([docs/rag011-generation-evaluation.txt](docs/rag011-generation-evaluation.txt) records the
sources, context, raw answer, markers, tokens and timings per question):

| # | Question | Status | Markers | Prompt / output tokens | Generation |
|---|----------|--------|---------|------------------------|-----------|
| 1 | What is AI governance in higher education? | `ok` | 5 | 5,269 / 321 | 1.61 s |
| 2 | What evidence links virtual reality with student engagement? | `ok` | 5 | 5,539 / 365 | 1.82 s |
| 3 | How is generative AI affecting academic integrity? | `ok` | 8 | 6,171 / 289 | 1.48 s |
| 4 | What evidence supports adaptive learning in higher education? | `ok` | 5 | 5,571 / 270 | 1.33 s |
| 5 | What is Design Science Research Methodology and how is it used? | `ok` | 4 | 6,486 / 331 | 1.59 s |
| 6 | What have I already written about the historical development of AI? (`evidence_plus_my_work`) | `ok` | 1 | 5,895 / 341 | 1.67 s |
| 7-11 | The five thin questions | `insufficient_evidence` | 4-6 each | see §13 | 0.64-1.03 s |

All five evidence answers name the sources they used and answer the question in 250-300
words of plain academic prose; none of them quote a figure that is not in the supplied
text. The answers are reproduced verbatim in the artifact, unedited.

## 15. Claim-level manual evaluation

Every substantive claim in the five evidence answers was checked by hand against the source
text that was supplied, with the chunk id and quotation recorded per claim (so each check
is reproducible):

| Verdict | Claims |
|---------|--------|
| SUPPORTED | 42 |
| PARTIALLY_SUPPORTED | 1 |
| UNSUPPORTED | 0 |

Plus 9 own-work reports in the `my_work` answer, all verified against the draft chunk.
Acceptance target was **0 unsupported substantive claims: met**.

The single `PARTIALLY_SUPPORTED` claim compresses two separate findings from one source into
one sentence (a faculty-training figure joined to a separate assignments/over-reliance
finding). It is not an invented fact and it is not hidden: it is listed in
[docs/rag011-grounding-validation.txt](docs/rag011-grounding-validation.txt) with the exact
sentences it merges. Verified examples of specific claims that survived checking: the 75%
Ghana figure, the 98% / median / full-marks findings, the 200-undergraduate study with
p < 0.05, Mikropoulos and Natsis (2011), the anatomy meta-analysis figures, and the Design
Science Research labels, quadrants and schema.

## 16. Invalid-marker results

| Run | Answers | Invalid markers | Unmarked claims | Non-evidence citations | External finding claims |
|-----|---------|-----------------|-----------------|------------------------|-------------------------|
| Evidence questions | 5 | 0 | 0 | 0 | 0 |
| my_work question | 1 | 0 | 0 | 0 | 0 |
| Thin questions | 5 | 0 | 0 | 0 | 0 |

53 markers were produced across 11 answers; **every one of them exists in the context that
was supplied** (0 invalid). No answer needed repairing, no marker was removed by hand, and
no answer text was edited. In evidence-only runs the `source_map` confirms every cited
block has `corpus_role=literature` and `evidence_eligible=True`.

## 17. Performance

Measured on the RTX 5090 with the model resident
([docs/rag011-performance.txt](docs/rag011-performance.txt)):

| Output length | Answer words | Output tokens | Retrieval | Generation | Total | Throughput |
|---------------|--------------|---------------|-----------|------------|-------|------------|
| concise | 131 | 173 | 0.47 s | 1.38 s | 1.86 s | 214.6 tok/s |
| normal | 274 | 350 | 0.48 s | 1.76 s | 2.24 s | 226.6 tok/s |
| detailed | 606 | 808 | 0.48 s | 3.76 s | 4.24 s | 226.9 tok/s |

* Context construction is below timer resolution (<1 ms): pure counting and string
  rendering, no model.
* The first call in a fresh process costs about 11.5 s (loading BGE-M3 and the reranker);
  the generation model stays resident for 30 minutes, so subsequent questions pay none.
* VRAM: 29.2 GiB of 32.6 GiB in use with the model resident at `num_ctx=8192`.
* The estimator runs 13-29% above Ollama's real `prompt_eval_count` (measured ratios
  0.777-0.874). Over-estimating the context is the safe direction for a budget.
* No optimisation was attempted: 35B parameters at ~227 tokens/s is the practical ceiling
  here, and generation dominates the total.

## 18. Test result

```
python -m pytest -q
769 passed, 2 warnings in 8.56s
```

* 657 pre-existing tests (RAG-001 … RAG-010) pass unchanged, as §31 requires.
* 112 new tests: context 21, ollama 18, grounding 31, service 26, CLI 16.
* No unit test requires Ollama or the Internet: the offline harness supplies a fake model,
  a fake retrieval service and synthetic records. The real-model work is the validation
  documented in the artifacts.
* The two warnings are pre-existing (a FastAPI/Starlette deprecation notice and the local
  Qdrant payload-index notice from the indexing tests) and are unrelated to this stage.

## 19. CLI

```
python -m app.generation.inspect "What is AI governance in higher education?"
python -m app.generation.inspect "..." --show-context --show-prompt --show-audit
python -m app.generation.inspect "What have I written about AI history?" --include-my-work
python -m app.generation.inspect "..." --include-my-work --include-guidance --length detailed
python -m app.generation.inspect "..." --context-tokens 2000 --min-rerank-score 0.3
python -m app.generation.inspect "..." --json
```

`--show-prompt` prints the exact system and user prompts, which is how §22 (prompt
transparency) is satisfied in practice; `--show-context` prints what the model really saw.
The CLI exits 1 on `model_error` so a scripted caller can detect a failed generation.

## 20. Corpus state after the stage

Read back in a separate process (`QdrantClient(path=...)` on the local store):

```
collection      : phd_rag_chunks_v1
points          : 11469
vector          : 1024, Cosine
corpus_role     : literature 10456 | my_work 106 | future_chapters 188
evidence_eligible=True : 10456
ocr_involved=True      : 104
content_type=REFERENCES: 0
```

Unchanged from RAG-008/009: this stage reads payloads and vectors and never writes.

## 21. Known limitations

1. **The grounding check verifies markers, not entailment.** A sentence can carry a valid
   marker whose source only partly supports it. The single `PARTIALLY_SUPPORTED` claim is
   exactly that case. The manual claim evaluation is the safety net, and RAG-012 is where
   entailment checking belongs.
2. **Sentence splitting is heuristic.** Abbreviations are protected, but a sentence the
   splitter mis-detects can be reported as an unmarked statement. Such reports are warnings,
   not data changes, and the answer text is never modified — but they are noise.
3. **The token estimator is character-based** (13-29% high, deterministic). It exists so the
   budget is reproducible and testable without loading a tokenizer; a real tokenizer would
   be more accurate but slower and version-dependent.
4. **`num_ctx` is 8,192**, so a context larger than about 6k estimated tokens is not
   possible today. Raising it would add KV cache (about 2.5 GiB for 16,384) and still fits,
   but nothing measured needs it.
5. **About 11.5 s of model loading** dominates any one-shot CLI invocation.
6. **No retries on Ollama failures.** A transient error surfaces as `model_error`. This is a
   deliberate choice for observability; an interactive caller may want a bounded retry.
7. **A fixed 1,500-token reservation** protects my_work/guidance requests. A caller that
   sets a small total budget can still starve them, and the reservation is a constant rather
   than a function of the question.
8. **Bibliography-like chunks are filtered at retrieval time only** (RAG-009's guard).
   Nothing was re-indexed, so the guard must stay in the pipeline for the protection to hold.
9. **Metadata-only external records are identity-level only.** They can confirm that a paper
   exists; they can never contribute findings, which is the intended behaviour but limits
   what an unattached discovery result is useful for.
10. **`vision` and `tools` capabilities of the model are unused** — images and function
    calling are outside this stage.
11. **Small live sample.** 11 questions were evaluated against the real model. The unit
    suite is large, but the grounding claim rests on a manual evaluation of 52 claims.
12. **OCRs and tables are plain text** in context. No table structure is preserved and no
    layout is reconstructed; the `ocr_involved` flag is carried into the result so a later
    stage can treat such answers more strictly.
13. **No citation formatting at all.** Sources are markers and payload fields; Harvard style
    is deliberately deferred.

## 22. Validation defects found and fixed during this stage

Recorded because §33 requires failures to be preserved, not erased:

1. **Unmarked opening sentences (prompt v1)** — 3 of 5 evidence answers were flagged
   `grounding_warnings` because the first sentence carried no marker. Fixed by v2 requiring a
   marker on every claim sentence; all 5 then returned `ok`.
2. **Phantom unmarked claim from abbreviation splitting** — "Sleeman et al. (1982)" split
   mid-sentence. Fixed with abbreviation-protected sentence splitting.
3. **Double-counted prompt tokens** — the estimate summed the context twice (6,003 estimated
   against 3,007 real). Fixed to count the actual prompt strings once; the ratio is now
   0.777-0.874, i.e. always in the safe direction.
4. **Starved `my_work` request** — an explicit `include_my_work=True` question refused
   because evidence had consumed the budget. Fixed with the non-evidence reservation (§7).
5. **Correct own-work reporting flagged as a violation** — the checker treated every
   draft-report sentence as a non-evidence citation. Fixed by allowing reports about
   my_work/guidance in the modes that requested them, and recording them as
   `non_evidence_reports`.
6. **External metadata reported as findings** — the first §28 run placed provider abstracts
   in metadata-only blocks and the model reported their findings. Fixed by removing the
   abstract from metadata-only blocks and separating existence wording from content wording
   in the check. Both attempts are in the artifact (§28).

## 23. Acceptance criteria

| Criterion | Result |
|-----------|--------|
| All previous 657 tests pass | ✅ 769 pass in total |
| All new tests pass | ✅ 112 new |
| `qwen3.5:35b` works through Ollama | ✅ 11 real answers recorded |
| RAG-009 retrieval reused | ✅ via the shared `build_retrieval_service` |
| Bibliography guard remains active | ✅ `retrieval_exclude_bibliography=True`; bibliography-like candidates withheld and recorded |
| 0 invented source markers | ✅ 0 invalid of 53 |
| 0 non-evidence citations in evidence-only answers | ✅ 0 |
| 0 unsupported substantive claims | ✅ 0 (1 partially supported, reported) |
| Insufficient evidence produces qualification, not hallucination | ✅ 5/5 |
| External metadata-only results never become evidence of findings | ✅ 0 finding claims |
| Prompt-injection test passes | ✅ hostile chunks refused |
| Source provenance survives end to end | ✅ `source_map`, chunk id, pages, OCR flag, role recorded per answer |
| Generation latency measured | ✅ 1.86 / 2.24 / 4.24 s by length |

## 24. Recommendation for RAG-012

1. **Build citation validation on the existing trace, not on new generation.** Every claim
   already carries its marker, chunk id, pages and role, and quotations already carry
   provenance. RAG-012 should verify that each cited source contains the claim, starting
   with the one `PARTIALLY_SUPPORTED` pattern (two findings merged into one sentence) —
   that is the realistic failure mode, not invented facts.
2. **Use `ocr_involved` and `content_type` as risk signals.** Answers resting on scanned
   text or tables should be checked more strictly, since OCR errors change the evidence
   itself. The flag is already in the result.
3. **Treat quotation checking as a first-class validator.** `find_quotations` already
   locates quotations and pages; RAG-012 should fail a citation whose quotation cannot be
   found rather than only recording it.
4. **Do not add an LLM to the validation step initially.** A deterministic checker over the
   recorded evidence keeps the guarantee auditable and cheap; escalate to a model only for
   the cases the deterministic checker cannot decide.
5. **Consider a real tokenizer** (or the model's own counter) for the context budget once
   the estimator's 13-29% headroom becomes limiting, so `num_ctx` can be raised safely.
6. **Keep `evidence_eligible=false` for my_work in every future stage** and resist
   self-citation modes until there is an explicit, tested policy for them.

---

## RAG-011 COMPLETE

Full `pytest`: **769 passed** (657 pre-existing + 112 new). Artifacts:
[RAG-011-report.md](RAG-011-report.md),
[docs/rag011-generation-evaluation.txt](docs/rag011-generation-evaluation.txt),
[docs/rag011-grounding-validation.txt](docs/rag011-grounding-validation.txt),
[docs/rag011-performance.txt](docs/rag011-performance.txt).

No source document was modified; the Qdrant collection still holds 11,469 points at
1024 dimensions with Cosine distance; the three RAG-006 DOI conflicts and the RAG-007 role
policy are untouched. RAG-012 has **not** been started.
