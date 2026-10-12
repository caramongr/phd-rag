# RAG-013 — Academic working modes

Five user-facing modes (**EVIDENCE**, **WRITING**, **CONSISTENCY**, **SUPERVISOR**,
**RESEARCH**) added as orchestration presets over the completed pipeline. No retrieval,
generation, citation-validation or discovery logic was designed, changed or duplicated in
this stage: every mode calls RAG-009 retrieval, RAG-011 generation, RAG-012 validation and
RAG-010 discovery as they are, and adds only (a) a frozen configuration object per mode,
(b) a thin orchestration service, and (c) one new deterministic comparison engine that
CONSISTENCY needs and that nothing previously provided.

A mode is a *promise about what material may do*. That promise is data
(`ModeConfig.context_policy`), enforced at two points: RAG-009's own vector-search filters
decide which corpus roles can even be returned for a retrieval mode, and the mode records
for every retrieved block the role it was allowed to play. A block is evidence only when
its role is `EVIDENCE` **and** the corpus itself marked it `evidence_eligible`.

## 1. Files created

| File | Lines | Role |
|---|---|---|
| `app/modes/__init__.py` | 91 | Public surface: the presets, the result models, `ModeService`, the comparison engine |
| `app/modes/models.py` | 452 | `ModeName`, `OutputStyle`, `SourceRole`, `ModeStatus`, `ConsistencyFindingKind`, `ModeConfig`, the five preset instances, `ModeSource`, `ExternalSourceView`, `ConsistencyFinding`, `ModeSection`, `ModeResult`, the writing-preference and promotional-term lists |
| `app/modes/service.py` | 906 | `ModeService.run_mode` and the four internal paths (generation / consistency / supervisor / research); `POLICY_NOTE`; `_retrieval_mode_for_role`; the shared audit trail |
| `app/modes/consistency.py` | 554 | The deterministic comparison engine: `ComparisonPolicy`, five checks (`terminology`, `duplication`, `unsupported_claim`, `numeric_discrepancy`, `style`), `_capped`, `compare`, `finding_counts`, `render_report` |
| `app/modes/inspect.py` | 369 | CLI (`python -m app.modes.inspect <mode> "<query>"`), renderers, exit codes |
| `tests/modes_support.py` | 265 | `FakeModeRetrieval`, `FakeDiscoveryService`, `build_harness`, real RAG-009/RAG-010/RAG-011/RAG-012 model builders, text fixtures |
| `tests/test_modes_config.py` | 132 | 12 tests — the presets and their determinism |
| `tests/test_modes_service.py` | 417 | 36 tests — the ten focused cases of §13 plus end-to-end runs of all five modes |
| `tests/test_modes_consistency.py` | 378 | 32 tests — the comparison engine |
| `tests/test_modes_cli.py` | 163 | 15 tests — CLI rendering, flags, exit codes |
| `docs/rag013-mode-validation.txt` | 631 | Real validation artifact: all five modes run end-to-end (§12) |
| `RAG-013-report.md` | this file | |

## 2. Files modified

| File | Change |
|---|---|
| `app/discovery/render.py` | Added the public `group_records()` wrapper so RESEARCH can group external records exactly as RAG-010's own CLI does (already-have / verified-new / discovery-only / conflicts) without copying the grouping rules |
| `app/citations/claims.py` | Extracted `content_tokens()` (the `(canonical term, surface token)` pairs) and made the existing `content_terms()` delegate to it, so CONSISTENCY can show the word the author wrote while still comparing canonical forms — the tokenising rules exist once, in RAG-012's module |
| `README.md` | RAG-013 CLI examples and the modes prose paragraph |
| `RAG-013-report.md` | New |

Nothing else was touched. In particular `app/retrieval/`, `app/generation/` and
`app/discovery/` behaviour is unchanged (`content_terms()` returns exactly what it returned
before, and the two RAG-012 wrappers added earlier are still the only additions to
`app/generation/`), and RAG-013 consumes the earlier stages through their existing public
entry points.

## 3. Architecture

```
python -m app.modes.inspect <mode> "<query>"
        |
        v
ModeService.run_mode(mode, query, ...)            app/modes/service.py
        |
        |  config = mode_config(mode)             app/modes/models.py  (frozen preset)
        |
        +-- EVIDENCE / WRITING ------------------------------------------------
        |     GenerationService.answer(retrieval_mode, generation_mode, ...)   RAG-011
        |         -> RetrievalService.search(...)                              RAG-009
        |     CitationService.validate(response)                               RAG-012
        |         -> validated_answer, references, citation_map, claim audit
        |
        +-- CONSISTENCY -------------------------------------------------------
        |     RetrievalService.search(MY_WORK) + search(EVIDENCE when asked)
        |         + search(GUIDANCE for the rules)
        |     app.modes.consistency.compare(...)      deterministic, no model call
        |
        +-- SUPERVISOR --------------------------------------------------------
        |     RetrievalService.search(SUPERVISION) + search(GUIDANCE)
        |     labelled excerpts                        no model call at all
        |
        +-- RESEARCH ----------------------------------------------------------
              research_discover(query, ...)                                RAG-010
                  -> local chunks (RAG-009) and external records, kept apart
        |
        v
ModeResult  (frozen, one shape for every mode)
```

Two properties follow from this shape and are asserted by tests:

* **The model is asked only where a model may answer.** `run_generation` is `True` for
  EVIDENCE and WRITING and `False` for the other three. CONSISTENCY, SUPERVISOR and
  RESEARCH therefore cannot produce a model-authored sentence at all: the recording
  doubles assert `harness.model.calls == []` in those modes
  (`test_consistency_mode_compares_the_draft_with_the_literature`,
  `test_research_mode_never_generates_or_cites`), and CONSISTENCY records
  `audits["no_model_used"] is True`. This is the same reasoning RAG-012 used for
  validation: a model asked to "check my draft for contradictions" invents findings about
  the author's text, so CONSISTENCY measures instead.
* **The pipeline is not re-implemented.** A test replaces the retrieval, generation,
  citation and discovery services with recording doubles and asserts what `run_mode`
  called: one generation call for EVIDENCE/WRITING, a citation validation for both, and
  no generation call anywhere else.

## 4. Mode configuration (§7)

One frozen `ModeConfig` per mode (`MODES`), looked up once per run by
`mode_config(mode)` — no per-mode branching outside that dict and the four design choices
it drives.

| | EVIDENCE | WRITING | CONSISTENCY | SUPERVISOR | RESEARCH |
|---|---|---|---|---|---|
| retrieval mode | `evidence` | `evidence` | `my_work` | `supervision` | `evidence` |
| generation mode | `evidence_only` | `evidence_plus_guidance` | `evidence_plus_guidance` | `evidence_plus_guidance` | `evidence_only` |
| `include_my_work` | – | yes (context) | yes (subject) | – | – |
| `include_guidance` | – | yes (instructions) | yes (instructions) | yes | – |
| `include_supervision` | – | – | – | yes | – |
| `use_external_discovery` | – | – | – | – | yes |
| `run_generation` | yes | yes | no | no | no |
| `run_citation_validation` | yes | yes | no | no | no |
| `compare_with_literature` | – | – | yes | – | – |
| output style | `validated_answer` | `validated_answer` | `structured_comparison` | `labelled_excerpts` | `separated_results` |

`context_policy` is the safety contract in data form — corpus role → the role it may play:

| corpus role | EVIDENCE | WRITING | CONSISTENCY | SUPERVISOR | RESEARCH |
|---|---|---|---|---|---|
| `literature` | evidence | evidence | evidence | *undeclared* | evidence |
| `my_work` | *undeclared* | context only | **subject** | context only | *undeclared* |
| `rules`, `guidance`, `summaries`, `future_chapters` | *undeclared* | instructions only | instructions only | guidance only | *undeclared* |
| `supervision` | *undeclared* | instructions only | instructions only | guidance only | *undeclared* |

An undeclared role resolves to `INSTRUCTIONS_ONLY` (`ModeConfig.role_for`), i.e. the
default is the most restrictive role, never `EVIDENCE`. `ModeSource.is_evidence` then
requires both `role is EVIDENCE` and `evidence_eligible`, so a role mapping alone can never
promote a block: only RAG-009's corpus-role resolution, which is the conservative one
(several physical copies of a document take the most restricted role), can mark material
eligible.

## 5. EVIDENCE mode (§2)

Retrieval `evidence` → generation `evidence_only` → RAG-012 validation and Harvard
citations. `my_work`, `summaries`, supervision notes and writing rules are not retrievable
at all in this mode (RAG-009 filters them at the vector-search level), so the "do not use
them as scholarly evidence" rule is enforced before generation rather than after.

Real run (`docs/rag013-mode-validation.txt`, §13 below): 8 local sources, all evidence, 12
claims, **12 supported**, 0 partial, 0 unsupported, 0 withheld, 12 citations written, 3
references, no `[Sx]` marker left in the answer, 13.3 s.

## 6. WRITING mode (§3)

Retrieval is `evidence` (literature is the evidence); the author's own draft and the
project rules are added by RAG-011 as explicitly non-citable context
(`include_my_work`/`include_guidance`), and RAG-012 then refuses to cite either of them.
The seven writing preferences of §3 are a named tuple (`WRITING_PREFERENCES`) and are what
the CONSISTENCY style checks are written against, so the same list drives generation
guidance and the deterministic check rather than two drifting copies.

Real run: 8 local sources (6 evidence, **2 non-citable**: one `my_work` block and one
`future_chapters` block), 9 claims, all supported, 2 references, and the recorded warning
*"your draft and the rules entered the context as non-citable material; they are never
cited"*. No non-citable source appears in the citation map. 2.5 s.

## 7. CONSISTENCY mode (§4)

The only mode with new analysis, and the only mode where the model is deliberately not
consulted. `app/modes/consistency.py` produces five kinds of finding, each carrying the
method that produced it and, where one exists, the chunk it rests on:

| Kind | Method | What has to be true |
|---|---|---|
| `terminology` | `content_term_frequency` | a content term appears in ≥2 of your chunks and in none of the retrieved literature (and the reverse). The report shows the **surface word** (`governance`), while the comparison uses the canonical form, so `governance`/`governances` group together without a stem ever reaching the reader |
| `duplication` | `content_term_containment` | the smaller passage of a pair (≥8 content terms) is ≥60 % contained in the larger — draft-vs-draft or draft-vs-literature (a close paraphrase to check) |
| `unsupported_claim` | `content_term_containment` | a draft sentence of ≥6 content terms whose best overlap with any retrieved literature sentence is <30 %. Every finding states *"no match was found in what was retrieved, not that the claim is wrong"* and counts the chunks and sentences it looked at (6 chunks / 75 sentences in the real run) |
| `numeric_discrepancy` | `numeric_discrepancy` | a figure in your draft where the best-matching literature passage (≥35 % overlap) states figures of the same kind but not that value |
| `style` | `character_check`, `term_list_check`, `sentence_length_check` | em dash, promotional terms, sentences >35 words — checked against the retrieved rules chunk when one was retrieved, otherwise against `WRITING_PREFERENCES` and recorded as `rag013:writing_preferences` |

Every check is a measurement with a threshold in one place (`ComparisonPolicy`), reported
with its inputs, so a finding can be argued with. Findings are capped per kind
(`max_findings_per_kind = 12`) and the list admits it when the cap bites — the last finding
of that kind records *"showing 12 of 85 findings of this kind"* — because a truncated list
that stays silent reads as a complete one. The draft is never cited and never rewritten:
CONSISTENCY compares and stops, exactly as §4 requires ("do not automatically rewrite
unless explicitly requested").

Real run: 15 local sources (6 literature, 9 of the author's own work and rules), 6
literature chunks + 75 sentences used as the comparison base, 19 findings across three
kinds (`{style: 5, terminology: 2, unsupported_claim: 12}`, the unsupported-claim list at
its cap and saying so: *"showing 12 of 85 findings of this kind"*), 0 citations, 1.1 s, no
model call.

## 8. SUPERVISOR mode (§5)

Retrieval `supervision` + `guidance`, output `labelled_excerpts`, no model call, no
citations, `evidence_sources == ()`. Every excerpt is printed with its corpus role
(`[supervision]`, `[future_chapters]`, `[guidance]`, `[rules]`) and with its document,
section and pages, and the result always carries the warning that supervision and guidance
material is the author's own scaffolding, is never cited, and cannot support a claim about
the literature. Because the retrieval mode cannot return `literature` blocks, no supervisor
note can become a citation: there is nothing citable in the result to cite.

Real run: 12 labelled excerpts (6 progress-review forms, 1 guidance policy, 3 future
chapter sections, 2 rules), 0 evidence sources, 0 citations, 0.6 s.

## 9. RESEARCH mode (§6)

`research_discover` (RAG-010) over RAG-009's local retrieval, with the results rendered in
four separate sections — `LOCAL EVIDENCE`, `NEW EXTERNAL PAPERS`, `ALREADY IN CORPUS`,
`POSSIBLE MATCHES` — using RAG-010's own grouping (`group_records`). `ModeSection` records
which local and external indices belong to which section, so the separation is part of the
structured result rather than only of the text. No generation, no citations, and every
external record is typed `ExternalSourceView` with `usable_as_evidence=False` written
literally on the model, so a consumer of the JSON cannot mistake it for evidence.

Real run: 6 local evidence chunks + **20 external records** from Crossref and OpenAlex,
all classified `POSSIBLE MATCHES` (RAG-010's own criterion: identity verified only when the
providers agree and a DOI or full text confirms it, none of which happened for these
records), 0 generated prose, 0 citations, 3.0 s.

## 10. Common result model (§9)

`ModeResult` is frozen and identical for every mode; fields a mode does not use stay empty.
Beyond the fields listed in §9 it carries `config` (the preset that ran), `external_sources`
and `sections` (RESEARCH separation), `findings` (CONSISTENCY), `claim_validations` (the
RAG-012 audit, so `--show-audit` can print the claim-by-claim verdicts), `generation_status`
and `seconds`.

Convenience properties encode the invariants instead of leaving them to callers:
`evidence_sources` / `context_only_sources` (a source is evidence only if role *and*
eligibility agree), `unsupported_claims` (the withheld RAG-012 claims), `has_citations`
(true only when a reference list exists), `new_external_papers`. `to_json()` is what the
CLI's `--json` prints.

Statuses: `ok`, `insufficient_evidence`, `no_citable_context`, `model_error`, `no_results`.
A RAG-011 `grounding_warnings` answer maps to `ok` with its warnings preserved, and a model
failure maps to `model_error` — it is reported, never converted into a plausible answer or
an exception.

## 11. Safety rules (§10)

| Rule | Where it is enforced | How it is verified |
|---|---|---|
| only `evidence_eligible` literature may support scholarly claims | RAG-009 vector-search filters + `ModeSource.is_evidence` + RAG-012's `non_evidence_citations` check | `test_evidence_mode_excludes_non_evidence`, `test_writing_mode_never_cites_the_draft_or_the_rules` |
| `my_work` stays non-evidence | retrieval mode `my_work` is not used by any generating mode; in WRITING it is context only | `test_writing_mode_includes_my_work_as_context_only`, `test_consistency_mode_never_treats_the_draft_as_evidence` |
| supervision stays non-evidence | retrieval mode `supervision`; SUPERVISOR never cites | `test_supervisor_mode_never_produces_scholarly_citations_from_supervision` |
| rules/guidance stay non-evidence | `INSTRUCTIONS_ONLY` / `GUIDANCE_ONLY` roles; never in the citation map | `test_supervisor_mode_audit_says_nothing_was_cited` |
| metadata-only Internet records stay non-evidence | `ExternalSourceView.usable_as_evidence = False`; no external record ever enters a generation context | `test_research_mode_keeps_local_and_external_separate`, `test_research_mode_never_generates_or_cites` |
| bibliography guard stays active | inherited: RAG-009 excludes bibliography-like chunks in `evidence`, `my_work` and `general` retrieval, and the modes retrieve only through RAG-009 | `test_evidence_mode_reports_the_bibliography_guard_still_applies` |
| unsupported claims must not be silently published | inherited from RAG-012: withheld from `validated_answer`, listed in `warnings` and `unsupported_claims`, surfaced as CLI exit code 3 | `test_unsupported_claims_remain_withheld`, `test_writing_mode_also_withholds_unsupported_claims` |

`POLICY_NOTE` is written into the audit trail of every run, so a saved or piped result
carries the policy that produced it rather than depending on documentation elsewhere.

## 12. CLI (§11)

```powershell
.\.venv\Scripts\python.exe -m app.modes.inspect evidence "What is AI governance in higher education?"
.\.venv\Scripts\python.exe -m app.modes.inspect writing "Write a short PhD paragraph on AI governance in higher education."
.\.venv\Scripts\python.exe -m app.modes.inspect consistency "Compare my existing treatment of AI governance with the literature."
.\.venv\Scripts\python.exe -m app.modes.inspect supervisor "What guidance do I have about methodology?"
.\.venv\Scripts\python.exe -m app.modes.inspect research "Find recent papers on AI governance in universities."

# sources and roles, the full audit trail (config, policy, claim verdicts), JSON
... --show-sources --show-audit --json
```

Retrieval, generation and discovery knobs are the ones the earlier stages already define
(`--candidate-k`, `--final-k`, `--local-k`, `--no-rerank`, `--min-rerank-score`,
`--length`, `--temperature`, `--max-output-tokens`, `--context-tokens`, `--model`, and for
RESEARCH `--external-k`, `--provider`, `--refresh`, `--year-min/--year-max`). `--external`
forces discovery on or off for a mode that would otherwise not run it. Exit codes: `0`
success, `1` model error, `3` at least one claim unsupported (withheld) — so a shell can
tell "answered" from "answered with a disclaimer" from "failed" without parsing prose.

## 13. Real validation (§12)

`docs/rag013-mode-validation.txt` (631 lines, 60 KB) was produced by running the five
required queries through the real pipeline in one process, against the real corpus, the
real local model (`qwen3.5:35b` through Ollama) and the real providers (Crossref,
OpenAlex). Each mode records its result summary, its safety block (non-citable sources
cited, external records used as evidence, local blocks outside the declared roles, withheld
claims, citations carrying a chunk id, leftover `[Sx]` markers), its full output, its
sources with their roles, and its warnings.

| §12 requirement | Result |
|---|---|
| EVIDENCE produces a validated Harvard-cited answer | **met** — 12/12 claims supported, `(Chan, 2024)` / `(Singh, 2024)` / `(PRESS, 2023)` in the text, 3 deduplicated references, 0 markers left, 5/5 citations carry a chunk id |
| WRITING uses literature as evidence, my_work/rules only as context | **met** — 6 evidence sources cited, 2 non-citable sources present and never cited, warning recorded, 9/9 claims supported |
| CONSISTENCY identifies differences without treating your work as evidence | **met** — 19 findings across 3 kinds (0 citations, 0 evidence sources), including the documented style deviations and the draft sentences the retrieved set does not cover (12 listed of 85, the cap stated); the report states the limit of that measurement |
| SUPERVISOR clearly labels supervision/guidance material | **met** — 12 excerpts, each labelled `[supervision]` / `[future_chapters]` / `[guidance]` / `[rules]`, 0 citations, non-evidence warning present |
| RESEARCH shows local and external results separately | **met** — 6 local evidence chunks and 20 external records in four named sections; every external record `usable_as_evidence=False` |

Totals: 5/5 runs `ok`, 49 local blocks classified by role, 20 external records, 21 claims
validated, 0 withheld, 0 policy violations, 20.5 s wall clock for the whole artifact.

## 14. Performance

Per mode, end to end, on this machine (real retrieval, real rerank, real model, warm cache):

| Mode | Seconds | Notes |
|---|---|---|
| EVIDENCE | 13.3 | 8 sources, generation + validation of 12 claims |
| WRITING | 2.5 | 8 sources, generation + validation of 9 claims |
| CONSISTENCY | 1.1 | retrieval + 5 deterministic checks over 6 literature chunks / 75 sentences |
| SUPERVISOR | 0.6 | retrieval and labelling only |
| RESEARCH | 2.9 | local retrieval + 2 providers + cache |
| whole artifact | 20.5 | five runs in one process (models loaded once) |

The deterministic modes cost what retrieval costs and no more; the orchestration layer adds
single-digit milliseconds (a `mode_config` lookup, result construction). RAG-012's measured
validation cost (0.029–0.043 s per answer) is unchanged.

## 15. Test result (§13)

```
990 passed, 2 warnings in 9.20s
```

895 pre-existing tests plus **95 new** ones, all passing: 12 preset/config, 36 service
(including the ten focused cases below), 32 comparison engine, 15 CLI. The two warnings are
pre-existing (a Starlette/httpx deprecation and Qdrant's local-mode payload-index notice).

| §13 required case | Test |
|---|---|
| correct preset configuration per mode | `tests/test_modes_config.py` (12 tests: `test_preset_configuration_per_mode`, `test_every_mode_has_a_preset`, `test_only_generation_modes_validate_citations`, ...) |
| evidence mode excludes non-evidence | `test_modes_service.py::test_evidence_mode_excludes_non_evidence` |
| writing mode includes my_work as context only | `test_modes_service.py::test_writing_mode_includes_my_work_as_context_only` (+ `test_writing_mode_never_cites_the_draft_or_the_rules`) |
| supervisor mode never produces scholarly citations from supervision | `test_modes_service.py::test_supervisor_mode_never_produces_scholarly_citations_from_supervision` |
| research mode keeps local/external separate | `test_modes_service.py::test_research_mode_keeps_local_and_external_separate` |
| bibliography guard preserved | `test_modes_service.py::test_evidence_mode_reports_the_bibliography_guard_still_applies` |
| citation validation invoked where required | `test_modes_service.py::test_modes_reuse_the_pipeline_instead_of_reimplementing_it`, `test_evidence_mode_produces_a_validated_cited_answer` |
| unsupported claims remain withheld | `test_modes_service.py::test_unsupported_claims_remain_withheld`, `test_writing_mode_also_withholds_unsupported_claims` |
| common output shape | `test_modes_service.py::test_result_shape_is_common_to_every_mode` (+ `test_every_mode_runs_end_to_end` over all five modes) |
| deterministic mode configuration | `tests/test_modes_config.py::test_mode_configuration_is_deterministic`, `test_modes_service.py::test_mode_runs_are_deterministic_for_the_same_inputs` |

## 16. Acceptance criteria (§14)

| Criterion | Status |
|---|---|
| all previous 895 tests pass | **met** — 895 + 95 = 990 passed |
| all new tests pass | **met** — 95/95 |
| all five modes work end-to-end | **met** — `docs/rag013-mode-validation.txt`, 5/5 `ok` |
| no evidence-policy leakage | **met** — 0 non-citable citations, 0 external records flagged usable as evidence, 0 blocks outside the declared roles in all five runs |
| no duplicated retrieval/generation logic | **met** — modes call RAG-009/010/011/012; the only new analysis (CONSISTENCY) is a measurement the pipeline did not have, and `group_records` was exposed rather than copied |
| local/external separation preserved | **met** — four named sections, `ModeSection` in the structured result |
| validated citations still used in EVIDENCE/WRITING | **met** — both modes end in RAG-012 with a reference list and a claim audit |

## 17. Known limitations

1. **CONSISTENCY measures retrieval coverage, not truth.** "unsupported claim" means *no
   match was found in the retrieved literature*, which is why every such finding states
   that limit and records how much was searched (6 chunks, 75 sentences in the real run).
   With a narrow query the check is uninformative, and the report says so rather than
   implying a verdict.
2. **CONSISTENCY finds differences, not contradictions in the strict sense.** Two sources
   that disagree in substance while sharing vocabulary are not detected as a contradiction;
   what is detected is a figure that the best-matching passage does not state, a sentence
   nothing covers, and a passage that repeats another. The stricter reading ("this
   literature contradicts what you wrote") is left to the author.
3. **Caps and truncation.** `max_findings_per_kind = 12` and `terminology_top = 8` bound the
   output; longer draft chapters have more findings than are listed, and the real
   CONSISTENCY run hit the unsupported-claim cap (12 listed of 85). Each capped list reports
   how many it withheld, but the withheld findings themselves are not listed anywhere.
4. **Terminology noise.** Content-term comparison can surface a reference name that the
   citation filter did not catch (e.g. `zawackirichter` from a hyphenated author name at the
   start of a citation).     It is a candidate for review, not a claim, and the finding's `method` and detail
   line say so.
5. **SUPERVISOR relevance is bounded by the corpus.** The supervision corpus is largely
   progress-review forms and tables, so excerpt relevance is limited by what is stored
   there; the mode labels what it found and never upgrades it.
6. **RESEARCH shows what discovery classifies.** In the real run every external record
   landed in `POSSIBLE MATCHES` because identity was not verified for any of them; the
   `NEW EXTERNAL PAPERS` section is therefore empty, and empty sections are printed rather
   than hidden so the absence is visible.
7. **No cross-mode session yet.** Each `run_mode` call is stateless; there is no shared
   "current chapter" context between a CONSISTENCY run and a WRITING run. That would be
   RAG-014+ work and was explicitly out of scope.

## 18. Defects found and fixed during this stage

| Defect | Fix |
|---|---|
| Two of the five checks silently truncated their findings at the cap (`if len(findings) >= cap: return`), so a capped list was indistinguishable from a complete one | All five checks now go through `_capped()`, which appends *"showing N of M findings of this kind"* to the last finding of that kind when the cap bites (visible in the real run as *"showing 12 of 85"*) |
| Terminology findings printed the internal stems, so a reader saw `governanc`, `analysi`, `nuanc` instead of words | `content_tokens()` (new, in `app/citations/claims.py`) yields `(canonical term, surface token)`; counts still use the canonical form, reports show the most frequent surface word. `content_terms()` now delegates to it, so the comparison rules exist once |
| A term that occurred twice inside a single chunk counted as "repeated across the corpus" | Counts are per chunk again (a term counts once per chunk), so `terminology_min_count` keeps its meaning; found by the existing test that asserts a stray mention is not a finding |
| The em-dash style finding quoted the *beginning* of the chunk, so the dash it complained about was often not in the quoted text | `_window()` centres the excerpt on the character or term that fired the finding |
| Repeated "N sentence(s) longer than 35 words" findings from different chunks were indistinguishable | The summary now states the longest sentence's word count, and findings name both their own chunk (`your chunk`) and the literature chunk (`lit. chunk`) in both renderers |
| Excerpts ended mid-word (`... into their teaching practices (S.`) | `_excerpt()` cuts at a word boundary and marks the cut |
| `ModeConfig.external_as_context` was declared but never consumed (dead configuration) | Removed; the presets and their tests are unchanged |

Verification of the first four was done by re-running the real pipeline and reading the
regenerated artifact, not only by unit tests: the terminology list now reads
`tutoring, insight, first, second, concrete, social, ...`, the em-dash finding quotes
`S1—Complete dataset`, and each finding names its chunk.

## 19. Recommendation for RAG-014

RAG-013 is a thin layer by design, and the two places it will feel thin first are the same
places the limitations above point at:

1. **A session/workspace object** that holds "my current chapter" across runs, so
   CONSISTENCY can compare a *named* draft rather than whatever the query retrieved, and
   WRITING can continue the real text instead of writing a free-standing paragraph.
2. **Contradiction detection with a source pair.** The comparison engine already matches
   draft sentences to literature sentences; the missing piece is a claim-level comparison
   where two *sources* disagree about the same quantity or finding, which is the check that
   would make `places where newer evidence conflicts with old wording` a first-class
   finding rather than a numeric special case.
3. **Mode-level evaluation set.** The five modes are validated here on the five queries the
   brief names; an evaluation file (like RAG-011's) would make mode behaviour regressions
   visible when the corpus or the model changes.

## Appendix — how the validation artifact was produced

A temporary driver (`_rag013_validate.py`, deleted after the run) built one `ModeService`
over the real settings, ran the five `(mode, query)` pairs from §12 in order, and wrote the
per-mode result summary, safety block, output, sources-with-roles and warnings to
`docs/rag013-mode-validation.txt`, followed by a summary table and the total wall clock.
The safety block is computed from the returned `ModeResult` — it counts citations whose
source is not evidence, external records flagged usable as evidence, local blocks whose
corpus role is not in the mode's `context_policy`, withheld claims, and citations carrying
a chunk id — so what the artifact reports is exactly what the service returned.

## RAG-013 COMPLETE

Five modes, one preset each, one orchestration service, one new deterministic comparison
engine, 95 new tests, 990 passing in total, and a real end-to-end artifact in which every
mode does what it promises and no non-evidence material becomes a citation.
