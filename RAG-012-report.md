# RAG-012 — Claim-to-source validation and Harvard citations

Status: **complete** — every claim sentence of a RAG-011 answer is now checked against the
sources it actually cites, and the internal `[Sx]` markers are converted into Harvard
in-text citations with one deduplicated reference entry per cited work. The validation is
deterministic by default (no language model decides, and none is allowed to override a
deterministic failure); an optional local BGE-M3 layer is available for the claims the
deterministic checks cannot decide and can never upgrade partial support to full support.
126 new tests; **895 tests pass** in total.

Scope guard: RAG-012 validates and cites. It does **not** regenerate the answer, does not
rewrite a claim with a model, does not build UI or agents, does not add retrieval or
generation logic, does not import anything into `data/`, does not re-index, does not touch
the Qdrant collection (read-only, by chunk id) and does not begin RAG-013.

---

## 1. Files created

| Path | Purpose |
|------|---------|
| [app/citations/__init__.py](app/citations/__init__.py) | Package exports (44 names) |
| [app/citations/models.py](app/citations/models.py) | `SupportVerdict`, `ValidatorMethod`, `ClaimKind`, `ClaimDecision`, `NumericKind`, `NumericMention`, `QuotationCheck`, `SourceSupport`, `ClaimValidation`, `ReferenceEntry`, `CitationMapEntry`, `ValidationPolicy`, `ValidationSummary`, `CitationValidationResult` |
| [app/citations/claims.py](app/citations/claims.py) | `sentence_spans` (exact spans into the answer), `source_sentences` (re-joins hard-wrapped PDF lines), `classify_sentence`, `content_terms`/`coverage` (canonicalised content words), `extract_numeric_mentions`, `NumericIndex`/`match_numeric`, `extract_quotations` |
| [app/citations/support.py](app/citations/support.py) | `validate_claim`, `validate_answer`, the deterministic checks, `lexical_result` (support bands), `merged_findings` (the RAG-011 failure-mode detector), `quotation_level`, `LexicalEntailmentChecker`, `EmbeddingEntailmentChecker` |
| [app/citations/harvard.py](app/citations/harvard.py) | `split_author_string`, `author_label`, `year_label`, `normalized_source_doi`, `reference_identity`, `format_reference`, `build_reference_list` (dedup), `citation_for_ids`, `citation_map` |
| [app/citations/service.py](app/citations/service.py) | `CitationService.validate`/`answer_and_validate`, the publication policy, the marker rewriting, warnings and summary |
| [app/citations/inspect.py](app/citations/inspect.py) | `python -m app.citations.inspect` CLI |
| [tests/citations_support.py](tests/citations_support.py) | Offline fixtures: real `SourceBlock`/`ContextPackage`/`GenerationResponse` builders (the RAG-011 shapes, not stand-ins) |
| [tests/test_citations_validation.py](tests/test_citations_validation.py) | 47 tests: claim support, markers, multi-source, numerics, quotations, OCR/table risk, the merging detector, the optional semantic layer |
| [tests/test_citations_harvard.py](tests/test_citations_harvard.py) | 30 tests: author parsing, citation forms, fallbacks, DOI identity, deduplication, ordering, reference formatting |
| [tests/test_citations_service.py](tests/test_citations_service.py) | 35 tests: end-to-end validation, the publication policy, references, warnings, the recorded RAG-011 partial claim |
| [tests/test_citations_cli.py](tests/test_citations_cli.py) | 14 tests: sections, flags, JSON, exit codes |
| [docs/rag012-citation-validation.txt](docs/rag012-citation-validation.txt) | The real run: five questions claim by claim, the recorded RAG-011 answers re-validated, the semantic layer, targeted checks on real chunk text, acceptance summary |
| [docs/rag012-harvard-output.txt](docs/rag012-harvard-output.txt) | The validated answers with Harvard citations and their reference lists |
| [RAG-012-report.md](RAG-012-report.md) | This report |

## 2. Files modified

| Path | Change |
|------|--------|
| [app/generation/grounding.py](app/generation/grounding.py) | Two public wrappers (`sentence_is_trivial`, `sentence_has_claim_signal`) so RAG-012's sentence classification is RAG-011's, not a second copy |
| [app/generation/__init__.py](app/generation/__init__.py) | Export the two wrappers |
| [README.md](README.md) | RAG-012 CLI and the validation/citation rules |

No file under `data/` was modified. The Qdrant collection was opened read-only (to read
back the exact chunks a recorded RAG-011 answer cited) and nothing was written to it.

## 3. Architecture

```
RAG-011 GenerationResponse (answer + source_map + context + audits)
   │
   ├─ sentence_spans(answer)                    exact spans, layout preserved
   ├─ classify_sentence(...)                    claim | non-evidence report | unmarked | non-claim
   │
   ├─ validate_claim(sentence, cited blocks)    app/citations/support.py
   │     1 determinant checks  id exists | evidence-eligible | text present |
   │                           not a reference list | quotation verbatim |
   │                           every numeral present or derivable
   │     2 support bands       how much of the claim the evidence contains
   │     3 merging detector    were two separately-sourced findings joined?
   │     4 optional checker    BGE-M3, only for what 1-3 could not decide
   │
   ├─ build_reference_list(cited blocks)        DOI → title → document identity, dedup
   ├─ citation_for_ids(...)                     (Author, Year) in a fixed order
   ├─ rewrite sentences                         markers → citations, policy applied
   └─ CitationValidationResult                  validated_answer, references, claims,
                                                citation_map, warnings, summary
```

The dependency direction is one way: `service` → `support`/`harvard`/`claims` → `models`.
`models.py` imports nothing from the package; `support.py` reuses RAG-011's sentence
splitter and marker regex (`app.generation.grounding`), RAG-009's bibliography detector
(`app.retrieval.bibliography.is_bibliography_like`) and RAG-006's DOI normalizer
(`app.metadata.doi.normalize_doi`), so the stage cannot disagree with the pipeline it
validates. The optional embedding checker is imported lazily and only when a caller asks
for it: the default path loads no model beyond the retrieval RAG-011 already ran.

## 4. What is validated, and how the answer is rebuilt

RAG-011 answers are prose with one `[S1]` marker per claim sentence. RAG-012 sets
`sentence_spans(answer)` — the sentences RAG-011's abbreviation-protected splitter produces,
located as *exact spans* in the original string — and rebuilds `validated_answer` by
substituting only those spans. Nothing is re-flowed, paragraph breaks survive, and
`original_answer` is kept byte-for-byte for auditability (§1, §13).

Every sentence is classified before it is judged:

| Kind | What it is | What RAG-012 does |
|------|-----------|-------------------|
| `claim` | a substantive sentence citing at least one citable block | fully validated |
| `non_evidence_report` | a sentence RAG-011 recorded as a report about the author's draft or the guidance | kept, marker replaced by a provenance note, no reference |
| `unmarked_claim` | a finding-shaped sentence with no marker (RAG-011 already reports these) | recorded as `UNCERTAIN`, never cited |
| `non_claim` | heading, bullet, question, transition | kept; a *marker* on a heading is removed so no internal marker can survive |

A sentence citing only non-citable material is **not** waved through: unless RAG-011
already classified it as a report, it is a claim and fails validation.

## 5. Deterministic checks (§3)

These run first and are never overridden by a similarity score or a model:

| Check | Failure verdict | Method |
|-------|-----------------|--------|
| The marker exists in the supplied context | `UNSUPPORTED` | `invalid_marker` |
| The cited block may support scholarly claims (evidence, not `my_work`/`guidance`/metadata-only) | `UNSUPPORTED` | `non_evidence_source` |
| The cited block carries source text | `UNSUPPORTED` | `missing_source_text` |
| The cited chunk is not a reference list (RAG-009's detector) | `UNSUPPORTED` | `bibliography_like_source` |
| Every direct quotation occurs in the *cited* source | `UNSUPPORTED` | `quotation_mismatch` |
| Every numeral occurs in the cited evidence or is directly derivable | `UNSUPPORTED` | `numeric_missing` |
| Author, year, title, DOI and page provenance are recorded | warnings + fallbacks | — |
| The DOI normalizes to the canonical `10.<registrant>/<suffix>` form | fallback to title identity + `metadata_warning` | — |

A deterministic failure records `confidence=0.95` and the exact reason, and the claim is
withheld from `validated_answer`.

## 6. Support bands, and how they were calibrated (§4, §5, §15)

The support layer answers "does the evidence contain this claim?" with two measurements:

* **`union_coverage`** — the share of the claim's content terms that occur anywhere in the
  cited passages. A paraphrase legitimately spreads its wording over several sentences of a
  3,000-4,000 character chunk, so this is the band that decides support.
* **`merged`** — whether quantified units of the claim rest on *different* source sentences
  (§7 below). This is the RAG-011 failure mode in testable form.
* **`single_sentence_coverage`** is reported for every claim, so an auditor can see whether
  a supported claim was stated in one place or assembled from several passages.

| Verdict | Condition |
|---------|-----------|
| `SUPPORTED` | evidence contains ≥ 60% of the claim's content terms *and* no merging detected |
| `PARTIALLY_SUPPORTED` | evidence contains 35-60% of the claim's content terms, **or** quantified findings from different passages were merged |
| `UNCERTAIN` | evidence contains 15-35% of the claim's content terms — never upgraded silently |
| `UNSUPPORTED` | a deterministic check failed, or the evidence overlaps the claim by < 15% |

Claims shorter than four content terms take the conservative short-claim branch (supported
only when the source shares their content, never `UNSUPPORTED` on overlap alone), and
claims resting on **OCR text or a table** are validated against stricter thresholds
(≥ 75% for `SUPPORTED`, ≥ 50% for `PARTIALLY_SUPPORTED`, and a higher merging floor) with
`elevated_validation_risk=true` recorded per claim — flagged, never auto-rejected (§8).

## 7. The merging detector: catching the RAG-011 failure mode

RAG-011's single `PARTIALLY_SUPPORTED` claim joined two findings from one work into one
sentence:

> "A majority of faculty members consider training on ethical AI usage as highly important,
> with many advocating for assignments that foster critical thinking to mitigate
> overreliance" — [RAG-011 §15](RAG-011-report.md)

`merged_findings()` tests exactly that shape: split the claim at its connectives, keep the
units that quantify a finding ("a majority …", "many …", "56.7% …"), locate the best-matching
*source sentence* for each, and require two things — every unit is grounded somewhere
(≥ 40% coverage) and the units rest on **different** passages — while no single passage
states the whole claim (otherwise nothing was merged). Measured on the corpus, the check
fires on **1 of 51 real claim sentences**, and that one is the documented case:

```
8. claim  PARTIALLY_SUPPORTED  method=lexical_merged_findings  sources=S3
   A majority of faculty members consider training on ethical AI usage as highly
   important, with many advocating for assignments that foster critical thinking to
   mitigate overreliance [S3]
   reason: two separately-sourced findings are joined in one sentence: 'A majority of
   faculty members consider training on ethical AI usa' matches passage 2, 'many advocating
   for assignments that foster critical thinkin' matches passage 9, and no single passage
   states the whole claim (47% at best)
```

This is the check §5 and §15 ask for: the claim is preserved, cited and annotated, it is
listed in the warnings, and it is **not** presented as fully validated.

## 8. Multi-source claims (§5)

For a sentence citing several sources, RAG-012 records per-source support
(`SourceSupport`: verdict, method, confidence, best-sentence and union coverage, matched and
missing numbers, quotations found per source, metadata warnings) *and* the combined support.
A claim that only holds over the combination is recorded as such: `combination_required=true`
with the warning "no single cited source supports the claim on its own; the combination does".
A weak source cited next to a strong one does not drag the claim down, and a weak source
never gets credit: `per_source` shows exactly which source carried which part.

## 9. Numeric claims (§6)

Percentages, p-values, ranges, years, sample sizes/counts and bare numerals are extracted in
a fixed priority order (so `p < 0.05` is one p-value, not a p-value plus `0.05`), with the
`[Sx]` markers stripped first so the `1` in `[S1]` is never read as a quantity. Each value
must be present in the cited evidence, allowing only *direct derivations*: `75%` ↔ `0.75`,
`p<0.05` ↔ `p < 0.05`, thousand separators, decimal commas, and a year taken from the cited
work's own metadata (so "Mikropoulos and Natsis (2011)" is satisfied by a 2011 source whose
body does not repeat the year). **A missing number is `UNSUPPORTED` with
`confidence=0.95`**, and semantic similarity is never consulted first — on real text:
`88%` absent → `UNSUPPORTED [numeric_missing]`, while `56.7%` present → `SUPPORTED`
([docs/rag012-citation-validation.txt](docs/rag012-citation-validation.txt), Part E).

## 10. Quotations (§7)

A direct quotation must occur in the source the sentence cites. Matching is exact after
Unicode and whitespace normalisation (recorded as `verbatim`); a case/punctuation-insensitive
match of the same word sequence is accepted and recorded as `normalized`; a quotation elided
with `...` is accepted only when **every** fragment is located and is recorded as `elided`.
There is no similarity threshold: on real text, the verbatim quote is
`SUPPORTED [quotation_exact]` and a one-figure change (`60%` for `56.7%`) is
`UNSUPPORTED [quotation_mismatch]`. A quotation found only in a *different* source than the
one cited also fails, with `outside_cited_sources=true` so the mistake is visible.

## 11. Harvard in-text citations (§9, §10)

| Metadata | Citation |
|----------|----------|
| one author | `(Smith, 2024)` |
| two authors | `(Smith and Jones, 2024)` |
| three or more, or `et al.` in the metadata | `(Smith et al., 2024)` |
| missing year | `(Smith, n.d.)` + `missing_year` |
| missing or unusable author | `("Short title", 2024)` + `missing_author` |
| no author and no title | `(Untitled work, 2024)` + both fallbacks |

Multiple sources in one marker are ordered deterministically — alphabetical by author label,
then year, then identity key — so `[S1,S4,S7]` always renders as
`(Brown, 2022; Chan, 2024; Smith et al., 2023)`. Several markers of one group that point at
chunks of the *same* work are cited **once** (the work is one reference entry either way).
Nothing is invented: a missing author or year is never guessed, and the fallback applied is
recorded per citation in `citation_map`.

Author strings are parsed from the metadata the corpus actually stores
(`first_author`): `"Jane Doe and John Smith"`, `"Doe, J. and Smith, J."`,
`"Jane Doe, John Smith, Anna Lee"`, `"Merchant et al."`, `"Jan van der Berg"`. Metadata that
cannot be an author — a watermark (`ssrn`), a stray single letter (from
`Japheth Kodua Wiredu a`), a fragment with digits — is rejected and the documented title
fallback is used instead, with the reason recorded: `(ssrn, n.d.)` is not a citation.

## 12. Reference list, identity and deduplication (§11, §12)

One entry per **work**, never per chunk. Identity is the DOI first (normalized, compared
case-insensitively), then the normalized title, then the document, then the chunk. Eight
chunks of one paper collapse into one entry that lists all its source ids and chunk ids:

```
Magaña, M.A.R. (2023) 'A NEW EDUCATIONAL APPROACH IN THE HYBRID ADAPTIVE LEARNING MODEL IN
HIGHER SECONDARY EDUCATION', pp. 5-6. doi:10.22533/at.ed.5583332308092
```

Only metadata that exists is used — no volume, issue, publisher, page range, URL or DOI is
ever invented — and gaps are surfaced as `metadata_warning` on the entry
(`missing_author`, `missing_year`, `missing_title`, `missing_doi`). A work without an author
puts its title in the author position (`"Title" (2024), pp. 6-7. doi:…`), which is the
Harvard convention for authorless works. Entries are ordered alphabetically by author label,
then year, then identity key.

## 13. Result and publication policy (§13, §14, §15)

`CitationValidationResult` carries `validated_answer`, `reference_list` (+ structured
`references`), `claim_validations`, `citation_map`, `marker_groups`, `warnings`,
`validation_summary`, the RAG-011 `source_map`, the `policy` that was applied, the
`original_answer`, the generation status/warnings and the timings.

The policy is fixed and stated in the result (`policy_note`):

| Verdict | What happens |
|---------|--------------|
| `SUPPORTED` | marker replaced by its Harvard citation |
| `PARTIALLY_SUPPORTED` | kept with its citation plus a visible `[partially supported]` annotation; warned |
| `UNCERTAIN` | kept with its citation plus `[unverified]`; warned |
| `UNSUPPORTED` | **withheld**: the sentence is replaced by `[claim withheld: the cited source does not support it]` and listed in the warnings |
| non-evidence report | kept, marker replaced by `[author's own work]` / `[supervision guidance]` / `[external record: metadata only]` |
| marker on a heading | removed, counted in `markers_removed` |

The answer is never rewritten by a model, and no internal marker can survive into
`validated_answer` (asserted for every real run, see Part A "INTEGRITY").

## 14. CLI (§16)

```
python -m app.citations.inspect "What is AI governance in higher education?"
python -m app.citations.inspect "..." --show-claims --show-source-map
python -m app.citations.inspect "..." --show-warnings --show-original
python -m app.citations.inspect "..." --semantic --support-coverage 0.7
python -m app.citations.inspect "..." --json
```

Output sections: `VALIDATED ANSWER`, `REFERENCES`, `VALIDATION` (supported / partial /
unsupported / uncertain, plus the claims that need attention), then `CLAIMS`,
`SOURCE MAP`, `WARNINGS` and `ORIGINAL RAG-011 ANSWER` on request. `--json` emits the whole
structured result plus a `generation` block. Generation options (`--model`, `--length`,
`--final-k`, `--no-rerank`, `--context-tokens`, `--min-rerank-score`, modes) are passed
through to RAG-011 unchanged. Exit codes: `0` validated, `1` the model could not answer,
`3` at least one claim was withheld.

## 15. Real validation (§18)

Run against the real corpus, the real RAG-009 retrieval and the real `qwen3.5:35b` through
Ollama. Nothing was edited by hand; both artifacts are the recorded output.

**Part A — the five RAG-011 evidence questions, fresh runs:**

| # | Question | Claims | Supported | Partial | Unsupported | Uncertain | References |
|---|----------|--------|-----------|---------|-------------|-----------|------------|
| 1 | What is AI governance in higher education? | 12 | 12 | 0 | 0 | 0 | 3 |
| 2 | What evidence links virtual reality with student engagement? | 12 | 12 | 0 | 0 | 0 | 3 |
| 3 | How is generative AI affecting academic integrity? | 9 | 7 | 1 | 0 | 1 | 4 |
| 4 | What evidence supports adaptive learning in higher education? | 9 | 8 | 1 | 0 | 0 | 3 |
| 5 | What is Design Science Research Methodology? | 9 | 8 | 1 | 0 | 0 | 2 |
| | **total** | **51** | **47** | **3** | **0** | **1** | **15** |

51 markers were replaced by Harvard citations, 0 internal markers survive, 0 invalid markers,
0 non-evidence sources cited, and every citation maps to a real citable source with a chunk
id (recorded per question under `INTEGRITY`). The three `PARTIALLY_SUPPORTED` claims and the
one `UNCERTAIN` claim are annotated in the validated answer and listed in the warnings; none
of them was withheld, because none of them failed a deterministic check.

**Part B — the recorded RAG-011 answers, re-validated as they were produced.** The answers
and the chunk ids are read back from
[docs/rag011-generation-evaluation.txt](docs/rag011-generation-evaluation.txt), and the exact
chunks are read from the corpus by id; nothing is regenerated. The verdicts are identical to
Part A in every question, which is the reproducibility check for this stage. Most
importantly, **the documented partially-supported claim is caught**:
`PARTIALLY_SUPPORTED [lexical_merged_findings]` in the integrity question, with the sentence
preserved in the audit, annotated in the validated answer and reported as a warning — and the
RAG-011 manual evaluation's other claims stay `SUPPORTED`.

**Part C — the optional semantic layer.** With `--semantic`, only the claims the
deterministic checks could not decide are escalated to BGE-M3. In the integrity question the
`UNCERTAIN` claim ("Despite the potential for enhanced learning efficiency, …") is decided by
the embedder and lands on `PARTIALLY_SUPPORTED` with method `embedding_semantic` — the layer
resolved the case without upgrading it, and no claim moved from partial to full support.

**Part E — targeted checks on real chunk text.** The deterministic layers demonstrated on
real evidence: `56.7%` present → `SUPPORTED`; `88%` absent → `UNSUPPORTED [numeric_missing]`;
`p < 0.05` absent from that chunk → `UNSUPPORTED [numeric_missing]`; a verbatim quotation →
`SUPPORTED [quotation_exact]`; the same quotation with one figure altered →
`UNSUPPORTED [quotation_mismatch]`; a real `my_work` chunk cited as evidence →
`UNSUPPORTED [non_evidence_source]`; a real OCR chunk → `UNCERTAIN` with
`elevated_validation_risk=true` (flagged and checked more strictly, not rejected).

## 16. Performance

| Step | Measured |
|------|----------|
| Deterministic validation of one answer | **0.029-0.043 s** |
| Validation with one embedding escalation | 0.215 s |
| Retrieval (first question, models loading) | 10.86 s |
| Retrieval (subsequent questions) | 0.44-1.11 s |
| Generation (`qwen3.5:35b`, normal length) | 1.29-1.71 s |
| Whole validation run in one process: 5 fresh questions + 5 recorded replays + 1 semantic run | **25.4 s** |

Validation is pure string work over the context that was already in memory: it needs no
model, no network and no extra corpus read, which is why it costs a few milliseconds per
answer.

## 17. Test result

```
python -m pytest -q
895 passed, 2 warnings in 9.40s
```

* 769 pre-existing tests (RAG-001 … RAG-011) pass unchanged.
* 126 new tests: validation 47, Harvard 30, service 35, CLI 14.
* No unit test needs Ollama, the Internet or the corpus: the fixtures build real RAG-011
  models offline. The real-model and real-corpus work is the validation documented above.
* The two warnings are pre-existing (a FastAPI/Starlette deprecation notice and the local
  Qdrant payload-index notice) and unrelated to this stage.

The 18 required cases of §17 are covered: valid one-source claim, valid multi-source claim,
invalid source marker, non-evidence source, unsupported numeric claim, supported numeric
claim, exact quotation success, exact quotation failure, partially supported claim, duplicate
chunks → one reference, two-author citation, 3+ author citation, missing-year fallback,
missing-author fallback, DOI normalization, deterministic citation ordering, OCR risk flag,
table risk flag — plus marker-removal, combination-required, threshold and semantic-layer
tests.

## 18. Acceptance criteria

| Criterion | Result |
|-----------|--------|
| All prior 769 tests pass | ✅ 895 pass in total |
| All new tests pass | ✅ 126 new |
| Every citation maps to a real source | ✅ 51/51 citations, every one with a chunk id (`INTEGRITY` per run) |
| No non-evidence source becomes a scholarly citation | ✅ 0 (`my_work`/guidance/metadata-only citations fail validation) |
| Unsupported numeric claims are caught | ✅ `88%` and `p < 0.05` absent → `UNSUPPORTED [numeric_missing]` on real text |
| Quotations are exact | ✅ `verbatim` accepted, one-figure change rejected, elision requires every fragment |
| Duplicate chunks collapse to one reference | ✅ one entry per DOI/title, all source ids listed |
| Harvard citations generated deterministically | ✅ alphabetical then year, stable for a given source set |
| The known partial-support case is detected | ✅ `PARTIALLY_SUPPORTED [lexical_merged_findings]`, annotated, warned, never published as validated |
| Unsupported claims are not silently published | ✅ withheld from `validated_answer`, listed in warnings |
| OCR/table evidence flagged | ✅ `elevated_validation_risk=true` + stricter thresholds |
| Original RAG-011 answer preserved | ✅ `original_answer`, byte-for-byte |

## 19. Known limitations

1. **Coverage is a lexical proxy for entailment.** The support bands measure the share of a
   claim's content terms that the evidence contains; they do not read the claim as a
   proposition. A claim can be lexically well covered and logically wrong (an inverted
   relation, a wrong attribution) — the deterministic layer cannot see that, and the report
   does not pretend it can. The optional embedding layer is a similarity check, not an
   entailment model.
2. **The band that was rejected is recorded, not hidden.** The first implementation decided
   support on the best *single source sentence*; measured against RAG-011's manual
   evaluation it flagged 4/12 and 8/12 claims in two questions as partial, because a
   paraphrase legitimately spans several sentences of a 3,000-4,000 character chunk — and the
   one known partial claim (47% single-sentence coverage) sat *inside* the distribution of
   claims the manual evaluation called supported, so no single-sentence threshold could
   separate them. The bands were recalibrated (§6) and the documented failure mode got its
   own structural test (§7). What is lost is the ability to say "the evidence states this in
   one place": that signal is now *reported* (`single_sentence_coverage`) rather than decided.
3. **The merging detector is deliberately narrow.** It fires on two or more *quantified* units
   of one sentence resting on different passages. A merged claim without quantifiers ("X
   improves retention and reduces cost") is not detected by it; it would have to be caught by
   the coverage bands (partial below 60%) or by a human.
4. **Sentence splitting is RAG-011's heuristic.** Abbreviations are protected, but a
   mis-split sentence is validated as two units. Such a case can only make a verdict more
   conservative (a split claim has less coverage per part), never less.
5. **Stemming is deliberately crude** (suffix stripping plus trailing-`e` removal): it makes
   fewer *false* matches than an embedding would, and more *missed* matches (for example
   `study`/`studies` will not match). It is deterministic and auditable, which is the point.
6. **Metadata quality is upstream.** Eight of the fifteen references in the real run carry a
   warning — five have a missing or unusable author, four a missing DOI, one a missing year —
   because the PDF extraction did not find them. RAG-012 rejects unusable values and records
   the fallback; it does not re-extract metadata (that is RAG-006/007 territory).
7. **No author-year disambiguation (`2024a`/`2024b`).** Two different works by one author in
   one year yield the same in-text citation. The reference list still separates them, and the
   citation map identifies the source id, but a reader cannot tell them apart in the prose.
8. **Annotated prose.** Partial and uncertain claims carry a visible
   `[partially supported]`/`[unverified]` marker inside `validated_answer`. That is a policy
   choice (§15: never present a partial claim as fully validated) and it makes the answer
   noisier for a reader who expected clean prose.
9. **UNCERTAIN is a real outcome, not a hedge.** A claim whose wording shares 15-35% of its
   content with the evidence stays uncertain unless the semantic layer is enabled; that is
   exactly what happens to one claim in the integrity answer.
10. **The OCR/table thresholds are a judgement call.** They were set from the reasoning of §8
    (scanned text and table cells change the evidence itself), not from a labelled OCR
    evaluation set; the flag is recorded per claim so the choice is visible.

## 20. Validation defects found and fixed during this stage

Recorded because failures are evidence, not noise:

1. **Over-flagged partial support (the serious one).** Deciding support on the best single
   source sentence gave 4/12 partial in question 1 and 8/12 in question 2, against a manual
   evaluation that called those claims supported. Diagnosed by measuring, not by tuning
   blindly: the known partial case (47% single-sentence) was *not* an outlier, so the metric
   could not separate it. Fixed by (a) deciding support on how much of the claim the evidence
   contains, (b) detecting the documented failure mode structurally, and (c) recording the
   single-passage coverage instead of deciding with it. Result: 1 of 51 claims flagged as
   merged, and it is the known case.
2. **Hard-wrapped PDF lines fragmenting source sentences.** A line break inserted by PDF
   extraction split one source sentence into several, which depressed single-passage coverage
   further. Fixed with `source_sentences`, which re-joins a line unless the previous one ends
   with terminal punctuation. (On this corpus the stored chunk text is already joined, so the
   fix changed no verdict here — it is robustness, verified by unit test.)
3. **`p < 0.05` parsed as nothing.** The p-value mention was built by splitting the raw match
   on whitespace, which failed when the model wrote `(p<0.05)` with no space; the mention
   then had no value and every p-value claim looked unverifiable. Fixed by extracting the
   numeral from the match, with a test for the spacing difference.
4. **`(Magaña, 2023; Magaña, 2023)`.** A sentence citing two chunks of one work produced the
   same citation twice, because each marker id was converted independently. Fixed by citing
   the *work* once per marker group.
5. **Watermark and fragment "authors".** Real metadata produced `(ssrn, n.d.)` and
   `(a, 2024)`. Fixed by rejecting author values that cannot be authors (a lone lower-case
   token, a single character, a fragment with digits) and using the documented title fallback
   with the reason recorded.
6. **Title fallback colliding with its own punctuation.** Wrapping a short title in single
   quotes broke on titles containing an apostrophe
   (`('Information Technology Students' Perceptions Toward', 2024)`). Fixed by using double
   quotes for the title fallback and for the authorless reference position.
7. **`Candidate` unused in the replay path.** The first version of the corpus replay built a
   RAG-009 `Candidate` only to copy its fields into a `SourceBlock`; the mapping is now direct
   and total, so the replay cannot silently diverge from `context.py`.

## 21. Recommendation for RAG-013

1. **Decide with entailment, not coverage, if the flags must become verdicts.** The bands are
   honest about what they measure, but a claim can be well covered and logically wrong. A
   local NLI model (or the same Qwen model with a strict, schema-checked prompt over the
   *already validated* claims) is the natural next step — and it should be an escalation for
   `UNCERTAIN`/`PARTIALLY_SUPPORTED` only, with the deterministic failures still final.
2. **Improve the metadata before the citations.** Half of the real references carry a warning;
   fixing author and year extraction (RAG-006/007) improves RAG-012's output more than any
   change to the formatter.
3. **Add author-year disambiguation** (`2024a`) when two references share author and year, and
   a bilingual style switch (`and`/`&`, `n.d.`/`s.a.`) if a non-English citation style is ever
   needed.
4. **Consider a `--strict` mode** that withholds `PARTIALLY_SUPPORTED` and `UNCERTAIN` claims
   as well, for a supervisor who wants only fully verified prose.
5. **Keep the policy visible.** Any later stage that reuses `validated_answer` must preserve
   the `[partially supported]`/`[unverified]` annotations and the withheld sentences, or the
   guarantee that unsupported claims are never published silently is lost.
6. **Do not let the citation layer re-index or re-generate.** RAG-012's value is that it
   changes nothing but the markers; a stage that "fixes" claims in the text would destroy the
   audit trail this one produces.

## Appendix — how Part B was produced

Part B re-validates the *recorded* RAG-011 answers, which is the §1 route ("reuse the existing
`GenerationResponse`; do not regenerate the answer"). The run was a temporary script, in four
steps, and needs no model:

1. parse [docs/rag011-generation-evaluation.txt](docs/rag011-generation-evaluation.txt) into
   one record per evidence question: the question, the `S<n> → chunk id` map from its
   `RETRIEVED SOURCES` block, and the `RAW MODEL ANSWER` text (whitespace-normalised);
2. read each chunk back from `phd_rag_chunks_v1` by id (read-only) and build a
   `SourceBlock` per chunk with the same fields `app/generation/context.py` uses
   (`title`, `first_author`, `year`, `doi`, pages, `corpus_role`, `evidence_eligible`,
   `ocr_involved`, `content_type`, `chunk_id`, `document_id`);
3. assemble the real `ContextPackage` and a `GenerationResponse` carrying that answer and
   `source_map`, with `marker_audit` empty;
4. run `CitationService().validate(response)` — the same code path the CLI uses.

The same script produced Part E by validating hand-written claim sentences against real chunk
ids (the cited chunk of the integrity answer, one real `my_work` chunk and one real
`ocr_involved` chunk, both located by a Qdrant filter).

---

## RAG-012 COMPLETE

Full `pytest`: **895 passed** (769 pre-existing + 126 new). Artifacts:
[RAG-012-report.md](RAG-012-report.md),
[docs/rag012-citation-validation.txt](docs/rag012-citation-validation.txt),
[docs/rag012-harvard-output.txt](docs/rag012-harvard-output.txt).
