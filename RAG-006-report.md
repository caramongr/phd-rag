# RAG-006 Report — Academic metadata extraction and controlled enrichment

Task: implement document-level academic metadata extraction (title, authors, year,
DOI, venue, publication type, URL) with explicit provenance and confidence, plus
controlled Crossref/OpenAlex enrichment. No chunking, embeddings, Qdrant,
reranking, generation, Harvard citations or UI were implemented.

## 1. Files created

Application package `app/metadata/`:

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `enums.py` | `MetadataSource`, `Confidence`, `PublicationType`, `ExternalMatchStatus`, `CorpusRole` |
| `models.py` | `MetadataValue[T]`, `Author`, `AcademicMetadata` |
| `text.py` | Comparison/normalisation helpers (no destructive cleaning) |
| `layout.py` | `PageLine` / `PageLayout` typographic primitives |
| `doi.py` | DOI detection, normalization, validation |
| `titles.py` | Title validity checks and first-page title candidate extraction |
| `authors.py` | Conservative author parsing and first-page author detection |
| `year.py` | Publication-year extraction from text and PDF/DOCX dates |
| `identifiers.py` | arXiv identifiers and filename hints |
| `pdf_reader.py` | Read-only PDF embedded metadata + first-page layout reader |
| `docx_reader.py` | Read-only DOCX core-properties reader |
| `corpus.py` | Corpus-role detection and enrichment gating |
| `resolver.py` | Field-specific deterministic resolver |
| `extractor.py` | Local metadata extraction orchestrator |
| `enrichment.py` | External enrichment orchestrator |
| `external/http.py` | HTTP client with timeout, bounded retry, rate limiting |
| `external/cache.py` | JSON response cache |
| `external/clients.py` | Crossref and OpenAlex clients + `ExternalRecord` |
| `external/matching.py` | Match/ambiguity rules |
| `inspect.py` | `python -m app.metadata.inspect` CLI |
| `report.py` | `python -m app.metadata.report` corpus reporting CLI |
| `doi_stage.py` | DOI-only corpus enrichment stage + `python -m app.metadata.doi_stage` CLI |
| `title_stage.py` | Title-based enrichment validation sample + `python -m app.metadata.title_stage` CLI |

Tests: `tests/metadata_support.py` and
`tests/test_metadata_{pdf,doi,titles,year,models,local_extractors,external,corpus_roles,cli,doi_stage,title_stage}.py`
(98 tests).
Validation artifacts: `docs/rag006-offline.txt`, `docs/rag006-sample.txt`,
`docs/rag006-doi-stage.txt` (Stage 1, before the title correction),
`docs/rag006-doi-stage-corrected.txt` (Stage 1 rerun after it) and
`docs/rag006-stage2a-sample.txt` (Stage 2A validation sample).

## 2. Files modified

* `app/config.py` — added RAG-006 metadata-enrichment settings.
* `pyproject.toml` — declared the runtime dependencies (see §3).
* `.gitignore` — ignore `.metadata_cache/`.
* `README.md` — documented the metadata module, settings and CLI.

`Document`, `PhysicalLocation`, the ingestion scanner and the parsing layer are
**unchanged**; RAG-006 adds a parallel, additive metadata model.

## 3. Dependencies added

`pyproject.toml` now declares the runtime dependencies actually required by
RAG-001…RAG-006, with floors matching the verified development environment (no
package was upgraded to obtain a newer version):

```
fastapi, pydantic>=2, requests,
pymupdf>=1.28, python-docx>=1.2, pytesseract>=0.3.13, Pillow>=12.3
```

`pip check` reports no broken requirements. The test-only tools (`pytest`, `httpx`
via the FastAPI test client) are still installed separately and not declared.

## 4. Final pytest result

```
276 passed, 1 warning
```

**Test-count reconciliation (correction to the first draft).** The first report
claimed `225 = 65 new + 160 pre-existing`. Both of those numbers were wrong, and
they were wrong for the same reason: I counted "new" tests with
`pytest -k metadata`, and that keyword also matches **three pre-existing tests**
whose *names* contain the word "metadata":

* `tests/test_document_model.py::test_document_accepts_full_metadata`
* `tests/test_pdf_parser.py::test_uses_pdf_metadata_title`
* `tests/test_pdf_parser.py::test_missing_metadata_title_is_none`

So `-k metadata` reported 65 = 62 new + 3 old, and subtracting 65 from 225 gave a
bogus 160. The truth is `225 = 62 new + 163 pre-existing` at that point. After the
post-review revisions (DOI conflict safeguard, local title correction, Stage-2A sample,
Stage-2B corpus run, closure hygiene) the total is `276 = 113 new + 163 pre-existing`.

RAG-005's **163** is therefore intact and unchanged — no pre-existing test was
lost, renamed or weakened (I did not touch any test to make the arithmetic fit).

Actual current count by RAG stage (derived from per-file collection, matching the
historical totals in each report):

| Stage | Tests added | Cumulative | Files |
|-------|-------------|------------|-------|
| RAG-001 | 19 | 19 | `test_health.py` (2), `test_chunk_model.py` (10), `test_document_model.py` (7 of 9) |
| RAG-002 | 49 | 68 | `test_file_hash.py` (5), `test_physical_location.py` (5), `test_scanner.py` (37), `test_document_model.py` (+2) |
| RAG-003 | 50 | 118 | `test_docx_parser.py` (17), `test_markdown_parser.py` (12), `test_parser_dispatcher.py` (9), `test_parsing_corrections.py` (12) |
| RAG-004 | 25 | 143 | `test_pdf_parser.py` (25) |
| RAG-005 | 20 | 163 | `test_ocr.py` (20) |
| RAG-006 | 113 | 276 | `test_metadata_*.py` (113) |

Cumulative values 68 / 118 / 143 / 163 reproduce the figures stated in the
RAG-002…RAG-005 reports exactly.

## 5. Metadata data model

```python
MetadataValue[T] = { value: T, source: MetadataSource,
                     confidence: Confidence, evidence: str }

Author = { full_name, given_name?, family_name?, orcid? }

MetadataConflict = { field, local_value, external_value, external_source,
                     similarity, route, reason }

AcademicMetadata = {
  file_hash?, filename, source_path, extension, corpus_role,
  title?/authors?/year?/doi?/venue?/publisher?/publication_type?/url?/abstract?/arxiv_id?,
  candidates: dict[str, list[MetadataValue]],   # full audit trail
  external_match_status?,
  conflicts: list[MetadataConflict],            # recorded disagreements
}
```

Resolved fields are *selections over* `candidates`; the candidate history is never
discarded, so a resolved value never destructively replaces the value it superseded.
A withheld external value is preserved in `conflicts` rather than merged (§13).

## 6. Metadata source / provenance model

`MetadataSource` = `PDF_METADATA`, `DOCX_PROPERTIES`, `FILENAME`,
`DOCUMENT_TEXT`, `CROSSREF`, `OPENALEX`, `MANUAL`, plus `FOLDER`.

The listed values are kept strictly distinct — a PDF-metadata title is never
conflated with a first-page text title or a Crossref-verified title. `FOLDER` was
added as one extra distinct value so that the author's own-work classification
(`data/my_work` → `AUTHOR_MANUSCRIPT`) has honest provenance instead of being
mis-labelled as `DOCX_PROPERTIES`; it is always `LOW` confidence so it never
outranks a bibliographic source. No existing source value was collapsed.

Every value carries a concise `evidence` string, e.g. `PDF metadata field "title"`,
`first-page DOI pattern`, `Crossref DOI lookup`. No large excerpts are stored.

## 7. Confidence model

Controlled levels only — `HIGH`, `MEDIUM`, `LOW` (no floating-point
pseudo-precision). Externally DOI-verified values are `HIGH`; strong title-based
external matches are `MEDIUM`; local first-page/embedded candidates are `MEDIUM`;
filename hints and PDF/DOCX file dates are `LOW`.

## 8. DOI extraction strategy

* Robust grammar `10.<4-9 digits>/<suffix>`; a bare `10.x` string is rejected.
* Wrappers stripped: `https://doi.org/`, `http://dx.doi.org/`, `doi:`.
* Trailing sentence punctuation trimmed; suffix validated to contain alphanumerics.
* Deterministic scan order: **first page → PDF metadata (`subject`/`keywords`/…) →
  filename**. First valid DOI wins.
* **Trailing pages / references are deliberately no longer scanned** (post-review
  correction — see §20). A DOI that appears only on the last pages is
  overwhelmingly a DOI *inside the bibliography*, i.e. it belongs to a cited work,
  not to this document. A document's own DOI is essentially always on page one or
  in its embedded metadata, so nothing legitimate is lost in practice.
* **DOCX and Markdown text is never used as a DOI source** (stage-1 correction —
  see §21). A `.docx` draft's body and its bibliography cannot be told apart, and
  the corpus drafts carry up to **110** DOIs in their reference lists; adopting the
  first one mislabels the author's own draft as a published paper. DOCX/Markdown
  still contribute title/author (DOCX properties) and filename hints.
* The suffix character class is restricted to printable-ASCII DOI characters, so
  extraction artefacts (e.g. a U+FFFD replacement character) are not swallowed
  into the DOI.
* Confidence: first-page DOI `HIGH`; PDF-metadata DOI `MEDIUM`; filename DOI `LOW`.
* The resolved DOI also yields a `https://doi.org/…` URL candidate.

## 9. Local title extraction strategy

* Embedded title accepted only if `is_meaningful_title` passes — rejecting
  `untitled`, `document`, `Microsoft Word …`, `Adobe Acrobat`, scan/filename
  artefacts, named section headings, journal headers, ISSN/volume lines and arXiv
  margin stamps — without over-cleaning legitimate titles.
* First-page candidate uses font size (allowed in this stage): body size is
  estimated from the most common line size (median fallback); only lines ≥ 1.15×
  body size in the top 60% qualify. Adjacent same-size lines are joined (wrapped
  titles) while a trailing author line in smaller type is trimmed.
* DOI/URL/e-mail/copyright/affiliation lines are rejected. If two equally
  prominent, similarly positioned candidates exist the result is left unresolved.
* Source `DOCUMENT_TEXT`, confidence `MEDIUM`; missing when uncertain.
* **Post-review correction (§22):** candidates that read as a journal/venue
  masthead, a proceedings template label, a production identifier, a proof marker, a
  reprint notice, a running header or extraction garbage are rejected
  deterministically; after a rejection the extractor continues to the next candidate;
  a line belonging to the title is no longer merged into it; and an embedded metadata
  title that is a venue masthead no longer outranks the page-one title.

## 10. Author / year extraction strategy

**Authors** — order preserved; `full_name` authoritative; given/family split only
for the reliable `Family, Given` form or a two-token name. Sources: PDF metadata
`author` (`MEDIUM`), DOCX core `author` (`MEDIUM`), first-page author lines
(`MEDIUM`, all-or-nothing), filename `name-year`/`name-arxiv` token (`LOW`),
Crossref/OpenAlex (`HIGH`/`MEDIUM`). Affiliations are never treated as authors.
**Post-review correction (§22.3):** title lines are skipped by the first-page author
scan, repository/cover-sheet boilerplate and prose fragments are rejected as names,
all-caps name lines below the title are accepted, and a leading `and`/`by` is
stripped rather than rejecting the name.

**Year** — publication/copyright line on page one (`DOCUMENT_TEXT`/`MEDIUM`) is
preferred; the PDF `creationDate` and DOCX `created` are weak `LOW` candidates
that never override a stronger publication year (verified by test 15); filename
years are `LOW`. Provenance distinguishes the file date from the publication date.

## 11. Crossref integration

`CrossrefClient` does DOI lookups (`/works/{doi}`) and bibliographic title
searches (`/works?query.bibliographic=…`). Type mapping covers journal-article,
proceedings-article, book, book-chapter, dissertation, report, posted-content
(preprint), etc. Raw responses are cached; a polite `mailto` User-Agent is
optional. Lookups never raise — a failure yields `None`/`[]`.

## 12. OpenAlex integration

`OpenAlexClient` does DOI lookups (`/works/https://doi.org/{doi}`) and title
searches (`/works?search=…`). It is the fallback when Crossref yields no
DOI-exact record, and supplies `PREPRINT`/venue/publisher and reconstructed
abstracts (from `abstract_inverted_index`).

## 13. Matching and ambiguity rules

* DOI route: normalised case-insensitive equality → `EXACT`; otherwise `NO_MATCH`.
* Title route: blended similarity (whole-string + token-set). `≥ 0.92` → `STRONG`,
  `< 0.75` → `NO_MATCH`, otherwise `AMBIGUOUS`. Two candidates within `0.03`
  similarity are `AMBIGUOUS`. A year differing by more than one, or a disagreeing
  first-author family, downgrades to `AMBIGUOUS`.
* **DOI conflict safeguard (post-review correction).** An exact DOI is only as
  trustworthy as the DOI extracted from the document, so a DOI-exact record is
  additionally checked against a *strong* local title candidate (source
  `PDF_METADATA`/`DOCX_PROPERTIES`/`DOCUMENT_TEXT` with confidence `HIGH`/`MEDIUM`;
  a filename token is too weak to raise a conflict). If
  `title_similarity(local, external) < 0.60` the record is **materially
  inconsistent**:

  ```
  DOI extracted locally
        ↓
  exact Crossref/OpenAlex DOI result
        ↓
  compare external title with strong local title candidate
        ↓
  materially inconsistent  →  external_match_status = CONFLICT
                              external metadata withheld (not made preferred)
                              both candidates preserved, conflict recorded
  ```

  `CONFLICT` is a new, explicit `ExternalMatchStatus` (distinct from `AMBIGUOUS`,
  which means the *search* returned several equally plausible records). On a
  conflict the whole external record is withheld — not just the title — and a
  `MetadataConflict{field, local_value, external_value, external_source,
  similarity, route}` is appended to `AcademicMetadata.conflicts`. Minor
  punctuation or subtitle differences stay well above `0.60` and are still
  accepted as `EXACT`.
* Only `EXACT`/`STRONG` values are merged; `AMBIGUOUS`/`CONFLICT`/`NO_MATCH`
  change nothing except the recorded `external_match_status` (and, for a conflict,
  the recorded conflict). Merged values are *added* as candidates, so local
  metadata is preserved for inspection.

## 14. Cache / rate-limit behaviour

`HttpClient` enforces a request timeout, bounded exponential-backoff retry on
network errors and 429/5xx, and a monotonic rate limiter (default 1 req/s). A
single-file `JsonCache` keys raw responses by DOI (or normalized title+rows);
cache read/write errors are non-fatal. The enricher additionally memoizes per
file hash so binary-identical duplicates make at most one request set. Offline
mode (`enrich_external=False`, or the CLI `--offline` flag) performs zero requests.

## 15. Offline corpus statistics (phase 1, pre-correction snapshot)

`python -m app.metadata.report data` (read-only; full output in
`docs/rag006-offline.txt`).

**Note:** the numbers below are the phase-1 snapshot taken *before* the §22 title
correction (the artifact has since been regenerated with the corrected extractor:
title 421, authors 391, DOI 250, year 458, publication type known 70, errors 0). The
before/after comparison is in §22.4.

```
unique documents examined: 466      (PDF 444, DOCX 18, MD 3, XLSX 1)
extraction errors:         0
corpus roles: literature 442, rules 8, future_chapters 8, my_work 4,
              supervision 2, summaries 1, guidance 1
title available:           423
title missing:              43
DOI detected:              250
authors detected:          408
year detected:             458
venue detected:              0
publication type known:     70
publication type unknown:  396
source counts (candidates): PDF_METADATA 803, DOCX_PROPERTIES 35,
                            DOCUMENT_TEXT 1669, FILENAME 222
```

Venue is deliberately `0` offline: venue is preferred from external structured
metadata and local venue extraction is intentionally not attempted.

`DOI detected` is **250** after the stage-1 corrections. It went `303 → 265` when
trailing-page/reference-list PDF DOIs were dropped (38 documents), then `265 → 250`
when DOCX/Markdown text DOIs were dropped (15 documents — the corpus drafts' cited
DOIs). Both reductions are the intended "missing beats wrong" trade-off.

## 16. 20-document external sample (phase 2)

`python -m app.metadata.report data --sample 20` (full output in
`docs/rag006-sample.txt`):

| status | count |
|--------|-------|
| EXACT | 4 |
| STRONG | 5 |
| AMBIGUOUS | 4 |
| NO_MATCH | 7 |
| **accepted (EXACT+STRONG)** | **9** |

Lookup routes: 4 DOI, 12 title, 4 none (no usable title/DOI). The sample mixed
DOI-present, DOI-absent-with-strong-title, embedded-title-only, title-missing,
arXiv and non-scholarly-report cases, both recent (2024–2025) and older (2009–2012)
works, and journal articles, conference papers and preprints. Per-document local
metadata, lookup route, match status, resolved metadata, provenance and conflicts
are in the artifact file. This 20-document run was a **pre-correction snapshot**
(executed before the reference-list-DOI fix and before Stage 1); it used title
lookups deliberately as an exploratory check, and no source files were modified.
Corpus-wide **title-based** enrichment remains deferred to Stage 2 (§21); the
corpus-wide **DOI-based** run has since been executed as Stage 1.

`+6089-.pdf` now behaves correctly after the correction: no DOI is adopted, its
true local title is preserved, and its title-based lookup is `AMBIGUOUS` (four
near-identical JLA candidates), so nothing is overwritten.

## 17. False / ambiguous matches observed, and the `+6089-.pdf` investigation

### `+6089-.pdf` — root-cause investigation (requested)

| Item | Finding |
|------|---------|
| Extracted DOI (pre-correction) | `10.18608/jla.2023.7743` |
| Where it was found | **Trailing pages / references only**, as plain text on the last (22nd) page — *not* on page 1 and *not* in the PDF metadata |
| Local title candidate | “Stakeholder Perspectives on the Ethics of AI in Distance-Based Higher Education” (`DOCUMENT_TEXT`, first page, `MEDIUM`) |
| Crossref title for that DOI | “Six Practical Recommendations Enabling Ethical Use of Predictive Learning Analytics in Distance Education” |
| Does the DOI belong to this document? | **No.** The DOI appears exactly once, inside a bibliography entry: “Rets, I., Gillespie, A., & Herodotou, C. (2023). *Six Practical Recommendations …*. Journal of Learning Analytics, 10(1) … https://doi.org/10.18608/jla.2023.7743”. It is a **cited work's DOI**, not this paper's. |
| Ground truth identity of the PDF | Title “Stakeholder Perspectives on the Ethics of AI in Distance-Based Higher Education” by Holmes, Iniesto, Anastopoulou & Boticario, *International Review of Research in Open and Distributed Learning*, 24(2), May 2023 — exactly the local title candidate. |

**Conclusion:** the failure was *not* in matching — the DOI resolved exactly to
what that DOI is. The failure was that the local DOI extraction adopted a DOI from
the reference list. Two changes address it:

1. **Root cause** (§8, §20): trailing pages/references are no longer scanned for
   DOIs, so the reference-list DOI is never adopted (offline DOI count 303 → 265).
2. **Defence in depth** (§13): had that DOI still been extracted, the conflict
   safeguard now fires — verified on the real pair, which produces
   `external_match_status = CONFLICT`, `similarity = 0.5`, the external title
   withheld and the correct local title preserved.

### Other observations

* No accepted match was found to be a false positive elsewhere in the sample.
* Four AMBIGUOUS title lookups (`+6089-.pdf`, two further JLA items,
  `AI_Teachers…`) correctly refused to overwrite local metadata.
* The UNESCO guidance report and the EPRS briefing returned NO_MATCH — correct, as
  they are not bibliographic works.

## 18. Remaining metadata gaps

Final state after the whole pipeline (local extraction → §22 title correction → DOI
Stage 1 → title Stage 2B, §24.8). Populated / missing out of 466 unique documents:

| field | populated | missing |
|-------|----------:|--------:|
| title | 431 | 35 |
| authors | 432 | 34 |
| year | 460 | 6 |
| DOI | 322 | 144 |
| venue | 308 | 158 |
| publisher | 318 | 148 |
| publication type | 354 | 112 |
| URL | 326 | 140 |
| abstract | 206 | 260 |

* Venue/publisher/abstract are **only** obtainable through external enrichment; they
  were 0 before Stage 1 (venue 0 → 237 → 308, publisher 0 → 246 → 318, abstract
  0 → 168 → 206). Publication type went 70 → 312 → 354.
* 35 / 466 documents have no title, and 47 DOI-less documents have no *strong* local
  title, so they are never sent to a title search (the deliberate "filename-only titles
  cannot initiate a search" rule).
* 216 / 466 have **no local DOI** (250 detected) — the intended outcome of dropping
  reference-list DOIs: a missing DOI is preferred over a DOI belonging to another work.
  72 of those were recovered through title matching, so 144 remain DOI-less.
* Author given/family parts are only split for reliable forms; `full_name` is
  authoritative otherwise.

## 19. Architectural concerns

1. **DOI trust (now guarded).** A locally extracted DOI used to be treated as
   fully authoritative. It is now (a) not sourced from reference lists at all, and
   (b) checked against a strong local title, with disagreement surfaced as
   `CONFLICT` instead of an override. Residual risk: when a document has *no* strong
   local title, an exact DOI is still accepted without a cross-check — those 10
   documents are listed explicitly in §22.7 and were not given a new heuristic.
2. **Reference-list DOIs are now dropped, not resolved.** A handful of legitimate
   DOIs that only ever appear in a footer beyond page one, or in a DOCX/Markdown
   body, may be missed. This is the deliberate "missing beats wrong" trade-off
   (38 PDF + 15 DOCX/Markdown documents affected) and is easy to revisit if a
   footer-only pattern turns out to be common.
3. **Local first-page titles were the biggest remaining weakness — addressed in §22.**
   The 31 Stage-1 conflicts were dominated by journal-name-like and typesetting-
   artefact local titles (§21), not by wrong DOIs. Rejecting those candidates
   locally improved titles *and* let good external metadata through: conflicts fell
   to 3 and `EXACT` acceptances rose from 218 to 246, with no change to the
   safeguard itself.
4. **`FOLDER` source addition.** One extra `MetadataSource` value was introduced to
   keep the author's own-work classification honest; if a strict seven-value enum
   is required this needs revisiting.
5. **Body-size heuristic.** First-page body size is estimated from line-size
   frequency; on pages with very few lines it falls back to the median, which is
   adequate but not perfect.
6. **Undeclared test-only dependencies.** `pytest` and `httpx` (FastAPI test
   client) are still installed separately; only runtime dependencies are declared.
7. **MuPDF C-level warnings.** A few corpus PDFs emit `MuPDF error: … color
   space` messages on stderr during reading; they are harmless and do not affect
   extraction results. They are kept out of the report artifacts.
8. **Cross-`data` folder heuristics.** Corpus-role detection keys off folder names
   below a `data` root; a differently-organised corpus would fall back to `OTHER`
   and be excluded from enrichment (a safe default).
9. **External values are merged verbatim** — *fixed in §25.1.* Crossref/OpenAlex text can
   carry provider-side markup or encoding artefacts (JATS `<scp>`/`<jats:p>`, HTML
   entities, double-encoded UTF-8). A deterministic external-only hygiene step now removes
   them before a value becomes a candidate, while the provider's raw text stays inspectable.
10. **Anonymous OpenAlex budget.** 17 OpenAlex queries failed with HTTP 429 during the
    corpus run (deterministic per query, unaffected by pacing/retries). **Corrected in
    §25.3:** this is an exhausted *shared* free daily IP budget ("$0 remaining; resets at
    midnight UTC"), not a polite-pool issue — OpenAlex requires an `api_key` in that state,
    so a configured `mailto` would not restore it. Crossref is always queried first, so no
    identity was fabricated; 76 STRONG remains a lower bound.

## 20. Post-review correction (this revision)

Applied before any corpus-wide enrichment, in response to review:

* **New explicit status `CONFLICT`** in `ExternalMatchStatus`, distinct from
  `AMBIGUOUS` (search ambiguity) — this is *identifier vs. document-identity*
  disagreement.
* **Conflict safeguard** in `MetadataEnricher`: a DOI-exact record is compared
  against the strong local title; below `CONFLICT_TITLE_SIMILARITY = 0.60` the
  record is withheld (all fields, not just the title) and a
  `MetadataConflict{field, local_value, external_value, external_source,
  similarity, route}` is appended. Both candidates are preserved and the CLI
  prints the conflict.
* **Reference-list DOIs are no longer adopted.** Trailing-page/hidden-margin DOI
  scanning was removed; `PdfReadResult.tail_pages`/`tail_text` were dropped and the
  PDF DOI order is now first page → PDF metadata → filename. Corpus DOI count went
  303 → 265, removing exactly the 38 trailing-page-sourced DOIs.
* **Regression tests added** (7): DOI + matching local title → accepted; DOI +
  strongly conflicting title → `CONFLICT`, no override, both candidates preserved;
  DOI + minor subtitle/punctuation differences → accepted; missing local title +
  valid exact DOI → accepted; filename-derived weak title does not raise a
  conflict; a reference-list DOI is not adopted; the CLI prints conflicts.
* **Test-count statement reconciled** (§4): the correct figures are `240 = 77 new +
  163 pre-existing`; the earlier `65 new / 160 old` came from `-k metadata`
  matching three pre-existing test names. No test was changed to fix the
  arithmetic.

## 21. Stage 1 — DOI-only corpus enrichment (executed)

Run with `python -m app.metadata.doi_stage data --out docs/rag006-doi-stage.txt`.
Full artifact: `docs/rag006-doi-stage.txt`. **No title-based search was issued**
(`title-based lookups performed: 0`), trailing-page/reference-list DOI extraction
stayed disabled, and no source file was modified. Documents were deduplicated by
SHA-256 (the scanner yields one logical document per hash; the enricher also
memoizes by hash).

```
unique documents examined:              466
documents with local DOI:               250
eligible for external DOI lookup:       250
  skipped (corpus role not enrichable):   0
  enrichable without a DOI (stage 2):   193
Crossref exact matches:                 249
OpenAlex DOI fallback matches:            0
EXACT accepted:                         218
CONFLICT:                                31
NO_MATCH:                                 1
title-based lookups performed:            0
lookup/network failures:                  0
cached responses used:                  248
API requests made:                        3   (api.crossref.org 2, api.openalex.org 1)
```

`249 = 218 accepted + 31 conflicts`, and the single `NO_MATCH` is a documented
placeholder DOI (`10.3390/xxxxx` from an MDPI template) that correctly does not
resolve. Only 3 API requests were needed because the JSON cache already held the
rest from the earlier sample runs.

Metadata coverage before → after (documents with the field populated):

| field | before | after | delta |
|-------|--------|-------|-------|
| title | 423 | 430 | +7 |
| authors | 408 | 428 | +20 |
| year | 458 | 458 | +0 |
| DOI | 250 | 250 | +0 |
| **venue** | 0 | **209** | **+209** |
| publisher | 0 | 218 | +218 |
| **publication type** | 70 | **284** | **+214** |
| URL | 250 | 250 | +0 |
| abstract | 0 | 152 | +152 |

Venue and publication type — the weakest local fields — improved by +209 and
+214 respectively, as intended. `year` did not move because the 250 DOI-bearing
documents already had a year locally.

### All 31 conflicts (external record withheld, both candidates preserved)

| filename | DOI | local title | external title | sim | src |
|----------|-----|-------------|----------------|-----|-----|
| Adaptive_Learning_Using_Artificial_Intel.pdf | 10.3390/educsci13121216 | education sciences | Adaptive Learning Using Artificial Intelligence in e-Learning: A Literature Review | 0.26 | CROSSREF |
| Course_signals_at_Purdue_Using_learning_analytics_.pdf | 10.1145/2330601.2330666 | Proceedings Template - WORD | Course signals at Purdue | 0.45 | CROSSREF |
| Developing a model for AI Across the curriculum _ ….pdf | 10.1016/j.caeai.2023.100127 | Computers and Education: Artificial Intelligence | Developing a model for AI Across the curriculum: … | 0.32 | CROSSREF |
| Evaluating artificial intelligence literacy courses ….pdf | 10.1016/j.chbr.2022.100223 | Computers in Human Behavior Reports | Evaluating artificial intelligence literacy courses … | 0.18 | CROSSREF |
| Recommender_Systems_for_Teachers_A_Syste.pdf | 10.3390/educsci14070723 | education sciences | Recommender Systems for Teachers: A Systematic Literature Review | 0.22 | CROSSREF |
| artificial-intelligence-in-design-education-….pdf | 10.1017/dsj.2023.28 | DSJ_2300028 1..24 | Artificial intelligence in design education: evaluating ChatGPT … | 0.07 | CROSSREF |
| un-nuevo-enfoque-educativo-….pdf | 10.22533/at.ed.5583332308092 | Human Sciences Research | A NEW EDUCATIONAL APPROACH IN THE HYBRID ADAPTIVE … | 0.17 | CROSSREF |
| 26962-59303-1-PB.pdf | 10.11591/ijai.v14.i3.pp1743-1751 | IAES International Journal of Artificial Intelligence (IJ-AI) | Artificial intelligence multilingual image-to-speech … | 0.37 | CROSSREF |
| Chen-6777-Article Text-34169-4-10-20200919.pdf | 10.18608/jla.2020.72.1 | 1 - 6777 - FINAL - 2020-09-19 | Utilizing Student Time Series Behaviour in Learning Management Systems | 0.09 | CROSSREF |
| David-John-etra22-davidjohn.pdf | 10.1145/3517031 | For Your Eyes Only: Privacy-preserving eye-tracking datasets | 2022 Symposium on Eye Tracking Research and Applications | 0.40 | CROSSREF |
| Ginkel_Ruiz_Mononen_Karaman_de_Keijzer_Sitthiworachart.pdf | 10.1111/jcal.12424 | This is an electronic reprint of the original article. … | The impact of computer-mediated immediate feedback … | 0.20 | CROSSREF |
| Kanwal,-18.pdf | 10.47391/JPMA.AKU-9S-19 | AKU Surgical Suppl--Pages--Final - Adil … | The application of virtual reality and augmented reality … | 0.26 | CROSSREF |
| Radianti -main.pdf | 10.1016/j.compedu.2019.103778 | Computers & Education | A systematic review of immersive virtual reality applications … | 0.21 | CROSSREF |
| Rashidi-10.1177_2374289519873088.pdf | 10.1177/2374289519873088 | APC873088 1..17 | Artificial Intelligence and Machine Learning in Pathology: … | 0.07 | CROSSREF |
| Skulmowski-main.pdf | 10.1016/j.cexr.2023.100023 | Computers & Education: X Reality | Ethical issues of educational virtual reality | 0.57 | CROSSREF |
| guan-2019-JMI-006-031411.pdf | 10.1117/1.JMI.6.3.031411 | JMI-18219SSR 11..11 ++ | Breast cancer detection using synthetic mammograms … | 0.06 | CROSSREF |
| komnini-095204_1_10.0039265.pdf | 10.1121/10.0039265 | Online ¤ | Can speech foundation models effectively identify languages … | 0.09 | CROSSREF |
| kovanovic-7543-Article Text-35849-2-10-20210903.pdf | 10.18608/jla.2021.7543 | 1 - 7543-Article Text-35739-1-11-20210714 - PROOF II | Learning Analytics for Primary and Secondary Schools | 0.22 | CROSSREF |
| radianti-2020.pdf | 10.1016/j.compedu.2019.103778 | Computers & Education | A systematic review of immersive virtual reality applications … | 0.21 | CROSSREF |
| radianti-main.pdf | 10.1016/j.compedu.2019.103778 | Computers & Education | A systematic review of immersive virtual reality applications … | 0.21 | CROSSREF |
| ratiandi.pdf | 10.1016/j.compedu.2019.103778 | Computers & Education | A systematic review of immersive virtual reality applications … | 0.21 | CROSSREF |
| ribiero-1602.04938v3.pdf | 10.1145/2939672.2939778 | "Why Should I Trust You?" Explaining the Predictions of Any Classifier | "Why Should I Trust You?" | 0.49 | CROSSREF |
| 10.21733-ibad.1215907-2818323.pdf | 10.21733/ibad.1215907 | Uluslararası Bilimsel Araştırmalar Dergisi; Journal of the International Scientific Research | The Relationship between Teaching-Learning Conceptions and … | 0.27 | CROSSREF |
| feduc-06-645758.pdf | 10.3389/feduc.2021.645758 | feduc-2021-645758 1..9 | Feedback That Leads to Improvement in Student Essays: … | 0.11 | CROSSREF |
| 7-The Effectiveness of Embodied Pedagogical Agents.pdf | 10.3390/app10051739 | applied sciences | The Effectiveness of Embodied Pedagogical Agents and … | 0.17 | CROSSREF |
| 84-93.pdf | 10.28925/2518-7635.2022.78 | DIGITALISATION OF EDUCATION | TEACHER STUDENTS TRAINING TO IMPLEMENT AR AND VR TECHNOLOGIES | 0.17 | CROSSREF |
| A systematic review of immersive virtual reality applications ….pdf | 10.1016/j.compedu.2019.103778 | Computers & Education | A systematic review of immersive virtual reality applications … | 0.21 | CROSSREF |
| CHI20_WalkingVibe_preprint.pdf | 10.1145/3313831.3376847 | SIGCHI Conference Proceedings Format | WalkingVibe: Reducing Virtual Reality Sickness … | 0.14 | CROSSREF |
| Impact_of_VR_Application_in_an_Academic.pdf | 10.3390/app13084748 | applied sciences | Impact of VR Application in an Academic Context | 0.29 | CROSSREF |
| Towards_Integrating_Virtual_Reality_into.pdf | 10.3390/educsci13050477 | education sciences | Towards Integrating Virtual Reality into Medical Curricula: … | 0.22 | CROSSREF |
| frvir-03-914392.pdf | 10.3389/frvir.2022.914392 | frvir-2022-914392 1..16 | A Separate Reality: An Update on Place Illusion and Plausibility | 0.10 | CROSSREF |

**Interpretation (important).** In ~29 of the 31 cases the disagreement is *not* a
wrong DOI — the DOI matches the identifier printed in the PDF. The **local title**
is the unreliable side: the first-page title extractor picked a running
journal/venue header or a typesetting artefact (`education sciences`,
`Computers & Education`, `applied sciences`, `Proceedings Template - WORD`,
`APIC873088 1..17`, `feduc-2021-645758 1..9`, …). Two are mixed: `David-John` has
the paper's own title locally while the DOI is the *proceedings volume* DOI (so
Crossref's "title" is the venue), and `ribiero` has a longer, more complete local
title than Crossref returns. Under the approved rule the safeguard withholds in all
31 cases — conservative and safe, but the practical cost is that 31 documents keep
a poor local title and do not receive venue/publisher/type/abstract. The single
actionable improvement is a **local** one: reject journal-name-like and
typesetting-artefact first-page title candidates. That is intentionally *not* done
here (the review asked not to invent further heuristics) and is recommended as the
first item of Stage 2.

### DOI-exact accepted with no strong local title (no cross-check possible) — 7

These could not receive the title-conflict cross-check and are reported separately,
as instructed:

```
1726-JASR202414(3)277-288Nigeria.pdf         10.55493/5003.v14i3.5073
3126-Article Text-14139-1-10-20240405.pdf    10.46799/syntax-idea.v6i3.3126
Personalised_Presentation_of_Mathematics.pdf 10.5334/uproc.116
The_Transition_From_Intelligent_to_Affec.pdf 10.1109/ACCESS.2020.3036990
jrspelt738003.pdf                            10.54850/jrspelt.7.38.003
whitton_langan_the.pdf                       10.1080/13562517.2018.1541885
Maister_et_al_TICS_authors_version_Figures.pdf 10.1016/j.tics.2014.11.001
```

### DOI-bearing documents that remain unresolved — 32

31 `CONFLICT` (listed above) plus 1 `NO_MATCH`
(`Creative_Use_of_OpenAI_in_Education_Case.pdf`, DOI `10.3390/xxxxx`, a template
placeholder). Full list in `docs/rag006-doi-stage.txt`.

### Stage-1 corrections made during this run

1. **DOCX/Markdown are no longer DOI sources.** Verified root cause: the corpus
   `.docx` drafts have 17–110 DOIs in their bibliographies (e.g.
   `second-paper-draft-final.docx` → the first DOI found is a cited 2020 IEEE
   Access paper). Before the fix, 5 `my_work` drafts were enriched from a *cited*
   DOI and mis-rendered as published papers. After the fix they are
   `AUTHOR_MANUSCRIPT` with no DOI and make no external request — exactly the
   §20 requirement.
2. **DOI character class restricted to printable ASCII**, so a trailing U+FFFD from
   a bad PDF encoding is no longer swallowed into the DOI (turned the `Sharara.pdf`
   `NO_MATCH` into a clean `10.1109/cloudnet55617.2022.9978863`).
3. **Stage runner added** (`doi_stage.py`) with `title_lookup_enabled=False`, plus
   HTTP/cache instrumentation (`request_count`, `network_error_count`, cache
   `hits`/`misses`) to produce the metrics above.

Stage 2 (title-based enrichment of the DOI-less enrichable documents) was **not**
started at that revision — it became Stage 2A (§23) and Stage 2B (§24) later; the
"193" above is the Stage-1-era estimate, corrected to 169 eligible documents in §23.2.
RAG-007 is **not** started.

## 22. Post-review correction 2 — local title-candidate rejection (this revision)

Review finding: of the 31 Stage-1 `CONFLICT`s, ~29 had a *correct* DOI and a *wrong*
local title. The defect was in the local first-page title extractor, not in the DOI
safeguard. Nothing in this section weakens the safeguard, trusts Crossref because a
DOI exists, or uses an LLM.

### 22.1 Deterministic rejection classes (`app/metadata/titles.py`)

| Class | Signals used (all structural / format-based) |
|-------|----------------------------------------------|
| `JOURNAL_OR_VENUE_HEADER` | short line near the top *plus* a more title-like candidate below (longer, or lower with strictly more words) *plus* venue context: ISSN, volume/issue, "journal homepage", "contents lists available", publisher branding, `www.`, or a publication label word (`journal`, `proceedings`, `transactions`, `dergisi`, `revista`, …) in the line itself. Vocabulary alone never rejects a title. |
| `TEMPLATE_LABEL` | `Proceedings Template - WORD`, `SIGCHI Conference Proceedings Format`, `Pages--Final`, `Template`, `FINAL` |
| `PRODUCTION_IDENTIFIER` | `<CODE> 1..24`, `<CODE> 1..17`, `1 - 7543-Article Text-35739-…- PROOF II`, `Article Text` (only with a digit nearby), `DSJ_2300028`, and any ≤6-word string that is code-dominated (<50 % letters) |
| `PROOF_IDENTIFIER` | uppercase `PROOF` (prose "proof" is not rejected) |
| `REPRINT_NOTICE` | `This is an electronic reprint of the original article…`-style publisher notices |
| `RUNNING_HEADER` | page-edge band (top/bottom 6 %) with a trailing page number |
| extraction garbage | unmappable/private-use glyph runs (e.g. `Heterogeneous Graph A\ue03cention Network`) |

After a candidate is rejected the extractor **continues to the next conservative
candidate**; it only returns `None` when nothing survives. Journal names are *never*
matched from a list — the signals are layout, context and generic publication labels.

Two latent false-positive bugs found while doing this were fixed as well:

* `looks_like_boilerplate_line` used substring matching, so a real title containing
  "issues" matched "issue" — replaced by a word-boundary `_VENUE_HEADER_RE`.
* `looks_like_affiliation` used substring markers including `"center"`, which
  classified the legitimate title "…A Single **Center** Student Survey" as an
  affiliation and blocked masthead rejection — replaced by a word-boundary regex
  where *school/college/center/centre/academy* additionally require "of"/"for".

### 22.2 Embedded "title" metadata is no longer trusted blindly

Journal templates routinely stamp the journal name into the PDF `title` field. The
embedded title is now discarded when it reads as a venue masthead (publication label
inside the string, plus an accepted first-page article title that does not share its
wording and is not shorter) — and it is still used when page one yields no title.
Example: `26962-59303-1-PB.pdf` had `PDF_METADATA = "IAES International Journal of
Artificial Intelligence (IJ-AI)"` and now resolves the real title from page one.

### 22.3 Author-block consequences (same helpers, so fixed together)

`titles.py` and `authors.py` share the author-line heuristics, so the correction had
to cover both sides. Local *title* lines are no longer emitted as authors, while real
authors are preserved or recovered:

* lines that are part of the detected title are skipped by the first-page author scan
  (previously 14 documents reported title fragments such as
  "Opportunities and Ethical Challenges" or "RELIABILITY" as authors);
* an all-caps multi-word **name** line below the title is accepted
  (`MUHAMMAD ASIF HASAN`) while an all-caps multi-word **heading** is not;
* repository/cover-sheet boilerplate ("Document Version", "Peer reviewed version",
  "To cite this version", "OPEN ACCESS", "Published in", "Accepted author
  manuscript", …) and lowercase-prose fragments are rejected as names;
* leading `and`/`by` is stripped from a name instead of rejecting the name;
* parenthetical annotations (`Predescu (Burciu)`, `Bogosian(B)`) are ignored when
  validating a name but kept in the stored value;
* a leading author line at the same size as the title is no longer merged into the
  title (`Artificial_Intelligence_and_Teachers_New.pdf`,
  `Impact_of_VR_Application_in_an_Academic.pdf`,
  `Performance_Driven_VR_Learning_for_Robot.pdf` now yield both the correct title and
  the correct author list).

### 22.4 Real-corpus validation (offline, full corpus)

Pre-correction numbers are the ones recorded in §15/§21 above (423 titles, 408
authors). Per-document before/after comparison uses an in-process reconstruction that
re-enables the previous rules; it reproduces the recorded baseline to within two
documents (425 titles, 406 authors), so the deltas below are read from that
reconstruction.

```
                       pre-correction   post-correction
title available             423                421
authors detected            408                391
DOI detected                250                250
year detected               458                458
publication type known       70                 70
extraction errors             0                  0
```

Title extraction changed for **38 documents**: 34 titles were corrected, 4 were
rejected outright, none was newly found.

* 4 rejections, all defensible: an electronic-reprint notice, the production
  identifier `JMI-18219SSR 11..11 ++`, and two titles containing unmappable glyph
  codes (`Wang-3308558.3313562.pdf`, `shen-3746059.3747803.pdf`). Under the task rule
  *missing is acceptable, wrong is not*, a corrupted string is not kept as a title.
* 34 corrections: journal/venue mastheads, proceedings templates, production codes
  and running headers replaced by the real article title, plus the three merged
  author/title blocks of §22.3. Full list:

| File | Before | After |
|------|--------|-------|
| `10.21733-ibad.1215907-2818323.pdf` | Uluslararası Bilimsel Araştırmalar Dergisi; Journal of the International Scientific Research | The Relationship between Teaching-Learning Conceptions and Sense of Self-Efficacy of Pre-Service EFL Teachers … |
| `26962-59303-1-PB.pdf` | IAES International Journal of Artificial Intelligence (IJ-AI) | Artificial intelligence multilingual image-to-speech for accessibility and text recognition |
| `3396-Article Text-14836-1-10-20240121.pdf` | Journal of Computer Networks, Architecture and High Performance Computing | Exploration of Artificial Intelligence (AI) Application in Higher Education: A Research Study |
| `689-Article Text-3059-3-10-20230710.pdf` | Journal of Applied Learning & Teaching | ChatGPT: Bullshit spewer or the end of traditional assessments in higher education |
| `7-The Effectiveness of Embodied Pedagogical Agents.pdf` | applied sciences | The Effectiveness of Embodied Pedagogical Agents and Their Impact on Students Learning |
| `84-93.pdf` | DIGITALISATION OF EDUCATION | TEACHER STUDENTS TRAINING TO IMPLEMENT AR AND VR TECHNOLOGIES IN FOREIGN LANGUAGE … |
| `A systematic review of immersive virtual reality applications … .pdf` | Computers & Education | A systematic review of immersive virtual reality applications for higher education |
| `Adaptive_Learning_Using_Artificial_Intel.pdf` | education sciences | Adaptive Learning Using Artificial Intelligence in e-Learning: A Literature Review |
| `Artificial_Intelligence_and_Teachers_New.pdf` | Catherine Adams, Patti Pente, … & Geoffrey Rockwell | Artificial Intelligence and Teachers' New Ethical Obligations |
| `CHI20_WalkingVibe_preprint.pdf` | SIGCHI Conference Proceedings Format | WalkingVibe: Reducing Virtual Reality Sickness and Improving Realism while Walking |
| `Chen-6777-Article Text-34169-4-10-20200919.pdf` | 1 - 6777 - FINAL - 2020-09-19 | Utilizing Student Time Series Behaviour in Learning Management Systems |
| `Course_signals_at_Purdue_Using_learning_analytics_.pdf` | Proceedings Template - WORD | Course signals at Purdue: Using learning analytics to increase student success |
| `Developing a model for AI Across the curriculum ….pdf` | Computers and Education: Artificial Intelligence | Developing a model for AI Across the curriculum: Transforming the higher education landscape |
| `Evaluating artificial intelligence literacy courses ….pdf` | Computers in Human Behavior Reports | Evaluating artificial intelligence literacy courses for fostering conceptual learning |
| `Impact_of_VR_Application_in_an_Academic.pdf` | applied sciences | Impact of VR Application in an Academic Context |
| `Kanwal,-18.pdf` | AKU Surgical Suppl--Pages--Final - Adil 26-03-2024 … | The application of virtual reality and augmented reality in dentistry |
| `Radianti -main.pdf` | Computers & Education | A systematic review of immersive virtual reality applications for higher education |
| `Rashidi-10.1177_2374289519873088.pdf` | APC873088 1..17 | Artificial Intelligence and Machine Learning in Pathology: The Present Landscape |
| `Recommender_Systems_for_Teachers_A_Syste.pdf` | education sciences | Recommender Systems for Teachers: A Systematic Literature Review |
| `Skulmowski-main.pdf` | Computers & Education: X Reality | Ethical issues of educational virtual reality |
| `The_Flipped_Classroom_Strategy_What_Is_I.pdf` | The Flipped Classroom Strategy | The Flipped Classroom Strategy \| What Is it and How Can it Best be Used? |
| `Towards_Integrating_Virtual_Reality_into.pdf` | education sciences | Towards Integrating Virtual Reality into Medical Curricula: A Single Center Student Survey |
| `artificial-intelligence-in-design-education-evaluating-chatgpt-as-a-virtual-colleague-for-post-graduate-course-development.pdf` | DSJ_2300028 1..24 | Artificial intelligence in design education: evaluating ChatGPT as a virtual colleague |
| `ethical-use-of-artificial-intelligence-and-new-technologies-in-education-50 (1).pdf` | Journal of Artificial Intelligence, Machine Learning and Data Science | Ethical Use of Artificial Intelligence and New Technologies in Education |
| `feduc-06-645758.pdf` | feduc-2021-645758 1..9 | Feedback That Leads to Improvement in Student Essays |
| `frvir-03-914392.pdf` | frvir-2022-914392 1..16 | A Separate Reality: An Update on Place Illusion and Plausibility in Virtual Reality |
| `komnini-095204_1_10.0039265.pdf` | Online | Can speech foundation models effectively identify languages in low-resource settings |
| `kovanovic-7543-Article Text-35849-2-10-20210903.pdf` | 1 - 7543-Article Text-35739-1-11-20210714 - PROOF II | Learning Analytics for Primary and Secondary Schools |
| `ncowie,+1487-ASCILITE+Conference+Full+287-297.pdf` | ASCILITE 2024 | GenAI Teachers: Constructivist Learning Design and Value Propositions |
| `radianti-2020.pdf` | Computers & Education | A systematic review of immersive virtual reality applications for higher education |
| `radianti-main.pdf` | Computers & Education | A systematic review of immersive virtual reality applications for higher education |
| `ratiandi.pdf` | Computers & Education | A systematic review of immersive virtual reality applications for higher education |
| `rudolf-771-Article Text-3333-4-10-20230616 (3).pdf` | Journal of Applied Learning & Teaching | War of the chatbots: Bard, Bing Chat, ChatGPT, Ernie and beyond |
| `un-nuevo-enfoque-educativo-en-el-modelo-de-aprendizaje-adaptativo-hibrido-en-la-educacion-media-superior.pdf` | Human Sciences Research | A NEW EDUCATIONAL APPROACH IN THE HYBRID ADAPTIVE LEARNING MODEL IN HIGHER EDUCATION |

Author coverage moved 408 → 391. That is a *precision* change, not a regression:
19 documents that previously reported title fragments or repository boilerplate as
"authors" now report none (e.g. `feduc-06-645758.pdf` reported "Feedback is Most
Powerful"; `Rashidi-…pdf` reported "Review Article"; `Gentry-thesis.pdf` reported "A
DISSERTATION"), 4 documents gained real author lists they did not have before
(`ALEKS_McKeown (1).pdf`, `The_Transition_From_Intelligent_to_Affec.pdf`,
`Antonelli-education-13-00528.pdf`, `Performance_Driven_VR_Learning_for_Robot.pdf`),
and 95 documents had their author list corrected (boilerplate removed, real names
recovered). Remaining known weaknesses are listed in §22.10.

A 35th title change was caught and reverted during validation: the new masthead rule
initially rejected the title `Differential Privacy` in favour of the section heading
`1 Introduction`, because a comparable-size line *below* the candidate counted as
"more title-like". The rule now also requires such a line to carry strictly more
words than the candidate; that restores `Differential Privacy` and leaves the rest of
the corpus unchanged (421 titles before and after).

### 22.5 DOI Stage 1 rerun (existing cache, DOI-only)

```
                       before (Stage 1)   after (this correction)
Crossref exact matches        249                 249
EXACT accepted                218                 246
CONFLICT                       31                   3
NO_MATCH                        1                   1
lookup/network failures         0                   0
cached responses used         248                 249
API requests made               3                   2
eligible for DOI lookup       250                 250
title-based lookups             0                   0
```

Metadata coverage around the DOI run (documents with the field populated):

| field | before | after | delta |
|-------|--------|-------|-------|
| title | 421 | 431 | +10 |
| authors | 391 | 422 | +31 |
| year | 458 | 458 | +0 |
| DOI | 250 | 250 | +0 |
| **venue** | 0 | **237** | **+237** |
| publisher | 0 | 246 | +246 |
| **publication type** | 70 | **312** | **+242** |
| URL | 250 | 250 | +0 |
| abstract | 0 | 168 | +168 |

`246 + 3 + 1 = 250`: every DOI-bearing document is now accounted for as accepted,
conflicted or unresolved, with no failures.

### 22.6 Remaining conflicts — all 3, with assessments (not auto-resolved)

| # | file | DOI | local title | external title | sim. | assessment |
|---|------|-----|-------------|----------------|------|------------|
| 1 | `Course_signals_at_Purdue_Using_learning_analytics_.pdf` | `10.1145/2330601.2330666` | Course signals at Purdue: Using learning analytics to increase student success | Course signals at Purdue | 0.4752 | **external record unusually coarse/short** — local title is the full article title |
| 2 | `ribiero-1602.04938v3.pdf` | `10.1145/2939672.2939778` | "Why Should I Trust You?" Explaining the Predictions of Any Classifier | "Why Should I Trust You?" | 0.4944 | **external record unusually coarse/short** — Crossref stores a shortened title; the local fuller title is legitimate and must not be replaced |
| 3 | `David-John-etra22-davidjohn.pdf` | `10.1145/3517031` | For Your Eyes Only: Privacy-preserving eye-tracking datasets | 2022 Symposium on Eye Tracking Research and Applications | 0.4000 | **likely genuine conflict** — the DOI resolves to the *proceedings-level* record, not the paper; neither side is simply wrong |

Both special cases called out in the review (`David-John` = proceedings-level DOI,
`ribiero` = fuller local title than Crossref) are still `CONFLICT` and nothing was
changed to force them to match. Note the conflict count is now 3 because the other 28
cases were *not* conflicts at all — they were bad local titles, now fixed locally. No
assessment in this table was applied automatically.

### 22.7 DOI-exact accepted without a strong local title (no cross-check possible) — 10

```
1726-JASR202414(3)277-288Nigeria.pdf            10.55493/5003.v14i3.5073
3126-Article Text-14139-1-10-20240405.pdf       10.46799/syntax-idea.v6i3.3126
Personalised_Presentation_of_Mathematics.pdf    10.5334/uproc.116
jrspelt738003.pdf                               10.54850/jrspelt.7.38.003
Ginkel_Ruiz_Mononen_Karaman_de_Keijzer_Sitthiworachart.pdf  10.1111/jcal.12424
Wang-3308558.3313562.pdf                        10.1145/3308558.3313562
guan-2019-JMI-006-031411.pdf                    10.1117/1.JMI.6.3.031411
shen-3746059.3747803.pdf                        10.1145/3746059.3747803
whitton_langan_the.pdf                          10.1080/13562517.2018.1541885
Maister_et_al_TICS_authors_version_Figures.pdf  10.1016/j.tics.2014.11.001
```

Three of them (`Wang-3308558`, `guan-2019-JMI`, `shen-3746059`) are in this list
*because* their corrupted-glyph titles were rejected in §22.4 — an accepted trade:
their final title now comes from Crossref instead of from a corrupted local string.
No heuristic was invented to give these documents a cross-check; they are simply
reported separately, as instructed.

### 22.8 DOI-bearing documents that remain unresolved — 4

The three `CONFLICT`s above plus one `NO_MATCH`
(`Creative_Use_of_OpenAI_in_Education_Case.pdf`, DOI `10.3390/xxxxx`, an MDPI
template placeholder).

### 22.9 Test result

```
254 passed, 1 warning
```

`254 = 91 RAG-006 metadata tests + 163 pre-existing (RAG-001…005, unchanged)`. Per
file: `test_metadata_titles.py` 23, `test_metadata_corpus_roles.py` 17,
`test_metadata_external.py` 15, `test_metadata_local_extractors.py` 9,
`test_metadata_doi.py` 8, `test_metadata_models.py` 6, `test_metadata_cli.py` 5,
`test_metadata_pdf.py` 4, `test_metadata_doi_stage.py` 2, `test_metadata_year.py` 2.
The new tests cover the 12 requested title cases (masthead above a real title,
`education sciences`, `DSJ_2300028 1..24`, `APC873088 1..17`, `feduc-2021-645758 1..9`,
`Proceedings Template - WORD`, `SIGCHI Conference Proceedings Format`, reprint
notice, next-candidate selection, legitimate short title, journal-like vocabulary,
offline determinism) plus the embedded-venue-title and author-block cases of §22.2–3.
No test was changed to make a number fit.

### 22.10 Known remaining weaknesses (reported, deliberately not "fixed" now)

* Publisher/repository cover artefacts can still be picked up as authors in a few
  documents (e.g. `IJ-AI`, `IOR PRESS`, `python-docx`, `Unknown`, `Form`). Some of
  these come from the *filename* heuristic or from embedded metadata that is itself
  wrong; they are pre-existing behaviour, listed here rather than silently patched.
* A few lower-cased surname-only tokens from filenames (`chen`, `jmir`, `mededu`)
  remain — same heuristic, same reason.
* Titles that consist of a single word (`plagiarism.pdf`, `Proofreading.pdf`) are
  still rejected by the pre-existing "≥2 words" rule and stay missing.
* 45 documents still have no local title; most are scanned/ERIC-style files whose
  first page carries no typeset title.

Stage 2 (title-based enrichment of the DOI-less enrichable documents) was **not**
started at that revision; it became Stage 2A (§23, sample) and Stage 2B (§24, corpus).
RAG-007 is **not** started.

## 23. Stage 2A — title-based enrichment validation sample (this revision)

Scope: **validation sample only**. No corpus-wide title enrichment was run, no
source file was modified, no threshold was lowered, and the three remaining DOI
`CONFLICT`s from §22.6 were left exactly as they are.

### 23.1 Eligible population and sample

Eligibility (implemented as `is_title_sample_eligible`):

```
externally enrichable (corpus role)            rules / supervision / future_chapters /
   AND no accepted local DOI                   summaries / guidance never enriched;
   AND strong local title                      my_work drafts have no DOI -> excluded
```

A *strong local title* is `PDF_METADATA`, `DOCX_PROPERTIES` or `DOCUMENT_TEXT` with
confidence `MEDIUM`/`HIGH`. As an additional safeguard the enricher itself now refuses
to start a title search without such a title (`_title_search_source`), so a
filename-only token can never initiate an external search from any caller.

```
unique documents examined:                     466
documents with a local DOI (Stage 1):          250
documents without a local DOI:                 216
eligible population (DOI-less + strong title): 169   (of the 216 DOI-less documents,
                                                     47 have no strong local title)
sample size:                                    30
title searches attempted:                       30
```

The sample is deterministic (sorted buckets, round-robin) and stratified by title
source × title-length band × year presence × author presence, taking oldest/newest
alternately inside each bucket so the sample spans publication years:

```
title source            DOCUMENT_TEXT 16, PDF_METADATA 14
title length            short <=6 words 11, medium 7-12 12, long >=13 7
year                    year 26, no year 4   (range 1977-2026)
authors                 authors 18, no authors 12
local publication type  UNKNOWN 25, PREPRINT 5
with arXiv id           6
```

Journal/conference/report types were **not** selectable locally — the online type was
unknown for 25 of the 30 (that is precisely the gap this stage exercises); the 12
accepted records supplied 3 `JOURNAL_ARTICLE`, 5 `CONFERENCE_PAPER` and 4 `PREPRINT`
(the remaining 4 local `PREPRINT`s were confirmed, not added). The sample deliberately
keeps hard cases: template headings, HTML markup and cover-page titles, `View of …`
artefacts, all-caps titles, 2-word titles, and documents with neither a usable year nor
a usable author.

### 23.2 Route and safeguards (unchanged rules)

```
local title -> Crossref bibliographic title search -> OpenAlex title search (fallback)
```

No DOI lookup is performed (the population has no DOI). Only `STRONG` is merged;
thresholds are unchanged (`STRONG 0.92`, `MIN 0.75`, ambiguity margin `0.03`, year
`±1`, first-author-family agreement). `AMBIGUOUS` and `NO_MATCH` leave the document
untouched.

A DOI obtained from a title match is recorded as:

```
source     = CROSSREF | OPENALEX
confidence = MEDIUM
evidence   = strong title-based external match (<provider> title search); external DOI
```

so it is never presented as a DOI observed in the document: a DOI *observed* in the file
carries `DOCUMENT_TEXT` / `PDF_METADATA` / `FILENAME` as its source, whereas this is a
*bibliographic identification* of the document. The stage report prints a `DOI origin`
line for every row (`observed locally in the document` vs `title-based external match
(not observed in the document)`), and in this sample all 12 merged DOIs are of the
latter kind (`DOI` coverage inside the sample went 0 -> 12, all from an external
source).

### 23.3 Aggregate results

```
sample size                              30
title searches attempted                 30
STRONG                                   12
AMBIGUOUS                                 8
NO_MATCH                                 10
API/network failures                      0   (network 0, HTTP 0, retries 0)
accepted (merged)                        12
accepted matches manually confirmed      12
suspicious accepted matches               0
false-positive accepted matches           0
cached responses used                     7
cache misses                             43
API requests made                        45   (api.crossref.org 28, api.openalex.org 17)
```

Metadata coverage within the sample:

| field | before | after | delta |
|-------|--------|-------|-------|
| authors | 18 | 22 | +4 |
| year | 26 | 28 | +2 |
| DOI | 0 | 12 | +12 |
| venue | 0 | 10 | +10 |
| publisher | 0 | 12 | +12 |
| publication type | 5 | 13 | +8 |
| abstract | 0 | 6 | +6 |

Coverage on `title` was already 30/30 locally (an eligibility requirement), and the
resolved title was **replaced** in only 3 cases (the external form was the cleaner one:
`View of …` prefix, `MAPoRL2` artefact, all-caps local form); in the other 9 the two
titles were identical apart from case/typography.

### 23.4 Per-document results (30 rows)

`source` is the local title source (`TEXT` = `DOCUMENT_TEXT`, `PDF` = `PDF_METADATA`);
the full field set (local/external year, local/external first author, similarity,
resolved title, venue, publisher, abstract presence) is in the artifact
`docs/rag006-stage2a-sample.txt`.

| # | file | local title | src | local yr | local 1st author | provider | external title | sim | ext yr | ext 1st author | status | acc | external DOI | ext type |
|---|------|-------------|-----|----------|------------------|----------|----------------|-----|--------|----------------|--------|-----|--------------|----------|
| 1 | `jmir-2023-1-e39989.pdf` | The Effects of Immersive Virtual Reality Applic... | TEXT | 2000 | jmir | CROSSREF | The Effects of Immersive Virtual Realit... | 1.000 | 2023 | Justina Yat Wa... | AMBIGUOUS | no | 10.2196/39989 | - |
| 2 | `Situated Cognition and the culture of...` | Situated Cognition and the Culture of Learning | TEXT | 1989 | jsb | CROSSREF | Situated Cognition and the Culture of L... | 1.000 | 1988 | John S. Brown | AMBIGUOUS | no | 10.21236/ada204690 | - |
| 3 | `Zhou-ICRA2017.pdf` | Two-Eye Model-Based Gaze Estimation from A Kine... | TEXT | 2014 | - | CROSSREF | Two-eye model-based gaze estimation fro... | 1.000 | 2017 | Xiaolong Zhou | AMBIGUOUS | no | 10.1109/icra.2017.7989194 | - |
| 4 | `dwork-11787006_1.pdf` | Diﬀerential Privacy | TEXT | - | Cynthia Dwork | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 5 | `KenHubbellRoundtableWhitePaper.pdf` | Design evidence based learning experiences | TEXT | - | - | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 6 | `dwork.pdf` | Diﬀerential Privacy | TEXT | 1977 | Cynthia Dwork | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 7 | `Leutenegger-p37.pdf` | Keyframe-Based Visual-Inertial SLAM Using Nonli... | TEXT | 2013 | - | CROSSREF | Keyframe-Based Visual-Inertial SLAM usi... | 1.000 | 2013 | Stefan Leutene... | STRONG | yes | 10.15607/rss.2013.ix.037 | CONFERENCE_PAPER |
| 8 | `Mayer.pdf` | A Large Dataset to Train Convolutional Networks... | PDF | 2015 | Nikolaus May... | CROSSREF | A Large Dataset to Train Convolutional ... | 1.000 | 2016 | Nikolaus Mayer | STRONG | yes | 10.1109/cvpr.2016.438 | CONFERENCE_PAPER |
| 9 | `View of Challenges with Gamification ...` | View of Challenges with Gamification in Higher ... | PDF | 2024 | - | CROSSREF | Challenges with Gamification in Higher ... | 0.967 | 2024 | Kevin Fuchs | STRONG | yes | 10.47852/bonviewijce32021... | JOURNAL_ARTICLE |
| 10 | `2507.01597v1.pdf` | T3DM: Test-Time Training-Guided Distribution Sh... | PDF | - | Yuehang Si | OPENALEX | T3DM: Test-Time Training-Guided Distrib... | 1.000 | 2025 | Yuehang Si | STRONG | yes | 10.48550/arxiv.2507.01597 | PREPRINT |
| 11 | `Bender1-2014-CGF-Position_Based_Surve...` | A Survey on Position-Based Simulation Methods i... | PDF | 2014 | Jan Bender | CROSSREF | A Survey on Position‐Based Simulation M... | 1.000 | 2014 | Jan Bender | STRONG | yes | 10.1111/cgf.12346 | JOURNAL_ARTICLE |
| 12 | `Learning-Styles-and-Learning-Spaces.....` | Learning Styles and Learning Spaces: Enhancing ... | PDF | 2016 | - | CROSSREF | Learning Styles and Learning Spaces: En... | 1.000 | 2005 | Alice Y. Kolb | AMBIGUOUS | no | 10.5465/amle.2005.17268566 | - |
| 13 | `watkins-BF00992698.pdf` | <Emphasis Type="Italic">Q </Emphasis>-learning | PDF | 1992 | CHRISTOPHER ... | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 14 | `-John_Biggs_and_Catherine_Tang-_Teach...` | Teaching for Quality Learning at University | PDF | 2012 | - | CROSSREF | Book review: Teaching for quality learn... | 0.878 | 2016 | Christina Davi... | AMBIGUOUS | no | 10.18538/lthe.v13.n2.266 | - |
| 15 | `2504.03966v1.pdf` | Bridging LMS and Generative AI: Dynamic Course ... | TEXT | 2025 | Kovan Mzwri | CROSSREF | Bridging LMS and Generative AI: Dynamic... | 1.000 | 2025 | Kovan Mzwri | STRONG | yes | 10.21203/rs.3.rs-6256444/... | PREPRINT |
| 16 | `vr-rag-2504.08256v2.pdf` | RAG-VR: Leveraging Retrieval-Augmented Generati... | TEXT | 2025 | Shiyi Ding | CROSSREF | RAG-VR: Leveraging Retrieval-Augmented ... | 1.000 | 2025 | Shiyi Ding | STRONG | yes | 10.1109/vrw66409.2025.000... | CONFERENCE_PAPER |
| 17 | `Du-2025.acl-long.1459.pdf` | MAPoRL2: Multi-Agent Post-Co-Training for Colla... | TEXT | 2025 | - | CROSSREF | MAPoRL: Multi-Agent Post-Co-Training fo... | 0.995 | 2025 | Chanwoo Park | STRONG | yes | 10.18653/v1/2025.acl-long... | CONFERENCE_PAPER |
| 18 | `Bujanca_FullFusbujanca-ion_A_Framewor...` | The University of Manchester Research | TEXT | 2026 | General righ... | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 19 | `Plagiarism_detection_in_Programming_c...` | Plagiarism detection in Programming coursework. | TEXT | 2025 | - | CROSSREF | Plagiarism detection in Programming cou... | 1.000 | 2024 | Constantine Mu... | STRONG | yes | 10.18260/1-2--38399 | CONFERENCE_PAPER |
| 20 | `thermal-2509.18929v1.pdf` | Native Mixed Reality Compositing on Meta Quest ... | PDF | 2025 | Muhammad Kai... | OPENALEX | Native Mixed Reality Compositing on Met... | 1.000 | 2025 | Muhammad Kaif ... | STRONG | yes | 10.48550/arxiv.2509.18929 | PREPRINT |
| 21 | `Lin-View of Design and Application of...` | View of Design and Application of Intelligent A... | PDF | 2026 | - | CROSSREF | Design and Application of Intelligent A... | 0.958 | 2024 | Yushi Lin | AMBIGUOUS | no | 10.52783/jes.3114 | - |
| 22 | `ssrn-5016008.pdf` | The Impact of Virtual Reality on Student Engage... | PDF | - | ssrn | CROSSREF | The Impact of Virtual Reality on Studen... | 1.000 | 2025 | Nandita Suhag | STRONG | yes | 10.2139/ssrn.5016008 | PREPRINT |
| 23 | `valverde-2510.14270v1.pdf` | GauSSmart: Enhanced 3D Reconstruction through 2... | PDF | 2025 | Alexander Va... | - | - | n/a | - | - | NO_MATCH | no | - | PREPRINT |
| 24 | `1777-Article Text-3547-1-10-20230831....` | Novel Mechanism of Action Drug Candidates for T... | PDF | 2023 | - | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 25 | `Manukonda-1820-Article Text-4650-5556...` | Paper Title (use style: paper title) | PDF | 2024 | ashock | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 26 | `wise-assessment-briefing9.pdf` | A3 fold - briefing 9.pages | PDF | 2015 | - | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 27 | `smithEJ982673.pdf` | PREDICTIVE MODELING TO FORECAST STUDENT OUTCOME... | TEXT | 2012 | RROGERS | CROSSREF | Predictive Modeling to Forecast Student... | 1.000 | 2012 | Vernon C. Smith | STRONG | yes | 10.24059/olj.v16i3.275 | JOURNAL_ARTICLE |
| 28 | `Welch-Motion_Tracking_No_Silver_Bulle...` | Motion Tracking: No Silver Bullet, but a Respec... | TEXT | 2002 | Motion Track... | CROSSREF | Motion tracking: no silver bullet, but ... | 1.000 | 2002 | G. Welch | AMBIGUOUS | no | 10.1109/mcg.2002.1046626 | - |
| 29 | `Discussion_on_the_Cultivation_of_Coll...` | Discussion on the Cultivation of College Studen... | TEXT | 2016 | - | - | - | n/a | - | - | NO_MATCH | no | - | - |
| 30 | `Intelligent_tutoring_systems_D_Sleema...` | Book Reviews | TEXT | 1985 | ARTIFICIAL I... | CROSSREF | Book reviews…Book reviews…Book reviews | 1.000 | 1999 | - | AMBIGUOUS | no | 10.1111/j.2042-3292.1999.... | - |

### 23.5 Manual quality assessment of the 12 accepted matches

Each accepted DOI was re-resolved independently (Crossref, plus the document itself)
and compared with the local title/year/author before classification:

| # | file | accepted DOI | independent check | classification |
|---|------|--------------|-------------------|----------------|
| 7 | `Leutenegger-p37.pdf` | 10.15607/rss.2013.ix.037 | title identical (case only); year 2013 = 2013; RSS 2013, Leutenegger/Furgale | clearly correct |
| 8 | `Mayer.pdf` | 10.1109/cvpr.2016.438 | title identical; same first author; local 2015 = arXiv, record 2016 = CVPR | clearly correct |
| 9 | `View of Challenges…pdf` | 10.47852/bonviewijce32021604 | external title == local minus the `View of` PDF-viewer prefix; Kevin Fuchs, IJCE | clearly correct |
| 10 | `2507.01597v1.pdf` | 10.48550/arxiv.2507.01597 | title identical, same author; arXiv number in the DOI == filename | clearly correct |
| 11 | `Bender1-2014-CGF…pdf` | 10.1111/cgf.12346 | title identical (non-breaking hyphen apart); Jan Bender; CGF 2014 | clearly correct |
| 15 | `2504.03966v1.pdf` | 10.21203/rs.3.rs-6256444/v1 | title identical, same author, Research Square preprint 2025 | clearly correct |
| 16 | `vr-rag-2504.08256v2.pdf` | 10.1109/vrw66409.2025.00034 | title identical, same author, IEEE VRW 2025 | clearly correct |
| 17 | `Du-2025.acl-long.1459.pdf` | 10.18653/v1/2025.acl-long.1459 | ACL 2025 record; DOI suffix == filename anthology id; local `MAPoRL2` is an artefact | clearly correct |
| 19 | `Plagiarism_detection_in_Programming_c…` | 10.18260/1-2--38399 | title identical *including* the trailing period; author Constantine Mukasa; the PDF itself says "Paper ID #35519" / "© ASEE, 2021" | clearly correct |
| 20 | `thermal-2509.18929v1.pdf` | 10.48550/arxiv.2509.18929 | title identical, same author; arXiv number in the DOI == filename | clearly correct |
| 22 | `ssrn-5016008.pdf` | 10.2139/ssrn.5016008 | title identical; Nandita Suhag; SSRN id == filename | clearly correct |
| 27 | `smithEJ982673.pdf` | 10.24059/olj.v16i3.275 | title identical (case only); 2012 = 2012; Smith/Lange/Huston, Online Learning 16(3) | clearly correct |

```
clearly correct   12
probably correct   0
suspicious         0
false positives    0
```

**Zero known false-positive accepted matches**, which was the target. Two accepted
records carry a *year* nuance that does not affect identity (`Mayer.pdf`: local arXiv
2015 vs CVPR 2016; `Plagiarism…`: Crossref `issued` 2024 while the paper is the ASEE
2021 paper) — both are the same work as the document, verified from the document itself.
Two DOIs are DataCite-registered arXiv DOIs (`10.48550/…`) and therefore do not resolve
at Crossref; they came from OpenAlex and were verified by the arXiv number embedded in
the DOI matching the document's own filename.

### 23.6 Withheld matches — why (nothing merged)

`AMBIGUOUS` (8). The reason was derived from the reported fields and, where needed,
from the cached search response:

| # | file | similarity | reason the match was withheld |
|---|------|-----------|-------------------------------|
| 1 | `jmir-2023-1-e39989.pdf` | 1.000 | local year 2000 vs record 2023 (delta 23) |
| 2 | `Situated Cognition…pdf` | 1.000 | **two** records share the title (1988 report `10.21236/ada204690` and 1989 *Educational Researcher* article `10.2307/1176008`) |
| 3 | `Zhou-ICRA2017.pdf` | 1.000 | local year 2014 vs record 2017 |
| 12 | `Learning-Styles…pdf` | 1.000 | local year 2016 vs record 2005 |
| 14 | `-John_Biggs…pdf` | 0.878 | below the 0.92 STRONG threshold **and** the record is a *book review of* the book, not the book |
| 21 | `Lin-View of Design…pdf` | 0.958 | local year 2026 vs record 2024 |
| 28 | `Welch-Motion_Tracking…pdf` | 1.000 | local first author "Motion Tracking Survey" (garbage) vs "Welch" |
| 30 | `Intelligent_tutoring_systems…pdf` | 1.000 | local title is the section heading "Book Reviews"; year 1985 vs 1999 |

Two of these are **saved false positives**: record 14 would have attached a
book-*review* DOI to the book, and record 30 would have attached an unrelated 1999
equine-veterinary "Book reviews…" record to a 1985 AI paper. Both were refused by
rules that already existed — no new default was needed.

`NO_MATCH` (10): `dwork-11787006_1.pdf` and `dwork.pdf` ("Diﬀerential Privacy"),
`KenHubbellRoundtableWhitePaper.pdf`, `watkins-BF00992698.pdf`,
`Bujanca_FullFusbujanca-ion…pdf`, `valverde-2510.14270v1.pdf`,
`1777-Article Text-3547…pdf`, `Manukonda-1820-Article Text…pdf`,
`wise-assessment-briefing9.pdf`, `Discussion_on_the_Cultivation_of_College.pdf`. All
ten are correct refusals — either a non-bibliographic document, a title that is a
typesetting artefact, or a record that the provider search does not surface.

### 23.7 Observations (reported, deliberately not changed)

Per the instruction not to add heuristics during this validation, the following are
reported rather than fixed:

1. **The year/first-author guards withhold genuine matches when the *local* value is
   wrong** (6 of the 8 withheld cases: local years 2000/2014/2016/2026 and one wrong
   local author, against a perfect 1.000 title match). This is the intended
   "missing beats wrong" trade-off; the productive fix is better local year/author
   precision, not a looser guard.
2. **Short generic titles are not resolvable by bibliographic search.** For
   "Differential Privacy" the top-5 candidates from both Crossref and OpenAlex are
   unrelated papers (best similarity 0.567), so the document is correctly left
   unmatched. I checked and ruled out the typographic-ligature hypothesis: the
   ligature and ASCII forms of the query return the same candidates
   (`Diﬀerential` → `Differential` case-folds identically in the cache key).
3. **Local title artefacts seen in this sample** (all produced `NO_MATCH`, no wrong
   data): `Paper Title (use style: paper title)` (template instruction),
   `A3 fold - briefing 9.pages` (Pages export name), `<Emphasis Type="Italic">Q
   </Emphasis>-learning` (HTML markup leaked into PDF metadata), `The University of
   Manchester Research` (repository cover page), `Book Reviews` (section heading),
   and the `View of …` viewer prefix (which merged fine because the remainder matched).
   The template/markup and section-heading classes are candidate future local title
   corrections; none of them caused a false identity here.
4. **Local author noise is now the most common blocker** rather than title noise
   (`Motion Tracking Survey`, `RROGERS`, `jsb`, `jmir`): the known §22.10 weaknesses
   have a measurable cost on title-search coverage. Acceptable for now, per review.
5. Sample-wide, 12/30 (40 %) of DOI-less documents with a strong title receive a
   high-confidence bibliographic identity, all of it verified — a materially better
   precision/fill trade-off than the earlier exploratory sample (§16) and a plausible
   expectation for Stage 2B (169 eligible documents).

### 23.8 Tests, artifact, stop condition

* New orchestration has tests: `tests/test_metadata_title_stage.py` (7 tests) covers
  eligibility (DOI present / filename-only title / LOW confidence / non-enrichable
  role), deterministic stratified selection, a STRONG match recording the DOI as a
  `CROSSREF`/`MEDIUM` candidate with the "strong title-based external match" evidence
  and a `DOI origin` that says it was *not* observed in the document, an AMBIGUOUS
  match merging nothing, a NO_MATCH that still consults the OpenAlex fallback, and the
  presence of every required report field.
* `261 passed` = 98 RAG-006 tests + 163 pre-existing (RAG-001…005 unchanged).
  **Superseded:** after Stage 2B's three new tests the current total is
  `264 = 101 + 163` (recounted with `pytest --collect-only`, §4 and §24.10).
* Stage-2A artifact: `docs/rag006-stage2a-sample.txt` (45 API requests, 7 cache hits).
  Re-running the same command with a warm cache produces a **byte-identical result
  except for the cache/request counters** (0 requests, 50 cache hits), which confirms
  the sample selection and the match decisions are deterministic.
* Stage 2B (title enrichment of all 169 eligible documents) was **not** started at that
  revision — it is reported in §24. RAG-007 is **not** started.

## 24. Stage 2B — corpus-wide title-based enrichment (this revision)

Scope: title-based enrichment of the **entire eligible population** (169 documents),
using exactly the Stage-2A eligibility and matching rules. No eligibility was broadened,
no threshold was lowered, no local extraction heuristic was added, and the Stage-1 DOI
results were not touched.

Artifacts: `docs/rag006-stage2b.txt` is the current (post-hygiene, §25) run;
`docs/rag006-stage2b-throttled.txt` preserves the run as first reported here;
audit: `docs/rag006-stage2b-audit.txt`. The two runs differ only as described in §25.2.

### 24.1 Eligibility, route, acceptance (unchanged)

```
externally enrichable corpus role
  AND no locally observed DOI
  AND strong local title (PDF_METADATA | DOCX_PROPERTIES | DOCUMENT_TEXT, >= MEDIUM)
```

The 169 eligible documents are the 216 DOI-less ones **minus the 47 that have no strong
local title**; those 47 are never searched — a filename token still cannot initiate a
search (enricher-side `_title_search_source` guard preserved). The route remains
`local title -> Crossref bibliographic search -> OpenAlex fallback` with no DOI lookup as
the initial identity route. Acceptance is unchanged (`STRONG 0.92`, `MIN 0.75`,
ambiguity margin `0.03`, year `±1`, first-author-family consistency); only `STRONG`
merges. A DOI obtained this way is stored as `CROSSREF`/`OPENALEX`, `MEDIUM`,
evidence `strong title-based external match (<provider> title search); external DOI`.

### 24.2 Aggregate results

```
unique documents examined                             466
eligible population                                   169
title searches attempted                              169
STRONG (merged)                                        76   (CROSSREF 58, OPENALEX 18)
AMBIGUOUS (nothing merged)                             58
NO_MATCH (nothing merged)                              35
API/network failures                                   17   (17 x HTTP 429, 0 network errors)
retries                                                34
cached responses used                                 606
cache misses                                           18
API requests made                                      52   (api.crossref.org 1, api.openalex.org 51)
source files modified                                   0
DOI-bearing documents re-checked through the DOI route:
  EXACT accepted                                      246   (unchanged)
  CONFLICT (withheld)                                   3   (unchanged)
  NO_MATCH                                              1   (unchanged placeholder DOI)
```

Those are the counters of the artifact of record (`docs/rag006-stage2b.txt`), which ran
against a warm cache. The first, cold-cache pass of the same run made **232** requests
(Crossref 124, OpenAlex 108) with 203 cache misses and 421 cache hits, and counted **16**
HTTP 429 responses and 29 retries; every outcome counter (`STRONG`/`AMBIGUOUS`/`NO_MATCH`,
the by-source split and the DOI route) was identical (§24.6).

`76 + 58 + 35 = 169`: every eligible document got a decision, and `246 + 3 + 1 = 250`:
every DOI-bearing document is still accounted for exactly as in §22.5.

### 24.3 Whole-corpus metadata coverage at the three points

| field | local only | after DOI Stage 1 | after title Stage 2B |
|-------|-----------:|------------------:|---------------------:|
| title | 421 | 431 | 431 |
| authors | 391 | 422 | **432** |
| year | 458 | 458 | **460** |
| DOI | 250 | 250 | **322** |
| venue | 0 | 237 | **308** |
| publisher | 0 | 246 | **318** |
| publication type | 70 | 312 | **354** |
| URL | 250 | 250 | **326** |
| abstract | 0 | 168 | **206** |

`title` does not move in Stage 2B because every eligible document already had a local
title (an eligibility requirement); what changes is its *provenance*: the accepted external
record supplies a verified title candidate (`CROSSREF`/`OPENALEX`, `MEDIUM`), which the
resolver prefers over the equally-confident local candidate (verified in §24.9).
`authors`, `year`, `venue`, `publisher`, `publication type`, `URL` and `abstract` all
improve because the 76 accepted
records (and the 4 records accepted without a DOI) contribute those fields.

### 24.4 Why 93 matches were withheld

| deciding reason | documents | of which title similarity >= 0.92 | representative example |
|-----------------|----------:|----------------------------------:|------------------------|
| title similarity below minimum (0.75) | 35 | 0 | `ALEKS_McKeown (1).pdf` (`The Impact of ALEKS…` best candidate 0.674), `2021_UNESCO_AI-and-education…pdf` (0.672), artefact titles such as `A3 fold - briefing 3.pages` |
| local year disagreement | 28 | 17 | `GNN_LS_A_Learning_Style_Prediction_in_On.pdf` (title 1.000, local year wrong) |
| local author disagreement | 14 | 14 | `Cai-2206.02102v1.pdf`, `Arrietaa-1910.10045v2.pdf` (title 1.000, wrong local first author) |
| multiple external records | 12 | 11 | `Intelligent_tutoring_systems_D_Sleeman_a.pdf`, `4-AI-Powered.pdf` (two records share the title) |
| title similarity below STRONG (0.75–0.92) | 4 | — | `Pottle-main.pdf` (0.889), `Generative-ai-higher-education-book.pdf` (0.776) |

Reading of these numbers, which is the question this stage was meant to answer:

* **42 of 93 (45 %) were decided by local year/author evidence**, and in **31 of those
  93 (33 %) the title itself matched at >= 0.92** (17 local-year, 14 local-author). That
  is local metadata noise, not bibliographic ambiguity, and the guards behaved exactly as
  approved — they are also what prevented the Stage-2A false positives. No threshold was
  relaxed to recover those matches.
* **12 (13 %) are genuine bibliographic ambiguity** — several external records share the
  same title (11 of the 12 at >= 0.92), e.g. the 1988 report and the 1989 article, or
  ICLR/arXiv versions of one paper.
* **39 (42 %) scored below the acceptance range** (35 below the 0.75 minimum, 4 in
  0.75–0.92): text that is not a title at all (`A3 fold - briefing 3.pages`,
  `Storyboard Template`, `Full Papers.doc`, `QAA277 Transforming Assess.qxd`,
  `Paper Title (use style: paper title)`), works the providers do not surface, local
  titles carrying a journal/running-header prefix (`EDUCATION AND TRAINING Virtual reality
  and the transformation of medical education`, 0.889) and generic short titles
  (`Diﬀerential Privacy`, 0.567).

### 24.5 Audit of every accepted identity (76/76)

Method (artifact `docs/rag006-stage2b-audit.txt`, one record per accepted identity): each
accepted DOI was re-resolved at its registered agency (Crossref, then OpenAlex for
DataCite-style DOIs) and the returned title was compared with the *document's own* local
title; accepted DOIs that are arXiv DOIs were checked by comparing the DOI's arXiv id with
the arXiv id observed in the document itself; the remaining accepted records were checked
by an independent, reproducing title search. Every row also carries the same secondary
evidence the enricher uses (year within ±1, first-author family name).

```
accepted identities                                      76
verified by Crossref re-resolution                       58   (median title similarity 1.000, min 0.921)
verified by arXiv identifier equality                    13   (DOI's arXiv id == the document's own arXiv id)
verified by field agreement (arXiv DOI, no local id)      1   (title 1.000, record reproducible)
verified by field agreement (no DOI in the record)        4   (title 1.000, record reproducible)
secondary evidence over all 76:
  year compatibility (|delta| <= 1)                      74 ok, 0 disagree, 2 unknown
  first-author family name                               51 agree, 0 disagree, 25 unknown
clearly correct                                          76
probably correct                                          0
suspicious                                                0
false-positive accepted identities                        0
```

The 25 `unknown` first-author cross-checks are the **documented local author weakness, not
a conflict**: the local "first author" is either absent or a single boilerplate/brand token
(`DRP`, `ssrn`, `ilkkaw`, `SIGCHI`, `LENOVO`, `RROGERS`) that yields no family name, so the
matcher correctly treats the author as unusable and the identity rests on the title (all
these rows match at >= 0.938) and on the agency re-resolution. **No accepted identity has a
usable local first-author family that disagrees with the resolved record.**

The four rows with title similarity < 0.97 are all benign and individually explained:
`AILiteracyReviewASIST.pdf` (0.938 — the provider stores the title with JATS markup,
below), `View of Challenges…pdf` (0.967 — the local title carries the PDF-viewer
`View of` prefix), `Russell-agapito-978-3-319-10584-0_38.pdf` (0.921 — the local title
carries the LNCS volume prefix `LNCS 8695 - `), `EJ1072608.pdf` (0.968 — capitalisation
only).

The audit is verifiable offline: `docs/rag006-stage2b-audit.txt` states, per identity, the
evidence that was used, and re-rendering it (from the stored agency re-resolutions plus the
local corpus) made **0 API requests** — the only network-adjacent step was 4 cached title
searches — and reproduced the same 76 classifications. The corpus-side checks
(arXiv-id equality, year, first-author family, title similarity) can be recomputed from the
artifact and the local files alone.

Two audit notes, reported rather than auto-corrected (the instruction was not to change
data based on the audit):

1. **One work appears twice — correctly.** `hou-2501.10462v2.pdf` (arXiv v2 preprint,
   4.8 MB) and `hou.pdf` (AAAI camera-ready, 6.8 MB) are the same paper in two different
   renderings (different SHA-256, so two logical documents); both received
   `10.1609/aaai.v39i4.32367` (BloomScene, AAAI 2025). This is the intended behaviour, not
   a false positive.
2. **Two external values carry provider-side encoding artefacts** that the merge copies
   verbatim: `AILiteracyReviewASIST.pdf` gets `<scp>AI</scp> Literacy: …` (JATS markup in
   the Crossref record) and `Ayulani-sriatmaja,…pdf` gets `…Studentsâ€™ Academic…`
   (double-encoded `’` in Crossref). The *identities* are correct, but the merged title
   text is worse than the local text in these two cases. Recommended follow-up (not done
   during this run): a small external-value hygiene step that strips JATS/XML tags and
   repairs double-encoded UTF-8 before merging. No other accepted record has this problem
   (2 of 76).

### 24.6 Lookup reliability (HTTP 429) and its effect

Seventeen queries hit **HTTP 429 from OpenAlex** during the run (16 on the cold-cache pass,
17 on the warm-cache pass; a throttled response is retried, so the counter reflects
responses rather than distinct queries — the search counters themselves are identical in
every pass). Investigation:

* the failures are **deterministic per query within a pass** — re-issuing the same query
  returned 429 again, and raising the request interval from 1 s to 2 s did not help;
* root cause: the OpenAlex **anonymous** rate limit (the project does not set
  `PHD_RAG_METADATA_CROSSREF_MAILTO`, so requests are not in the polite pool);
* effect: those searches had no OpenAlex fallback result. Crossref is always consulted
  *first*, so nothing was fabricated and no identity was altered — the impact is confined to
  documents whose only match would come from OpenAlex, so **76 STRONG is a lower bound**.

> **Correction (§25.3).** An earlier revision of this section claimed the throttling was
> temporary and had cleared on a post-run re-check. That claim was wrong: the diagnostic
> behind it read a host-keyed entry from a status-keyed counter (so it could never observe a
> 429) and therefore classified failed requests as answered. A direct probe shows OpenAlex
> returns `429 {"error":"Rate limit exceeded", "message":"Insufficient budget. ... the free
> daily budget shared by everyone on your network's IP address ... is used up ($0 remaining;
> resets at midnight UTC). Use your own key instead: add ?api_key=..."}`. The condition is
> therefore **not** recoverable with a `mailto`, and the re-run in §25.2 recovered **0**
> throttled lookups.

Three consecutive passes produced byte-identical outcomes (only the cache/request counters
differ), so the run is deterministic. The artifact of record is the last pass (warm cache,
52 requests) because the renderer gained the two missing coverage columns (`title`, `URL`)
after the first pass; the cold-cache counters are quoted alongside in §24.2.

### 24.7 Flagged local weaknesses — measured effect (not fixed, per review)

* Local author noise (`jmir`, `jsb`, `RROGERS`, `Motion Tracking Survey`) and wrong local
  years account for 31 of the 93 withheld matches (§24.4) — the single largest cause of
  lost coverage, and larger than the 11 genuine ambiguity cases.
* Artefact titles (`A3 fold - briefing N.pages`, `Paper Title (use style: paper title)`,
  `<Emphasis Type="Italic">Q </Emphasis>-learning`, `View of …`, `The University of
  Manchester Research`, `Book Reviews`, `Storyboard Template`) produce below-minimum
  similarity and therefore `NO_MATCH` — no wrong identity results from them.
* 47 DOI-less documents have no strong local title at all and were never searched, and 35
  documents have no title: a single-word or generic-title rejection plus scanned/ERIC-style
  files dominate that remainder.

### 24.8 Final metadata gaps (466 unique documents)

| field | populated | missing | note |
|-------|----------:|--------:|------|
| title | 431 | 35 | all 35 without a strong local title (never searched) |
| authors | 432 | 34 | local author noise, non-bibliographic documents |
| year | 460 | 6 | — |
| DOI | 322 | 144 | 250 local + 72 title-matched; 4 matched records had no DOI |
| venue | 308 | 158 | — |
| publisher | 318 | 148 | — |
| publication type | 354 | 112 | — |
| URL | 326 | 140 | follows the DOI |
| abstract | 206 | 260 | Crossref/OpenAlex abstracts only |

Unresolved identifiers, kept strictly separate:

* **DOI-stage conflicts (3, intentionally unresolved, untouched by Stage 2B):**
  `Course_signals_at_Purdue_Using_learning_analytics_.pdf` (`10.1145/2330601.2330666`),
  `David-John-etra22-davidjohn.pdf` (`10.1145/3517031`), `ribiero-1602.04938v3.pdf`
  (`10.1145/2939672.2939778`); plus the placeholder `NO_MATCH`
  `Creative_Use_of_OpenAI_in_Education_Case.pdf` (`10.3390/xxxxx`).
* **Title-search ambiguity (93 documents, no merged value, no conflict record):**
  58 `AMBIGUOUS` + 35 `NO_MATCH`, classified in §24.4.

### 24.9 Confirmations

* **No source document was modified.** The stage runner snapshots size+mtime of every file
  under `data/` before and after the run and reports `source files modified: 0`; the same
  check returned 0 in all three passes, and the corpus is only ever opened read-only.
* **All external provenance remains inspectable.** Real output of
  `python -m app.metadata.inspect "data\literature\AI\+6089-.pdf" --show-candidates`
  (this is the document from §17: no DOI is observed in it any more, and its identity now
  comes from a Crossref title search):

```
Preferred metadata:
Title: Stakeholder Perspectives on the Ethics of AI in Distance-Based Higher Education
DOI: 10.19173/irrodl.v24i2.6089
Venue: The International Review of Research in Open and Distributed Learning
Publisher: Athabasca University Press
Type: JOURNAL_ARTICLE

Provenance:
Title: CROSSREF / MEDIUM

External match status: STRONG

Candidate history:
  doi:
    - [CROSSREF / MEDIUM] 10.19173/irrodl.v24i2.6089 (strong title-based external
      match (Crossref title search); external DOI)
  title:
    - [DOCUMENT_TEXT / MEDIUM] Stakeholder Perspectives on the Ethics of AI in
      Distance-Based Higher Education (prominent first-page text (font-size based
      candidate))
    - [CROSSREF / MEDIUM] Stakeholder Perspectives on the Ethics of AI in
      Distance-Based Higher Education (strong title-based external match (Crossref
      title search); external title)
```

  The local first-page candidate stays in the history next to the external one, so the
  resolved value, its source, its confidence and the evidence string are all inspectable,
  and the externally identified DOI is never represented as observed in the document.

### 24.10 Tests, artifacts, stop condition

* `264 passed, 1 warning` = **101 RAG-006 tests + 163 pre-existing** (RAG-001…005
  unchanged). **Superseded by §25:** after the closure correction the current total is
  `276 = 113 + 163` (12 external-text-hygiene tests added).
* Artifacts: `docs/rag006-stage2b.txt`, `docs/rag006-stage2b-audit.txt`;
  `docs/rag006-doi-stage-corrected.txt` and `docs/rag006-stage2a-sample.txt` unchanged.
* Still not done (deliberately): resolving the three DOI conflicts, the 47 DOI-less
  documents without a strong title, and the external-value hygiene fix of §24.5.
* **RAG-007 is not started.**

## 25. Closure corrections — external text hygiene and lookup identity

Two targeted corrections were requested before closing RAG-006. Nothing else changed:
matching thresholds, eligibility, DOI/title/author extraction, provider priority and the
three DOI conflicts are untouched.

### 25.1 External metadata text hygiene (new)

`app/metadata/external/hygiene.py` adds one deterministic step that is applied to
**external values only**, before a record becomes a candidate:

```
unescape HTML entities            &amp; -> &              &#8211; -> –
remove inline presentation tags   <scp>AI</scp> -> AI       <italic>Q</italic>-learning -> Q-learning
remove block-level markup         <jats:p>a</jats:p><jats:p>b</jats:p> -> a b
repair double-encoded UTF-8       Studentsâ€™ -> Students’      Ã© -> é
collapse whitespace                                                    (pre-existing behaviour)
```

Rules and limits:

* it never rewrites wording, case or punctuation; a value that matches none of the rules
  is returned unchanged (`DREAMFUSION: TEXT-TO-3D USING 2D DIFFUSION`,
  `feduc-2021-645758 1..9`, `5 < 10 and 20 > 3 cases` are all untouched);
* valid Unicode is preserved (`Café`, `Başak Melis Öcal`, `‘Smart’`, `–`, CJK text, even
  the Portuguese words `Ângulo`/`Ãngela`, which merely *look* like mojibake);
* mojibake is repaired only when the byte round-trip is unambiguous: the value must be
  encodable as single bytes, must decode as valid UTF-8, and the repair must strictly
  reduce the mojibake signature — otherwise the original is kept;
* it applies to `title`, `venue`, `publisher`, author names and `abstract`. URLs are left
  alone; the provider type enum is untouched;
* **no LLM is involved** and local extraction is unchanged (the local extractor produces
  the same result offline).

Provenance is preserved: `ExternalRecord` now carries `raw_title` (the provider's text
when it differed) and `normalised` (which fields were normalised), and the merge marks
those fields in the candidate evidence:

```
title: - [CROSSREF / MEDIUM] AI Literacy: Definition, Teaching, Evaluation and Ethical
         Issues (strong title-based external match (Crossref title search);
         external title (provider text normalised))
```

so an auditor can always tell that a stored value is normalised rather than verbatim, and
can recover the provider text from `raw_title` or from the cached raw response. Raw
provider JSON in the cache is never modified.

Measured corpus-wide effect (enrichable documents, warm cache, **0 API requests**):

```
documents with at least one normalised external value   207 / 466
  abstract                                              190
  venue                                                  16
  title                                                   6
  authors                                                 1
```

The seven documents whose *title* or *authors* changed were each re-checked; every
preferred value is clean and its identity is unchanged:

| document | provider raw text | stored preferred value |
|----------|-------------------|------------------------|
| `AILiteracyReviewASIST.pdf` | `<scp>AI</scp> Literacy: Definition, Teaching, Evaluation and Ethical Issues` | `AI Literacy: Definition, Teaching, Evaluation and Ethical Issues` |
| `Ayulani-sriatmaja,+Bevina+16880-AAP.pdf` | `…for Predicting Studentsâ€™ Academic Performance` | `…for Predicting Students’ Academic Performance` |
| `Euro J of Education - 2022 - Holmes - …pdf` | `State of the art and practice in\n <scp>AI</scp>\n in education` | `State of the art and practice in AI in education` |
| `Jain-inf_terrain_generation.pdf` | `Adaptive &amp; Multi-Resolution …` | `Adaptive & Multi-Resolution …` |
| `Nursing_Midwifery_students_experience_of_immersiv.pdf` | `Nursing &amp; Midwifery …` | `Nursing & Midwifery …` |
| `sunkara-WJAETS-2025-0491.pdf` | `…Transforming CRM &amp; ERP…` | `…Transforming CRM & ERP…` |
| `frobt-01-00009.pdf` | (author name carried markup) | `Bernhard Spanlang` |

Regression tests: `tests/test_metadata_hygiene.py` (12 tests) covers JATS tag stripping,
nested/simple XML tags, block markup, HTML entities, the double-encoded apostrophe,
valid Unicode unchanged, ordinary titles unchanged, whitespace-only collapsing, empty
values, the two clients' record construction (cleaned value + `raw_title` + `normalised`),
the contact identity in requests, and the evidence marker on a merged title.

### 25.2 Stage 2B re-run against the warm cache

Both clients now build their request headers through one helper
(`_headers_with_contact`), so the configured contact identity is sent as a `User-Agent`
suffix **and** (for OpenAlex) as the `mailto` parameter. The re-run was requested to
recover the throttled OpenAlex fallbacks.

**Limitation, reported rather than worked around:** this environment has **no contact
identity configured** (`PHD_RAG_METADATA_CROSSREF_MAILTO` is unset and there is no `.env`
file), and no address was invented. The wiring is therefore exercised only by tests, not
by the live run; the live run identifies itself as
`phd-rag/0.1 (academic metadata extraction)`.

```
OpenAlex fallback queries attempted          17 distinct queries, 51 requests
                                             (each throttled query is retried twice)
HTTP 429 responses                           17
newly recovered provider responses            0
STRONG        before / after                 76 / 76
AMBIGUOUS     before / after                 58 / 58
NO_MATCH      before / after                 35 / 35
API requests made                            52  (Crossref 1, OpenAlex 51)
cached responses used / cache misses         606 / 18
source files modified                         0
DOI route re-check                           246 EXACT / 3 CONFLICT / 1 NO_MATCH (unchanged)
```

The only differences between `docs/rag006-stage2b-throttled.txt` and the current
`docs/rag006-stage2b.txt` are the hygiene results (11 changed lines overall):

* `AILiteracyReviewASIST.pdf` external title `0.938 -> 1.000` similarity (markup removed);
* `Ayulani-sriatmaja,…pdf` external title `0.979 -> 1.000` (mojibake repaired);
* one accepted venue `Language Learning &amp; Technology -> Language Learning & Technology`;
* one **withheld** similarity `watkins-BF00992698.pdf` `0.593 -> 0.273` — its local title
  keeps provider markup (`<Emphasis Type="Italic">Q </Emphasis>-learning`, local extraction
  is unchanged by design), so removing the markup from the *external* candidate drops a
  score that was previously inflated by markup shared between the two sides. The status is
  unchanged (`NO_MATCH`, below the 0.75 minimum either way), and the comparison is now
  against real text.

### 25.3 Corrected root cause of the HTTP 429s

The correction matters for anyone re-running this: the failures are **not** a polite-pool
problem and are **not** recoverable by setting a contact address. A direct probe of a
previously throttled query returns:

```
HTTP/1.1 429
{"error":"Rate limit exceeded","message":"Insufficient budget. This request has no API key,
 so it counts against the free daily budget shared by everyone on your network's IP address,
 and that budget is used up ($0 remaining; resets at midnight UTC). Use your own key instead:
 add ?api_key=YOUR_KEY t..."}
```

So the OpenAlex *shared* free daily budget for this network was exhausted at run time, and
OpenAlex requires an `api_key` in that state. Consequence for the recorded run: OpenAlex
contributed nothing for those 17 queries, `76 STRONG` is a lower bound, and the fix is
either an API key (a new configuration concern, outside this task's scope) or a re-run
after the daily reset / from a different network. Crossref was queried first for every
document and never failed, so no identity depends on a throttled query. Nothing was
re-issued for the 246 DOI-exact records; the cached responses were reused.

### 25.4 Re-audit of changed or newly accepted identities

* **Newly accepted identities: 0** (76 before, 76 after), so there is nothing new to audit.
* **Identities whose metadata changed: 3** (the two cleaned titles and the decoded venue
  above) plus the one author-name case. All were re-checked against the provider record in
  §25.1: same DOI, same match status, and the stored text now equals the provider's text
  after markup/entity/encoding removal. Two of them became *exact* title matches
  (`0.938` and `0.979` → `1.000`), i.e. the hygiene step strictly improved the evidence.
* **The other 73 accepted identities were not re-audited** (their metadata is unchanged),
  per the instruction.
* No `suspicious` and no `false positive` identity was found; the zero-false-positive
  result of §24.5 stands.

### 25.5 Final RAG-006 state

Whole-corpus coverage (466 unique documents):

| field | populated | missing |
|-------|----------:|--------:|
| title | 431 | 35 |
| authors | 432 | 34 |
| year | 460 | 6 |
| DOI | 322 | 144 |
| venue | 308 | 158 |
| publisher | 318 | 148 |
| publication type | 354 | 112 |
| URL | 326 | 140 |
| abstract | 206 | 260 |

Confirmations:

```
source files modified                     0     (size+mtime snapshot around the re-run)
external provenance preserved             yes   raw_title + normalised on each record;
                                                evidence marks normalised values;
                                                raw provider JSON untouched in the cache
three DOI conflicts unchanged             yes   246 EXACT / 3 CONFLICT / 1 NO_MATCH,
                                                conflicts listed in §22.6 and §24.8
title-search thresholds unchanged         yes   STRONG 0.92, MIN 0.75, margin 0.03,
                                                year ±1, first-author-family consistency
filename-only titles cannot initiate
  a search                                yes   _title_search_source guard (chapter 23)
my_work drafts protected                  yes   is_enrichable(role, has_doi=…) unchanged:
                                                my_work only with a DOI, rules /
                                                supervision / future_chapters / summaries /
                                                guidance never
```

Tests: `276 passed` = **113 RAG-006 tests + 163 pre-existing** (264 before this closure
step; 12 hygiene tests added). Artifacts: `docs/rag006-stage2b.txt` (current),
`docs/rag006-stage2b-throttled.txt` (run as first reported), `docs/rag006-stage2b-audit.txt`,
`docs/rag006-stage2a-sample.txt`, `docs/rag006-doi-stage-corrected.txt`.

**RAG-006 COMPLETE.**

RAG-007 is not started.

## Reproduction

```powershell.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m app.metadata.report "data"
.\.venv\Scripts\python.exe -m app.metadata.report "data" --sample 20
.\.venv\Scripts\python.exe -m app.metadata.doi_stage "data" --out "docs\rag006-doi-stage.txt"
.\.venv\Scripts\python.exe -m app.metadata.doi_stage "data" --out "docs\rag006-doi-stage-corrected.txt"
.\.venv\Scripts\python.exe -m app.metadata.title_stage "data" --size 30 --out "docs\rag006-stage2a-sample.txt"
.\.venv\Scripts\python.exe -m app.metadata.title_stage "data" --all --out "docs\rag006-stage2b.txt"
.\.venv\Scripts\python.exe -m app.metadata.inspect "data\literature\AI\paper.pdf" --offline
```

The Stage-2B audit artifact `docs\rag006-stage2b-audit.txt` was produced by an ad-hoc
verification script (agency re-resolution of the 76 accepted DOIs, arXiv-id comparison,
independent reproduction of the DOI-less matches) that is not part of the repository; every
result it records is restated in §24.5 and is recomputable from the artifact plus the local
corpus. `docs\rag006-stage2b-throttled.txt` keeps the Stage-2B run as first reported, before
the §25 hygiene correction; the `--all` command above writes the current run.
