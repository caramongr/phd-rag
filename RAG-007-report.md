# RAG-007 Report — Academic-aware normalization and structure-preserving chunking

Task: normalize parsed academic text for retrieval and cut it into chunks that keep
exact document, page, section and extraction provenance. Nothing else was built: no
embeddings, no Qdrant, no retrieval, no reranking, no generation, no citation
rendering, no UI, and no LLM involvement of any kind.

Reading order: §5 is the chunk model, §8 the algorithm, §14 the role policy, §16–§22
the real-corpus validation, §23–§24 the honest limitations and architectural notes,
and §25 the deduplicated-document role correction.

## 1. Files created

Application package `app/chunking/`:

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `models.py` | `ChunkingOptions`, `ChunkedDocument`, `SkippedItem` and the audit reason codes |
| `normalize.py` | Line-ending/Unicode/whitespace normalization, dehyphenation, wrapped-line joining, running header/footer detection |
| `classify.py` | Section-kind detection (abstract / references / conclusion / appendix), caption detection, heading-title normalization |
| `tokenizer.py` | Deterministic offline subword tokenizer (`subword-estimator-v1`) behind a `Tokenizer` protocol |
| `policy.py` | Corpus-role → retrieval/evidence/authority policy |
| `chunker.py` | `DocumentChunker` / `chunk_document`: the structure-preserving algorithm |
| `inspect.py` | `python -m app.chunking.inspect` CLI |
| `report.py` | `python -m app.chunking.report` corpus CLI |

Tests: `tests/chunking_support.py` and
`tests/test_chunking_{normalize,structure,policy,provenance,determinism,cli,duplicate_roles}.py`.

Artifacts: `docs/rag007-chunking.txt` (full-corpus report),
`docs/rag007-inspections.txt` (manual inspection of ten representative documents),
`docs/rag007-catmull-ocr.txt` (OCR provenance check on the scanned thesis),
`docs/rag007-chunking-pre-rolefix.txt` (the pre-correction corpus report kept for
the §25 before/after comparison) and this `RAG-007-report.md`.

## 2. Files modified

| File | Change |
|------|--------|
| `app/models/chunk.py` | **Additive** extension of the RAG-001 `Chunk` model (see §5) plus the new `ChunkMetadataRef`; every new field has a default, so all RAG-001 usage stays valid. `ChunkMetadataRef` also carries the duplicate-location audit fields of §25 |
| `app/models/enums.py` | `ContentType` gained `CAPTION` and `APPENDIX` (both actually emitted, see §18) |
| `app/models/__init__.py` | Exports `ChunkMetadataRef` |
| `README.md` | Documents the two new CLIs, the target sizes, the XLSX exclusion and the deduplicated-document role rule |
| `app/metadata/models.py` | Added `PhysicalRole` / `RoleResolution` and `AcademicMetadata.role_resolution` (§25) |
| `app/metadata/corpus.py` | Added `ROLE_PRECEDENCE`, `resolve_corpus_role`, `physical_paths_for_hash` (§25) |
| `app/metadata/extractor.py` | `extract_metadata(..., physical_paths=…)`: the role is resolved from every physical location (§25) |
| `app/metadata/report.py`, `doi_stage.py`, `title_stage.py`, `inspect.py` | Pass all physical locations of a logical document; the inspector prints the role resolution |
| `app/chunking/chunker.py` | Populates the duplicate-location / role-resolution audit fields (§25) |
| `app/chunking/report.py` | New "Duplicate copies and corpus roles" audit section (§25) |
| `app/chunking/inspect.py` | Optional `--corpus` root to resolve the role across duplicates; prints the resolution |

No parser, ingestion, metadata *value*, resolver or existing test file was
modified; no existing test was edited or skipped.

## 3. Dependencies added

**None were added by RAG-007.** The tokenizer is internal and deterministic
(`tokenizer.py`), so the
chunking layer adds no runtime dependency and needs no network. `transformers`,
`tokenizers` and `sentence-transformers` happen to be installed in this
environment, but they are not used: no tokenizer/vocabulary asset is present
locally (`%USERPROFILE%\.cache\huggingface` and the tiktoken cache are both
absent), so using one would require a model download and would tie chunk sizing to
the embedding model RAG-008 will choose — which the task explicitly forbids.
`pyproject.toml` was not touched by RAG-007 (its uncommitted change is RAG-006's
runtime-dependency declaration).

## 4. Final pytest result

```
362 passed, 1 warning
```

`362 = 276 pre-existing (RAG-001…RAG-006) + 86 new RAG-007 tests`. The
pre-existing 276 all still pass unchanged — no existing test was edited, skipped or
relaxed.

New RAG-007 tests by file:

| File | Tests | Covers |
|------|------:|--------|
| `tests/test_chunking_normalize.py` | 9 | Unicode/NFC, ligatures, NBSP/zero-width, line endings, whitespace, dehyphenation, wrapped-line joining, running headers |
| `tests/test_chunking_structure.py` | 18 | heading hierarchy, section kinds, abstract/references/conclusion/appendix, captions, table vs prose, table splitting, merge rules, oversize splitting |
| `tests/test_chunking_policy.py` | 19 | role → retrieval/evidence/authority policy, role matrix, corpus-role gating, filename-only and unsupported-extension handling |
| `tests/test_chunking_provenance.py` | 10 | page ranges, mixed native/OCR sources, locators, sequence ranges, metadata ref, skipped-item reasons |
| `tests/test_chunking_determinism.py` | 6 | stable chunk/document ids, stable ordering, idempotent re-chunk, token counts |
| `tests/test_chunking_cli.py` | 12 | `inspect` and `report` CLIs, limits, exit codes, canonical text-source buckets |
| `tests/test_chunking_duplicate_roles.py` | 12 | deduplicated documents: role from all physical locations, conservative precedence, evidence policy, provenance, reversed discovery order, one logical document, the RAG-006 stage gate |

The 12 duplicate-role tests are listed individually in §25.5.

## 5. Final chunk model

`app/models/chunk.py` — the RAG-001 model extended additively. Every new field has
a default, and the existing RAG-001 tests (`test_chunk_model.py`) still pass
untouched.

| Field | Type | Meaning |
|-------|------|---------|
| `id` | UUID | Deterministic chunk id (§15) |
| `document_id` | UUID | Deterministic logical-document id (`uuid5` of the file hash) |
| `chunk_index` | int | Zero-based position within the document (source order) |
| `text` | str | Normalized **source** text — never contains synthetic labels |
| `retrieval_text` | str \| None | Contextualised text for embedding; may carry `[Document: …]` / `[Section: …]` labels that are *not* in the document |
| `page_start`, `page_end` | int \| None | One-based page range (min/max of contributing items) |
| `section_title` | str \| None | Innermost heading of the chunk |
| `heading_path` | list[str] | Full heading trail, outermost first |
| `token_count` | int \| None | Tokens per the estimator in §7 |
| `content_type` | ContentType | `BODY`, `ABSTRACT`, `TABLE`, `REFERENCES`, `CONCLUSION`, `CAPTION`, `APPENDIX`, `SUMMARY_ENTRY`, `OTHER` |
| `text_sources` | list[TextSource] | `NATIVE_PDF` and/or `OCR` (both when mixed); `ocr_involved` property |
| `source_locators` | list[str] | `source_locator` of every contributing parsed item (`page:3:block:12`, `page:2:table:0`, `page:4:ocr:1`) |
| `sequence_start`, `sequence_end` | int \| None | Inclusive range in the canonical parsed content stream |
| `metadata_ref` | ChunkMetadataRef \| None | Minimal bibliographic payload — see below |
| `corpus_role` | CorpusRole \| None | RAG-006 role vocabulary, resolved from **all** physical locations of the logical document (§25) |
| `authority` | AuthorityLevel \| None | RAG-001 authority vocabulary, from the §14 policy |
| `retrieval_eligible` | bool | May answer normal retrieval (false for reference lists) |
| `evidence_eligible` | bool | May support a scholarly claim (false for every non-literature role) |
| `merged_sections` | list[str] | Sibling section titles merged into this chunk (empty unless the §8.7 merge happened) |
| `split_from_previous` | bool | This chunk continues a paragraph/table that exceeded the hard maximum |
| `overlap_tokens` | int | Always `0` in RAG-007 (structural context instead of blind overlap) |
| `created_at` | datetime | Injectable; tests pin it for determinism |

`ChunkMetadataRef` is deliberately small — `file_hash`, `filename`, `title`,
`year`, `first_author`, `doi`, `publication_type` — so a chunk carries what
retrieval/ranking needs without duplicating the whole RAG-006 record (with its
candidate history) into every chunk. The full `AcademicMetadata` stays
addressable by `file_hash`. Two further fields are populated **only** for the
uncommon case of a logical document with several physical copies:
`duplicate_locations` (every path) and `role_resolution` (why the effective role
was chosen, §25.4).

## 6. Normalization rules

`app/chunking/normalize.py`, version tag `rag007-normalize-v1`. Applied to parsed
content only; the raw extraction (`ParsedDocument.raw_text`, `ParsedPage.native_text`,
`ParsedPage.ocr_text`) is never modified, and every chunk points back to the parsed
items it came from.

| Rule | Effect | Example |
|------|--------|---------|
| Line endings | CRLF/CR → LF | `a\r\nb` → `a\nb` |
| Space-like characters | NBSP, figure/narrow spaces, ideographic space → space | `non\u00a0breaking` → `non breaking` |
| Zero-width characters | removed (ZWSP, ZWNJ, ZWJ, BOM, word joiner) | — |
| Unicode | NFC plus ligature folding | `de\ufb01nition` → `definition` |
| Whitespace | runs of spaces/tabs and blank-line runs collapsed, lines trimmed | `a   b` → `a b` |
| Wrap hyphenation | line-final letter+hyphen joined when the next line starts lowercase and the prefix is ≥3 chars | `educa-\ntion` → `education` |
| Wrapped lines | intra-block lines joined with one space; a newline is kept before list markers | — |
| Running headers/footers | first/last line of a page removed when it repeats on ≥ max(3, 50 %) of pages (digit runs masked, only leading/trailing lines of a block touched) | `Journal of Testing`, `1`, `2`, … removed |

Explicitly **not** done: sentence rewriting, grammar/spelling correction,
summarisation, paraphrasing, terminology changes, translation, citation removal,
or any LLM call. Academic in-text citations (`(Smith, 2022)`,
`Smith et al. (2023)`, `[14]`, `[12, pp. 4-8]`) are preserved verbatim — RAG-007
neither resolves nor parses them.

## 7. Token-counting strategy

`tokenizer.py` implements `subword-estimator-v1`:

* units: letter words (internal apostrophes kept), numbers, underscore runs,
  punctuation/symbol runs, whitespace;
* a word unit contributes `ceil(len / 4)` pieces, a symbol run `ceil(len / 2)`;
* whitespace is attached to the following piece, as BPE does, so it adds no token;
* `split_tokens` is the primitive, and `"".join(split_tokens(text)) == text`, so a
  hard split can cut at an exact token budget (`truncate`);
* `count_tokens` is `len(split_tokens(...))` — token counts are never character
  counts.

Measured behaviour on English prose: ~3.8 characters per token
(300-character paragraph → 79 tokens), i.e. within ±20 % of a GPT-2-class BPE and
slightly conservative for the hard maximum. The `Tokenizer` protocol allows a real
vocabulary to be substituted for measurement (see §24.5).

## 8. Structural chunking algorithm

`chunker.py`, in order:

1. **Content stream** — `ParsedDocument.content_items` is consumed in
   `sequence_index` order (blocks and tables share one ordered stream);
   if a parser supplied none, blocks and tables are merged by sequence index.
2. **Running-line detection** — first/last lines per page are counted once for the
   whole document; matches are stripped from the leading/trailing lines of blocks.
3. **Table de-duplication** — a page's table cell text becomes a coverage set; a
   prose block on that page is dropped when its collapsed text appears verbatim in
   the table text (≥20 chars) or when every one of its lines equals a table cell.
   Both the removal count and each removed item are recorded.
4. **Normalization** — each block is normalized (§6); empty results are recorded as
   skipped items rather than silently discarded.
5. **Classification** — section kind (abstract / references / conclusion /
   appendix / body) from the heading trail, caption detection by labelled pattern,
   `TABLE` for table items.
6. **Accumulation** — headings close the section; prose and captions accumulate
   while the section (heading path) stays the same and the token budget allows:
   * push an item; if the buffer reaches the target (1000) close it;
   * if adding an item would exceed the hard maximum (1200) close the buffer first;
   * a single item larger than the hard maximum is split at sentence boundaries
     (step 6 above), each part becoming its own chunk with
     `split_from_previous=True`;
   * tables always stand alone and are never mixed with prose.
7. **Short-section merging** — a chunk below 700 tokens is merged into the previous
   chunk when both are the same section kind, neither is a table, `following` sits
   under `previous`'s heading (or shares a non-empty parent heading), and the
   result stays ≤1000 tokens. Merged chunks record the contributing titles in
   `merged_sections` and take the *common parent* as `heading_path`, so the label
   never claims a section the text does not cover.
8. **Chunk construction** — page range, locators, sequence range, text sources,
   role policy, `retrieval_text` and the deterministic id are attached.

Invariants: chunks follow source order; `chunk_index` is contiguous from 0; no
chunk exceeds the hard maximum of 1200 tokens (1500 is never reached); no chunk is
empty; no chunk loses its page/section/role context.

## 9. Section/heading handling

Headings are **structure, not content**: a heading block never becomes a chunk of
its own; it closes the previous section and sets the context for what follows
(the parser's own `heading_path` is authoritative, and the heading's own text is
appended if a parser ever omits it from the trail). Content items therefore inherit:

* `heading_path` — the full trail, e.g.
  `['4 Virtual Reality in Higher Education', '4.2 Educational Applications', '4.2.1 Medical Training']`;
* `section_title` — the innermost heading;
* the section kind used for content typing:
  * `Abstract` / `Summary` / `Zusammenfassung` … → `ABSTRACT`;
  * `References` / `Bibliography` / `Works cited` / `Literaturverzeichnis` … → `REFERENCES`;
  * `Conclusion(s)` / `Concluding remarks` / `Discussion and Conclusion` … → `CONCLUSION`;
  * `Appendix` / `Annex …` → `APPENDIX`;
  * everything else → `BODY` (or `SUMMARY_ENTRY` for summary-role documents).

Two safeguards: a heading that receives no content is counted
(`headings_without_content`) instead of producing an empty chunk, and an `ABSTRACT`
label is applied only to the **first** chunk of an abstract section — a heading
detector that fails to close the abstract cannot then label whole body paragraphs
as an abstract.

§15's requirement is met by keeping two fields apart: `text` is the normalized
source content verbatim, `retrieval_text` additionally carries the synthetic label
`[Document: …]\n[Section: A > B]` for embedding. Because the label never enters
`text`, quote-capable source text is never contaminated.

## 10. Page provenance handling

* `page_start`/`page_end` are the min/max page of the contributing items; a chunk
  that stays on one page reports that page for both.
* Pages come from the parsers (RAG-004); documents without pages (DOCX, Markdown)
  leave both fields `None` rather than inventing a page.
* Chunks may cross page boundaries for semantic continuity (measured in §16); they
  never reorder pages, and `source_locators` pin the exact blocks on each page.
* A chunk is therefore never page-sized by construction: page breaks do not split
  a paragraph, and page boundaries do not create chunk boundaries.

## 11. OCR provenance handling

RAG-005 already marks OCR text at page and block level; RAG-007 carries that mark
to the chunk without changing it:

* a block whose `text_source` is `OCR` contributes `TextSource.OCR` to
  `chunk.text_sources`;
* a chunk with several contributing items lists every distinct source, so a
  native+OCR chunk is marked as **both** (`ocr_involved` is `True`);
* tables inherit the effective text source of their page;
* nothing is repaired, and OCR text stays ordinary retrievable prose with an
  explicit provenance mark — the RAG-005 rule that OCR-derived equations, figure
  labels and complex layout text are *not* verbatim-authoritative is preserved by
  the mark rather than by reinterpretation.

## 12. Table handling

* A table is rendered deterministically as pipe-separated rows, e.g.
  `Condition | Effect size | N` / `VR training | 0.72 | 120`; pipes inside cells
  are escaped (`\|`), empty rows are dropped.
* `content_type = TABLE`, page and section provenance preserved, text source
  inherited from the page.
* Tables are never mixed with prose in one chunk and never merged with a sibling.
* A table is kept intact up to 1200 tokens; beyond that it is split **by rows**
  (each part ≤1200, `split_from_previous=True` for continuations), and the header
  row is not repeated, so no text is duplicated. A table inside a reference list
  is `REFERENCES`, not `TABLE`.
* **The RAG-004 duplication issue is handled explicitly**: RAG-004 keeps table text
  in the prose blocks as well, so RAG-007 drops prose blocks that the page's
  structured table already represents (counted as
  `duplicate_table_blocks_removed`, and listed per item in `skipped_items`). This
  behaviour is tested both ways (`test_table_text_is_not_duplicated_when_the_parser_emits_it_as_prose`,
  `test_table_text_on_a_different_page_is_not_deduplicated`).

## 13. Reference-section handling

* Sections whose heading is a reference/bibliography heading (including several
  non-English spellings) are `content_type = REFERENCES`, and every entry is kept
  in the corpus — nothing is deleted.
* `retrieval_eligible = False` and `evidence_eligible = False` for those chunks, so
  a bibliography entry can never be retrieved as if it were a substantive claim;
  the RAG-007 policy enforces this independently of the corpus role, and a table
  inside a reference list inherits the same exclusion.
* A block whose own first line is `References`/`Bibliography` (heading glued to the
  first entry) is still recognised: the line becomes the section context and the
  rest of the block is typed as `REFERENCES`.
* References are **not** parsed into a citation graph and in-text citations are not
  resolved (out of scope for RAG-007).

## 14. Corpus-role and evidence policy

`policy.py`; the role vocabulary is RAG-006's `CorpusRole`, the authority
vocabulary is RAG-001's `AuthorityLevel`. Roles are never collapsed into one
conceptual vector corpus.

| Corpus role | retrieval_eligible | evidence_eligible | authority | meaning |
|-------------|--------------------|-------------------|-----------|---------|
| `literature` | yes | **yes** | `primary_evidence` | scholarly evidence |
| `my_work` | yes | **no** | `high` | terminology continuity, consistency checking, never external evidence |
| `summaries` | yes | **no** | `navigation_only` | navigate to the original; never cited as evidence |
| `supervision` | yes | **no** | `medium` | context for supervisory decisions |
| `rules` | yes | **no** | `instruction` | instruction context |
| `guidance` | yes | **no** | `instruction` | instruction context |
| `future_chapters` | yes | **no** | `navigation_only` | NAVIGATION_ONLY planning guidance, NOT_EVIDENCE |
| `other` (unknown provenance) | yes | **no** | `navigation_only` | fallback: retrievable, never evidence |

Content-type override: `REFERENCES` is never retrieval- or evidence-eligible in any
role. `summary_entry` typing (§9) keeps internal summaries distinguishable from the
literature they summarise.

## 15. Deterministic-ID strategy

* `document_id = uuid5(DOCUMENT_ID_NAMESPACE, file_hash)` — derived from the
  SHA-256 identity RAG-002 established, so the same file always yields the same
  logical document id.
* `id = uuid5(CHUNK_ID_NAMESPACE, f"{file_hash}:{chunk_index}:{sha256(text)[:32]}")`
  — document identity + position + normalized content. No random UUIDs are used,
  so re-running on an unchanged corpus reproduces identical ids.
* `created_at` is injectable (`ChunkingOptions.created_at`) and defaults to *now*;
  tests pin it, and payload comparison excludes it, so the only time-dependent
  field is the timestamp.


## 16. Full-corpus chunk statistics

Read-only run over the whole corpus (`python -m app.chunking.report data
--out docs/rag007-chunking.txt`), OCR fallback enabled, artifact
`docs/rag007-chunking.txt`:

```
unique documents considered      466
documents successfully chunked   465
documents without any chunk        0
documents skipped                  1     (.xlsx, see below)
parsing errors                     0
chunking errors                    0

by extension   .pdf 444   .docx 18   .md 3   .xlsx 1

total chunks                     14786
retrieval-eligible chunks        11469
evidence-eligible chunks         10456
chunks ineligible for retrieval   3317   (= the reference chunks, exactly)
total tokens                  10703684

logical documents with >1 physical copy   34
logical documents whose copies imply >1 role   2   (§25)

duplicate table blocks removed    4254
running header/footer lines removed 9891
headings without content           265
skipped parsed items              9409
```

**XLSX is unsupported for chunking in RAG-007** and is reported, not discarded:
the report counts `.xlsx: 1` and names the file (`alogirithms-8-enriched.xlsx`)
under “Skipped documents”, so structured spreadsheet ingestion can be added later
without changing the corpus statistics.

Eligibility sanity check: `11469 + 3317 = 14786`, i.e. **every** chunk that is not
retrieval-eligible is a reference chunk, and no other chunk loses eligibility.
Evidence eligibility is 10456 of 14786 because, by policy, only `literature`
chunks may support a scholarly claim (§14) — summaries, future chapters, own work,
rules, guidance and supervision records remain retrievable.

The run also audits deduplicated documents (§25): 34 of the 466 logical documents
have more than one physical copy, and **2** of those copies imply different corpus
roles; both resolve conservatively to `my_work` (see §25.6).

## 17. Token-size distribution

```
min     1        median  941      mean  723.9      p95  1135      max  1200

<200 tokens        2639
200-699 tokens     3036
700-1000 tokens    2791
1001-1200 tokens   6320
1201-1500 tokens      0
>1500 tokens          0
```

The accumulator closes a chunk as soon as it reaches the 1000-token target, so a
chunk normally lands in *target … target + last paragraph*, capped by the hard
maximum — which is why 1001–1200 is the largest bucket and why nothing exceeds
1200. The 2639 chunks below 200 tokens are structural short units: abstracts,
short conclusions, reference-list fragments, captions, page-1 preambles and
tables; they are reported rather than padded, because merging unrelated short
material would damage provenance (and short sibling sections *are* merged where it
is safe, §8).

## 18. Content-type distribution

```
body            9057
references      3317
table           1243
summary_entry    648
conclusion       289
abstract         178
appendix          50
caption            4
```

`CAPTION` is rare by design: captions are kept as context inside the chunk they
belong to, and become a `CAPTION` chunk only when a caption is the whole content
(4 cases). `APPENDIX` appears where a document has an appendix-style heading.
`SUMMARY_ENTRY` covers the internal summary files (648 chunks), which stay
distinguishable from the literature they summarise.

## 19. Role distribution

```
literature       13741
summary entries    648   (role: summaries)
future_chapters    210
my_work            113
rules               62
supervision         10
guidance             2
```

No document was classified as `other`, and no chunk carries a role different from
its document's role. `my_work` (113 chunks, including the 53 chunks of the draft
that also exists under `literature/`, §25) and `future_chapters` (210) are all
`retrieval_eligible = yes`, `evidence_eligible = no`; `rules`, `guidance` and
`supervision` are likewise non-evidence; only the 13741 literature chunks can be
used as scholarly evidence.

Text sources (canonical labels):

```
native_pdf                       13712
n/a (DOCX/Markdown)                961
ocr                                 88
native_pdf+ocr                      25
```

113 chunks contain OCR text and 25 of them mix native and OCR text in one chunk
(the mix is recorded, never flattened).

## 20. Manual validation results

Ten representative documents were inspected by hand with
`python -m app.chunking.inspect` (artifact `docs/rag007-inspections.txt`), checking
section continuity, chunk boundaries, citation preservation, page provenance, OCR
provenance, role/evidence classification, header/footer contamination and
bibliography leakage.

| # | Document | Chunks | Assessment | Notes |
|---|----------|-------:|------------|-------|
| 1 | `+6089-.pdf` (journal article, 22 p.) | 27 | **usable with minor issues** | Abstract, Introduction, Methodology, Results, Discussion, Conclusion and References all detected; page ranges (`p1-2`, `p12-13`, …) correct; 12 duplicated table blocks removed; in-text citations intact. Minor: a 124-token page-1 masthead preamble chunk, and the publisher's trailing “methodology” heading (a parser decision) starts a final section. |
| 2 | `David-John-etra22-davidjohn.pdf` (two-column ACM, 6 p.) | 20 | **usable with minor issues** | Two-column reading order is correct, page numbers/headers removed (5 lines), 15 duplicate table blocks removed. No headings are detected after the abstract, so the section label stays coarse (the abstract safeguard keeps the label honest); a 55-token author/affiliation preamble and a 17-token DOI footnote chunk remain. |
| 3 | `Radianti -main.pdf` (systematic review, 29 p.) | 51 | **good** | Nested heading paths such as `2. Theoretical background > 2.1. VR and immersion`; sensible page ranges; largest chunk 1183 tokens. Minor: the Elsevier licence masthead forms the first chunk. |
| 4 | `CHI20_WalkingVibe_preprint.pdf` (repository download, 13 p.) | 19 | **usable with minor issues** | Retrieval text is clean and citations survive, but the ResearchGate sidebar line `82 PUBLICATIONS 1,829 CITATIONS` is detected as a heading and becomes the section title for several chunks (parser heading inference — §23.2). |
| 5 | Biggs & Tang, *Teaching for Quality Learning* (book PDF) | 409 | **usable with minor issues** | Book structure produces very many small sections; front/back matter (index, series pages) yields coarse section titles. Chunk text itself is clean and page-ranged. |
| 6 | `Chapter_2_Theoretical_Foundations_Revised.docx` (thesis chapter) | 18 | **good** | Heading hierarchy preserved (`Chapter 2 … > 2.1 Purpose and scope`), tables detected and kept structured, role `future_chapters` with `evidence = no`. |
| 7 | `introduction-version-12-12-2024.docx` (own paper) | 10 | **good** | Clean section titles, 10 chunks around the target size, `retrieval = yes` / `evidence = no` (`my_work`). |
| 8 | `APR-Form-2025.docx` (supervision form) | 3 | **usable with minor issues** | A Word form: one big table, repetitive cell text, no headings, so there is no section structure to exploit. Content is preserved, split by rows within the ceiling, `evidence = no`. |
| 9 | `phd_direction_and_supervisor_plan.md` (rules) | 2 | **good** | Short sibling sections merged with `merged_sections` recorded; role `rules`, authority `instruction`, `evidence = no`. |
| 10 | `future_chapters_guidance_policy.md` (guidance) | 2 | **good** | Merged sections recorded, role `guidance`, `evidence = no`; the policy text that forbids using future chapters as evidence is itself non-evidence. |

Assessment summary: **5 good** (rows 3, 6, 7, 9, 10), **5 usable with minor
issues** (rows 1, 2, 4, 5, 8) and **0 poor**. The chunk text itself was sound in
every document; the two systematic weak spots are front-page boilerplate and
heading inference inherited from the parser, both recorded in §23.

Coverage of the §34 inspection slots: normal journal PDF → row 1; two-column PDF →
row 2; thesis/long document → row 5; DOCX thesis chapter → row 6; own-paper DOCX →
row 7; supervision/rules file → rows 8 and 9; `future_chapters` file → row 10 (plus
row 6); OCR-heavy PDF → `catmull_thesis.pdf`, inspected separately in §20a below.

### Special corpus checks (§35)

* **`catmull_thesis.pdf` — OCR provenance survives chunking.** The thesis is an
  84-page scan with no native text on any page, yet it chunks into 24 chunks /
  24267 tokens, all 24 recorded with `sources ocr`, their page range and their OCR
  locators (`page:N:ocr:1`) — see §20a for the full evidence. These 24 chunks are
  part of the 88 OCR-only chunks corpus-wide (a further 25 chunks mix native and
  OCR text).
* **Radianti heading/section/page behaviour** — “good”, see row 3 above and
  `docs/rag007-inspections.txt`.
* **`data/my_work/…` chunks** — `retrieval_eligible = yes`,
  `evidence_eligible = no` (`introduction-version-12-12-2024.docx`, 10/10 chunks;
  113/113 chunks corpus-wide, including the 53 chunks of `second-paper-draft-final.docx`,
  which is *also* filed under `literature/` and still resolves to `my_work` — §25.6).
* **`data/future_chapters/…` chunks** — role `future_chapters`,
  `authority = navigation_only`, `evidence_eligible = no` (210/210 chunks), i.e.
  the NAVIGATION_ONLY / NOT_EVIDENCE / planning-guidance semantics of §24 of the
  task brief.
* **References never enter normal evidence retrieval** — all 3317 reference chunks
  have `retrieval_eligible = no` (and `evidence_eligible = no`); the numbers match
  exactly, so no reference chunk leaked into retrieval eligibility anywhere in the
  corpus. Reference *content* is preserved in full.

### §20a. `catmull_thesis.pdf` OCR provenance (evidence)

`python -m app.chunking.inspect "data\literature\Algorithms\catmull_thesis.pdf"
--show-locators --show-text --max-chunks 2`, saved as
`docs/rag007-catmull-ocr.txt`:

```
File: data\literature\Algorithms\catmull_thesis.pdf
Extension: .pdf
Corpus role: literature
Pages: 84
Chunks: 24 (24267 tokens, tokenizer subword-estimator-v1)
Duplicate table blocks removed: 0
Running header/footer lines removed: 0
Headings without content: 0

  [  0]  1169 tokens  pages       1-6  type body   section –
        sources ocr     retrieval yes  evidence yes  role literature
        locators: page:1:ocr:1, page:2:ocr:1, page:3:ocr:1, page:4:ocr:1,
                  page:5:ocr:1, page:6:ocr:1 (sequence 0…5)
  [  1]   943 tokens  pages       7-8  type body   section –
        sources ocr     retrieval yes  evidence yes  role literature
```

Verified: the 84-page scan has **no native text**, so the document only exists for
chunking because of the OCR fallback; every chunk records `sources ocr`, the exact
one-based page range it came from, the OCR locator of each contributing parsed item
(`page:N:ocr:1`) and its position in the canonical content stream. The chunk text
is the OCR output, normalised but never rewritten: the sample shows OCR noise
preserved verbatim (`fscuily`, `Suodivision`, `ACKNOWLEOGMENTS`), which is the
intended behaviour — RAG-007 must not silently "correct" extracted text.

## 21. Every >1200-token chunk and explanation

**There are none.** The report lists zero chunks above 1200 tokens, which is the
enforced hard maximum:

* prose is accumulated paragraph by paragraph and the buffer is closed before an
  item would push it above 1200;
* a paragraph that alone exceeds 1200 tokens is split at sentence boundaries
  (target 1000 per part), with a token-exact fallback for a single pathological
  sentence;
* a table that alone exceeds 1200 tokens is split by rows;
* the merge step (§8.7) refuses any merge whose result would exceed the 1000-token
  target.

## 22. Any >1500-token violation

**None** (`chunks above 1500 tokens (failures): 0`). Since the hard maximum of 1200
is already enforced everywhere, the 1201–1500 extension band is never used in this
corpus. If a future rule allowed the extension for an intact table, the report
would list those chunks individually under “chunks above 1200 tokens” with the
reason.

## 23. Known limitations

Everything below was observed on the real corpus; none of it is hidden, and none
of it silently changes meaning.

1. **Front-page boilerplate is preserved, by design.** Running-header/footer
   removal only removes lines that *repeat* across pages. A publisher masthead,
   licence block or repository sidebar that appears once on page 1 therefore
   survives as a small preamble chunk (e.g. `+6089-.pdf`, `Radianti -main.pdf`,
   `CHI20_WalkingVibe_preprint.pdf`). Guessing that such text is not content would
   be a bigger risk than keeping it.
2. **Heading detection is inherited from RAG-003/RAG-004 and is imperfect.** Two
   observed failure modes:
   * a document with no detected headings (common in ACM/Emerald typesetting)
     keeps the last detected section, so body text can inherit an earlier
     section label; a safeguard (§9) stops an *abstract* label from spreading
     beyond its first chunk, but other labels can still be coarse;
   * unrelated sidebar text can be *detected* as a heading, e.g.
     `82 PUBLICATIONS 1,829 CITATIONS` in `CHI20_WalkingVibe_preprint.pdf` was
     treated as a numbered heading and became a section title. That is a parser
     decision (RAG-004's numbered-heading rule) which RAG-007 must not silently
     override; a future parser tweak should reject citation-count patterns.
3. **First-page author blocks and affiliations become body text** when they are
   laid out as prose (`David-John-etra22-davidjohn.pdf` produces a 55-token
   preamble chunk of names and addresses). They are retrievable but carry no
   special type.
4. **DOCX/Markdown chunks carry no `TextSource`.** RAG-003 records a page-level
   extraction source only for PDFs, and RAG-007 will not invent one. The format
   is still recorded on every chunk through `metadata_ref.filename` and the
   document's `extension`; the corpus report labels the bucket
   *“n/a (DOCX/Markdown: no page-level text source)”*.
5. **Page numbers can glue to heading text** (e.g. a book heading
   `150 Designing constructively aligned outcomes-based teaching and learning`),
   because the running-number line and the heading share one parsed block. The
   heading path is therefore accurate but noisier than ideal.
6. **Dehyphenation is conservative but not perfect.** A genuine compound broken
   exactly at its hyphen at a line end with a lowercase continuation
   (`real-` / `time`) is joined, because distinguishing it from a wrapped word
   (`educa-` / `tion`) needs a lexicon or layout geometry, both of which are out
   of scope. Compounds inside a line, and compounds with an uppercase
   continuation, are preserved; `AI-based`, `higher-order` and `real-time` typed
   inside a line are always preserved.
7. **Reference sections are found structurally.** Detection relies on a heading
   (`References`, `Bibliography`, `Works cited`, … plus common non-English
   spellings) or on a paragraph whose own first line is such a heading. A
   bibliography that carries no heading at all would be chunked as body text and
   stay retrieval-eligible. No prose-semantic reference detection was added,
   because guessing would risk mislabelling body text as a bibliography.
8. **Table de-duplication is text-based.** It removes prose blocks whose text is
   reproduced by the structured table on the same page (whole-block substring, or
   line-by-line equality with a cell). A block that differs only slightly from the
   table text (extra spaces, reordered spans) can survive, and a legitimate
   paragraph that happens to quote a table verbatim on the same page would be
   removed.
9. **Some chunks span many pages.** A 1200-token chunk can span up to 11 pages in
   a low-text-density PDF (slides, book front matter, an index); 6451 of 14786
   chunks cross a page boundary. Page order is
   never broken and the page range is always recorded, but the span is not capped
   by design: capping it would create page-sized chunks, which the task forbids.
10. **Underscore/rule lines from PDFs become tiny chunks** (a `_____` separator
    line survives as its own chunk). Removing them is not in the allowed
    normalization set.
11. **OCR text is treated as prose of unknown quality.** RAG-005 already marks it;
    RAG-007 preserves the mark (§11) but performs no repair, and figure/equation
    OCR text is chunked as ordinary text while remaining marked as OCR.
12. **XLSX is not chunked** (one file in the corpus). It is reported as
    unsupported for chunking in RAG-007 rather than flattened; see §16.

## 24. Architectural concerns

1. **Domain model now reuses parsing/metadata vocabularies.**
   `app/models/chunk.py` imports `TextSource` from `app.parsing.models` and
   `CorpusRole`/`PublicationType` from `app.metadata.enums`. That is a dependency
   from the domain layer onto the parsing/metadata layers; the alternative would
   have been duplicating controlled vocabularies, which would break the
   "never collapse provenance" principle. `app.parsing.models` and
   `app.metadata.enums` import nothing from `app.models`, so there is no cycle.
2. **`Chunk.document_id` is now deterministic, while RAG-002's `Document.id` is
   not yet.** RAG-002 assigns a random UUID when a `Document` is constructed, but
   no stage currently builds corpus `Document` objects, so nothing conflicts
   today. RAG-007 derives `document_id = uuid5(namespace, file_hash)`
   (`document_id_for`). **When the indexing stage builds `Document` records it
   must use the same derivation**, otherwise chunk → document links will not
   match. This is the one place where RAG-007 constrains a later task, and it is
   flagged rather than resolved silently.
3. **Corpus role is resolved from all physical locations** *(fixed in §25)*.
   Deduplication is by SHA-256, and a file that exists in both `literature/` and
   `my_work/` (e.g. `second-paper-draft-final.docx`) used to take the role of the
   first path discovered — alphabetical order put `literature` first, so such a
   draft was classified as *literature* and became evidence-eligible. Role
   resolution now looks at every physical location and the most conservative role
   wins, so the draft is `my_work` / not evidence regardless of discovery order.
   RAG-007 was the right place to fix this: it is the stage that turns a role into
   an evidence policy, and it was the stage whose output would have been wrong.
4. **Zero-overlap chunking is deliberate.** §13 of the task rules out blind token
   overlap; RAG-007 records `overlap_tokens = 0` on every chunk and relies on the
   carried `section_title` / `heading_path` / `metadata_ref` for context, and on
   `retrieval_text` for the injected `[Document: …] [Section: …]` labels. RAG-008
   should embed `retrieval_text`, never `text`, so that quote-capable source text
   is never mistaken for document text.
5. **Token counts are estimator-based.** Boundaries are defined by
   `subword-estimator-v1`. If RAG-008 validates chunk sizes with BGE-M3's real
   tokenizer the counts will differ (typically ±20 %), so the recommendation is to
   use the model tokenizer as a *measurement* and keep the estimator as the
   boundary rule, or to re-run chunking with a pluggable `Tokenizer`
   implementation if exact model alignment is required. The `Tokenizer` protocol
   exists for exactly that substitution.
6. **Content-type semantics are conservative in one direction.** A chunk whose
   section is not identifiable keeps `BODY`; only `REFERENCES` (never retrieval
   eligible) and the roles of §14 can make a chunk ineligible. Mis-labelling a
   body chunk as `ABSTRACT` was explicitly prevented (§9), because that would
   have made body text look like a summarised claim.
7. **No chunk is anonymous text.** Every chunk carries its page range, heading
   path, source locators, sequence range, text sources, corpus role, evidence
   policy and a minimal bibliographic reference; `sequence_start`/`sequence_end`
   plus `source_locator` locate the exact parsed items, and the raw text remains
   in `ParsedDocument.raw_text` / `ParsedPage.native_text` / `ParsedPage.ocr_text`
   for auditing any normalization.

## 25. Correction: deduplicated document role safety

### 25.1 The defect

A logical document is one set of identical bytes (SHA-256), but it can exist in
several folders. RAG-002 deduplicated by hash while RAG-006 derived the corpus role
from the **canonical (first-discovered) path**, so a draft that also lived under
`literature/` was classified as literature and its chunks became
`evidence_eligible = true` — even though the same bytes sat under `my_work/`. The
outcome therefore depended on alphabetical discovery order, which is not a
semantic signal.

### 25.2 Resolution from all physical locations

`app/metadata/corpus.py` gained `resolve_corpus_role(paths)`: every physical
location of the logical document contributes the role its path implies, and the
effective role is the first of those present in

```
my_work > rules > supervision > future_chapters > summaries > guidance > literature > other
```

`ROLE_PRECEDENCE` is *derived* from the existing `_ROLE_FOLDERS` detection order
(plus `other` last), so detection and resolution share one documented ordering
rather than two that could drift apart. Every non-literature role precedes
`literature`, which is exactly the required rule:

```
my_work / summaries / supervision / rules / guidance / future_chapters
        all beat literature
```

`extract_metadata(..., physical_paths=[…])` resolves the role from all locations;
an explicit `corpus_role` argument still overrides everything (and is recorded as
`explicit corpus_role override`). All corpus-wide stages now pass every location:
`app/chunking/report.py`, `app/metadata/report.py`, `doi_stage.py` and
`title_stage.py`. Paths are deduplicated and sorted before resolution, so the
physical-location order cannot influence the result.

### 25.3 Evidence policy follows the resolved role

Because the chunk policy is derived from `metadata.corpus_role`, the fix propagates
automatically: for a draft whose locations are

```
data/literature/Second paper/second-paper-draft-final.docx
data/my_work/current_paper/second-paper-draft-final.docx
```

the effective role is `my_work`, so all 53 chunks are
`retrieval_eligible = true` and `evidence_eligible = false` (authority `high`),
even though one physical copy is filed under `literature`. No self-citation
evidence mode was introduced: `my_work` stays retrievable and non-evidence.

### 25.4 Provenance

Nothing about the copies is discarded, and one SHA-256 document stays one logical
document (`scan_directory` still reports `unique_documents = 1` for the pair, and
only one chunk set exists — verified in §25.6).

* `AcademicMetadata.role_resolution` is a `RoleResolution` record holding
  `effective_role`, `reason`, `physical_roles` (path **and** role for every
  location, sorted) and `applied_precedence` (the ordering that was applied).
  Its `describe()` produces, for the real case:
  `my_work (conservative precedence: my_work preferred over literature; locations=2, roles=literature+my_work)`.
* `Chunk.corpus_role` stays singular (the effective role), and
  `ChunkMetadataRef.duplicate_locations` / `.role_resolution` carry the same audit
  trail on every chunk — but only for multi-location documents, so the ordinary
  single-copy chunk is unchanged.
* Both development CLIs print the resolution (and the per-location roles when they
  disagree); `python -m app.metadata.inspect … --corpus data` and
  `python -m app.chunking.inspect … --corpus data` find the duplicates by hash.
  Evidence: `docs/rag007-role-resolution.txt`.

### 25.5 Regression tests

`tests/test_chunking_duplicate_roles.py` (12 tests):

| # | Test | Requirement |
|---|------|-------------|
| 1 | `test_only_literature_is_evidence` | only literature → literature / evidence yes |
| 2 | `test_only_my_work_is_not_evidence` | only my_work → my_work / evidence no |
| 3 | `test_duplicate_in_literature_and_my_work_prefers_my_work` | same hash literature + my_work → my_work / evidence no |
| 4 | `test_duplicate_in_literature_and_summaries_is_not_evidence` | same hash literature + summaries → non-evidence |
| 5 | `test_duplicate_in_literature_and_future_chapters_is_not_evidence` | same hash literature + future_chapters → non-evidence |
| 6 | `test_discovery_order_does_not_change_the_effective_role` | reversed discovery order → identical role, policy, chunk ids and chunk order |
| 7 | `test_physical_locations_stay_inspectable` | all paths, roles, reason and precedence remain readable (document and chunk) |
| 8 | `test_duplicate_copies_produce_one_logical_document` | one logical document, one chunk set, unchanged totals; report audit row before/after |
| 9 | `test_inspect_cli_resolves_role_across_corpus` | the `inspect --corpus` route resolves duplicates |
| 10 | `test_chunk_without_metadata_keeps_path_derived_role` | chunking without metadata still derives the role from the path |
| 11 | `test_every_non_literature_role_beats_literature` | the precedence ordering itself |
| 12 | `test_doi_stage_gate_uses_all_locations` | the same rule protects the RAG-006 enrichment gate (§25.2) |

No RAG-001…RAG-007 test was weakened: the suite grew from 350 to **362** passing
tests. Test 12 was checked against the pre-fix code and fails there (the duplicate
`summaries` + `literature` document was treated as enrichable and made two HTTP
requests), so it is a real regression test rather than a restatement of the new
behaviour.

### 25.6 Real-corpus validation

The scan of `data/` (466 unique documents, 36 duplicate copies) gives:

```
logical documents with more than one physical copy: 34
logical documents whose copies imply >1 role:        2
```

Both multi-role documents:

| Filename | Hash (12) | Physical roles | Effective role before → after | Evidence before → after |
|----------|-----------|----------------|-------------------------------|--------------------------|
| `second-paper-draft-final.docx` | `fccd43a1833c` | `literature` (`data\literature\Second paper\…`) + `my_work` (`data\my_work\current_paper\…`) | `literature` → **`my_work`** | `true` → **`false`** |
| `alogirithms-8-enriched.xlsx` | `e5683a86d824` | `literature` (`data\literature\Second paper\…`) + `my_work` (`data\my_work\current_paper\…`) | `literature` → **`my_work`** | `true` → **`false`** (no chunks: XLSX is not chunked) |

Explicit verification of the required case
(`docs/rag007-role-resolution.txt`):

```
$ python -m app.chunking.inspect "data\literature\Second paper\second-paper-draft-final.docx" --corpus data
File: data\literature\Second paper\second-paper-draft-final.docx
Corpus role: my_work
Role resolution: my_work (conservative precedence: my_work preferred over literature; locations=2, roles=literature+my_work)
  literature       data\literature\Second paper\second-paper-draft-final.docx
  my_work          data\my_work\current_paper\second-paper-draft-final.docx
Document id: d8919983-e93b-5ce4-ab53-7fb470c99f8f
Chunks: 53 (44694 tokens)
  [  0]  … retrieval yes evidence no  role my_work
```

So `retrieval_eligible = yes`, `evidence_eligible = no`, `effective role = my_work`
— from **either** physical path, with the same document id and the same chunks.

Recalculated corpus totals (full re-run, `docs/rag007-chunking.txt`):

```
chunk count by corpus role      literature 13741   my_work 113   (others unchanged)
evidence-eligible chunks        10456   (was 10509)
retrieval-eligible chunks       11469   (unchanged)
total chunks                    14786   (unchanged)
```

The only movement is the 53 chunks of `second-paper-draft-final.docx`, which
switched from `literature`/evidence to `my_work`/non-evidence. Exactly as required,
**the total number of logical chunks did not change**: role resolution changed
*which* policy the chunks carry, not how the text was cut, and no chunk was
duplicated for the second physical copy. The remaining 32 duplicate documents agree
on their role (`all N physical locations agree`), so their policy is unchanged.

The corpus report now prints this audit itself, so a reviewer does not have to
trust the prose above:

```
Duplicate copies and corpus roles
---------------------------------
documents with more than one physical copy: 34
documents whose copies imply >1 role:       2
  second-paper-draft-final.docx  (fccd43a1833c…)
    location: literature (…\data\literature\Second paper\second-paper-draft-final.docx)
    location: my_work (…\data\my_work\current_paper\second-paper-draft-final.docx)
    effective role: literature -> my_work
    evidence eligible: True -> False
    reason: conservative precedence: my_work preferred over literature
```

### 25.7 Determinism

* `resolve_corpus_role` sorts and deduplicates the paths and applies a fixed
  precedence, so reversing discovery order changes nothing (test 6 asserts the
  effective role, the evidence policy, the chunk ids and the chunk order for a
  corpus where the `my_work` copy is discovered first).
* Chunk ids remain `uuid5(namespace, file_hash:chunk_index:sha256(text))`: paths and
  roles are **not** part of the id, so the same bytes always produce the same ids —
  the duplicated draft has the same 53 ids under either canonical path.
* `ChunkMetadataRef` gained no timestamp or unordered collection, so the payload
  stays deterministic; `duplicate_locations` is sorted.

## Final state (acceptance criteria, §37)

| Criterion | Verified |
|-----------|----------|
| all existing 276 tests pass | ✅ `362 passed` = 276 pre-existing + 86 new, no existing test edited or skipped |
| new RAG-007 tests pass | ✅ 86 tests across 7 files (§4) |
| no chunk >1500 tokens | ✅ 0 violations, max observed 1200 (`docs/rag007-chunking.txt`) |
| references excluded from normal retrieval | ✅ 3317/3317 reference chunks `retrieval_eligible = no`, and no other chunk is ineligible |
| exact page provenance retained | ✅ `page_start`/`page_end` + `page:N:block:M` locators; 6451 chunks cross pages without losing ranges |
| OCR provenance retained | ✅ 113 chunks marked OCR, 25 marked mixed; `catmull_thesis.pdf` fully OCR (§20a) |
| corpus roles retained | ✅ every chunk carries the RAG-006 `corpus_role`, now resolved from **all** physical locations (§25); distribution in §19 |
| evidence eligibility retained | ✅ only literature chunks are evidence-eligible; `my_work`/`future_chapters`/`summaries`/`rules`/`guidance`/`supervision` are not — including a draft that also exists under `literature/` (§25.6) |
| duplicate copies cannot promote evidence | ✅ 2 multi-role documents resolved conservatively, 0 chunk-count change, 1 logical document per hash (§25.6) |
| deterministic chunk IDs | ✅ `uuid5` over file hash + index + content hash; no random UUIDs, no clock, no path in the id |
| no source file modified | ✅ chunking only reads parsed content; every file under `data/` is unchanged (newest mtime 2026-10-10, this run 2026-10-11) |
| no embeddings / Qdrant / LLM usage | ✅ none, and no new dependency (§3) |

```
RAG-007 COMPLETE
```

Artifacts: `docs/rag007-chunking.txt`, `docs/rag007-inspections.txt`,
`docs/rag007-catmull-ocr.txt`, `docs/rag007-role-resolution.txt`,
`docs/rag007-chunking-pre-rolefix.txt`, `RAG-007-report.md`. RAG-008 was not started.
