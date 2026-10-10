# RAG-004 — End-of-Task Report

**Project:** PhD Academic RAG
**Task ID:** RAG-004 — Faithful PDF parsing with exact page provenance
**Status:** Complete — awaiting review
**Date:** 2026-10-10

---

## 1. Files created

| File | Responsibility |
|------|----------------|
| `app/parsing/pdf_parser.py` | `parse_pdf(path, file_hash=None)` — page-by-page PyMuPDF extraction |
| `tests/test_pdf_parser.py` | 22 PDF tests (synthetic PDFs only) |

## 2. Files modified

| File | Change |
|------|--------|
| `app/parsing/models.py` | `TextExtractionStatus`, `ParsedPage`; `page_count` + `pages` on `ParsedDocument`; `page_number` on `ParsedBlock`/`ParsedTable` |
| `app/parsing/exceptions.py` | Added `EncryptedDocumentError` |
| `app/parsing/parser.py` | `.pdf` → `parse_pdf` (recreated cleanly after a tooling hiccup) |
| `app/parsing/parse.py` | `Pages`, `Empty pages`, `Possibly scanned pages`; new `--show-pages N` |
| `tests/test_parser_dispatcher.py` | Removed `file.pdf` from the "unsupported" cases (now supported) |

RAG-001/002/003 behaviour is unchanged; DOCX/Markdown models gained only optional, defaulted fields.

## 3. Dependencies added

None. PyMuPDF (`pymupdf 1.28.2`) was already installed.

## 4. Final pytest result

```
143 passed, 1 warning in 1.74s
```

## 5. Exact PDF extraction strategy

- `pymupdf.open(path)` (read-only), then `document.load_page(i)` **one page at a time**.
- Per page: `page.get_text("dict")`; only `type == 0` (text) blocks are used; image blocks are skipped.
- Block text is rebuilt from `lines → spans` and joined with `\n` (line breaks preserved), then stripped. Empty blocks are skipped.
- Unicode, Greek, punctuation, symbols, citations, and hyphenation are preserved verbatim (no de-hyphenation, no header/footer removal).
- The document is closed promptly (`finally`).

## 6. Exact page provenance representation

- Every `ParsedBlock` and `ParsedTable` carries `page_number` (**1-based**; PyMuPDF's 0-based index is never exposed).
- Deterministic `source_locator`:
  - blocks: `page:{page_number}:block:{global_block_index}`
  - tables: `page:{page_number}:table:{global_table_index}`
- `ParsedDocument.pages: list[ParsedPage]` **and** flattened `blocks` / `tables` / `content_items` are both populated (the page blocks are the *same objects* as the flattened ones, verified by test).
- `ParsedDocument.page_count` is set.

## 7. Exact reading-order strategy

PyMuPDF's native block order from `get_text("dict")` is used as-is — it is deterministic and generally follows natural reading order. **No custom layout engine** and no bbox re-sorting were added (re-sorting by y/x can scramble multi-column papers). Two-column ordering is therefore only as good as PyMuPDF's own ordering — see limitations.

## 8. Raw-text page-boundary format

Pages are separated by an explicit marker (deterministic):

```
[PAGE 1]
<page 1 text>

[PAGE 2]
<page 2 text>
```

`raw_text` is these page chunks joined with `"\n\n"`. Page text is the page's block texts joined with `"\n\n"`. Tables are **not** re-appended to `raw_text` (their text is already present in the page blocks), avoiding duplication.

## 9. PDF title behaviour

- Title comes **only** from `document.metadata["title"]` when non-empty after trimming.
- Otherwise `None`. No first-page layout guessing, no font-size heuristics.

## 10. Table extraction behaviour

- Uses PyMuPDF's built-in `page.find_tables()` (best-effort). Each detected table → `ParsedTable` with `table_index`, `sequence_index`, `page_number`, `rows`/`cells`, `heading_path`, and locator `page:{n}:table:{i}`.
- If extraction fails or is unavailable, the table's visible text still exists in the page blocks — **nothing is invented and no large dependency (Camelot/Tabula/Java/OCR) was added**.

## 11. Heading inference behaviour

Conservative and deterministic, `heading_source = INFERRED`:
- single-line block, ≤ 90 chars, ≤ 12 words, not ending in `.`/`!`/`?`/`;`/`,`;
- matches a numbered pattern (`1`, `2.1`, `3.1.2`) with an uppercase/digit title, or a recognised name (`References`, `Bibliography`, `Works Cited`, `Conclusion`, `Introduction`, `Abstract`, `Methods`, `Results`, `Discussion`, …).
- No font size, no boldness, no LLM. Many PDFs legitimately yield zero headings. Heading hierarchy (`heading_path`) is maintained across pages.

## 12. Empty / scanned-page detection behaviour

Each `ParsedPage` gets `text_extraction_status`:
- `EMPTY` — no extractable text **and** no images;
- `POSSIBLY_SCANNED` — no text **but** page has images, or very little text (< 50 chars) plus images;
- `TEXT` — otherwise.

No OCR is performed.

## 13. Sample real-corpus validation (read-only, 10 PDFs)

Selected: largest PDF in each literature folder + largest overall until 10.

| File | Pages | Title | Blocks | Headings | Tables | Chars | Empty | Scanned |
|------|-------|-------|--------|----------|--------|-------|-------|---------|
| Generative-ai-higher-education-book.pdf | 287 | Generative AI in Higher Education; The ChatGPT Effect | 4104 | 189 | 1 | 844901 | 0 | 5 |
| Pseudo-walking_...Foot-sole_Vibrations.pdf | 12 | Pseudo-walking Sensation by … | 439 | 1 | 0 | 75856 | 0 | 0 |
| 978-3-030-84044-0.pdf | 316 | — | 4069 | 16 | 46 | 952157 | 0 | 0 |
| kerbl-2023.pdf | 15 | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | 411 | 2 | 6 | 79952 | 0 | 0 |
| Patney-2980179.2980246.pdf | 13 | Towards foveated rendering for gaze-tracked virtual reality | 507 | 4 | 7 | 73789 | 0 | 0 |
| Zhu-1703.10593v7.pdf | 18 | — | 339 | 15 | 6 | 54986 | 0 | 0 |
| kerbl-2308.04079v1.pdf | 14 | 3D Gaussian Splatting … | 397 | 2 | 6 | 78990 | 0 | 0 |
| kerbl.pdf | 14 | 3D Gaussian Splatting … | 397 | 2 | 6 | 78990 | 0 | 0 |
| Lin-2211.10440v2.pdf | 18 | — | 354 | 13 | 3 | 61838 | 0 | 0 |
| tewari-2020.pdf | 27 | State of the Art on Neural Rendering | 619 | 54 | 2 | 181365 | 0 | 0 |

**0 parse errors**, 0 encrypted, 0 corrupt. Tables were extracted from multi-column academic papers (e.g. 46 tables in a 316-page text).

## 14. Corpus-wide health check (read-only, all PDFs)

```
PDFs attempted:              478
PDFs successfully parsed:    478
PDFs failed:                 0
total pages:                 11490
pages with text:             11304
empty pages:                 20
possibly scanned pages:      166
encrypted PDFs:              0
corrupt PDFs:                0
PDFs with metadata title:    246
PDFs without metadata title: 232
PDFs with little text (<500 chars): 5
```

## 15. Failed / encrypted / problematic PDFs

- **Failed:** none. **Encrypted:** none. **Corrupt:** none.
- **Possibly-scanned pages:** 166 across the corpus (≈1.4% of pages) — the OCR decision input. These are concentrated in **19 PDFs**:

| Scanned / total pages | File |
|---|---|
| 84 / 84 | `Algorithms\catmull_thesis.pdf` (entirely image-only) |
| 20 / 23 | `Teaching\Experiential_Learning_Experience_As_The_Source_Of_.pdf` |
| 13 / 362 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` |
| 11 / 14 | `AI\8-AI_and_VR_Enabled_Modern_LMS_for_Student.pdf` |
| 10 / 10 | `Algorithms\Lin-View of Design and Application of Intelligent Assisted Learning System for Higher Education Students.pdf` |
| 6 / 6 | `AI\View of Challenges with Gamification in Higher Education.pdf` |
| 5 / 287 | `AI\Generative-ai-higher-education-book.pdf` |
| 2 / 28 | `AI\DEMOCRATISING_AI_final (2).pdf` |
| 2 / 10 | `Algorithms\perlin.pdf` |
| 2 / 418 | `Teaching\-John_Biggs_and_Catherine_Tang-_Teaching_for_Quali-BookFiorg-.pdf` |
| 2 / 28 | `VR\DEMOCRATISING_AI_final (2).pdf` |
| 2 / 2 | `VR\FundamentalVR_ValidationFeb2019_BOTA.pdf` |
| 1 / 50 | `AI\2021_UNESCO_AI-and-education-Guidande-for-policy-makers_EN.pdf` |
| 1 / 86 | `Algorithms\sellers.pdf` |
| 1 / 22 | `Teaching\Formative_Assessment_and_Self-Regulated_Learning_A.pdf` |
| 1 / 28 | `Teaching\LS_OverView.pdf` |
| 1 / 27 | `VR\Maister_et_al_TICS_authors_version_Figures.pdf` |
| 1 / 3 | `rules\plagiarism.pdf` |
| 1 / 67 | `rules\Writing-a-Dissertation.pdf` |

  Only **one** document is fully image-only (`catmull_thesis.pdf`); the rest have a small minority of unreadable pages.
- **PDFs with suspiciously little text (<500 chars extracted):**
  - `data\literature\AI\View of Challenges with Gamification in Higher Education.pdf`
  - `data\literature\Algorithms\Lin-View of Design and Application of Intelligent Assisted Learning System for Higher Education Students.pdf`
  - `data\literature\Teaching\_cbebd8b56e2c84f3ba31404a84542b37_Lesson-plan-template.pdf`
  - `data\literature\VR\FundamentalVR_ValidationFeb2019_BOTA.pdf`
  - `data\rules\plagiarism.pdf`

## 16. Important parsing limitations

1. **Two-column ordering** relies entirely on PyMuPDF's block order; some papers may interleave columns. A layout engine was deliberately not added.
2. **Headers/footers, page numbers, DOI banners** are preserved in the text (§10 of the brief).
3. **Line-break hyphenation** (`educa-\ntion`) is preserved, not rewritten.
4. **Table text may appear twice**: once in the page blocks and once as a structured table. This is intentional (raw fidelity first); de-duplication belongs to a later stage.
5. **Whole-file failure on a page error**: if a single page's text extraction raises, the whole PDF raises `DocumentParseError` (no partial document). No such case occurred in the corpus.
6. **NBSP**: synthetic PDFs produced by PyMuPDF's `TextWriter` encode spaces as U+00A0, so extraction can yield `\xa0`. Real publisher PDFs generally use normal spaces; a later normalization stage may want to fold NBSP.
7. **`find_tables()` noise**: PyMuPDF prints a one-off layout-package advertisement; this is now suppressed via `pymupdf.no_recommend_layout()`.

## 17. Architectural concerns / ambiguities

1. **`data\literature\Second paper\` contains no unique PDFs.** It was empty during validation; as of 2026-10-10 it holds `alogirithms-8-enriched.xlsx` and `second-paper-draft-final.docx`, both **binary-identical duplicates** of the copies in `data\my_work\current_paper\` (confirmed by RAG-002 SHA-256 scan: 0 new unique documents). Worth confirming where the second-paper source material actually lives.
2. **232 / 478 PDFs (49%) have no metadata title** — meaning RAG-006 metadata work will be needed for roughly half the evidence base.
3. **166 possibly-scanned pages** exist; we now have the data to decide whether an OCR layer is worth building.
4. **Page model added cleanly**: `ParsedPage` + optional `page_count` did not conflict with the RAG-003 models, so no redesign was needed.

---

## Scope confirmation

- No OCR, embeddings, chunking, Qdrant, Ollama, internet APIs, citation generation, or metadata enrichment implemented.
- All parsing is strictly read-only; no source file was modified.
- `GET /health`, RAG-002 scanning, and DOCX/Markdown parsing are unchanged.
- `python -m app.parsing.parse "paper.pdf"` works.
- RAG-005 has **not** been started.
