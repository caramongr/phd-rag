# RAG-005 — End-of-Task Report

**Project:** PhD Academic RAG
**Task ID:** RAG-005 — Selective OCR fallback for scanned PDF pages
**Status:** Complete. Tesseract 5.4.0 installed; **full-corpus selective OCR run finished — 478 PDFs / 11,490 pages inspected, 166 OCR-eligible pages, 162 recovered, 0 failures** (see §12). 163 tests pass. RAG-006 **not** started.
**Default OCR language:** changed from `eng+ell` to **`eng`** (Greek `ell` is not installed in the local Tesseract).
**Date:** 2026-10-10
**Run configuration:** `ocr_language=eng`, `ocr_dpi=300`, `ocr_min_native_chars=50`, `ocr_enabled=true`

---

## 1. Files created

| File | Responsibility |
|------|----------------|
| `app/parsing/ocr.py` | Tesseract wrapper (lazy import), page rendering, diagnostics, `OcrPageResult` production |
| `tests/test_ocr.py` | OCR tests (engine mocked — suite never needs Tesseract) + CLI metric regression tests |

## 2. Files modified

| File | Change |
|------|--------|
| `app/config.py` | OCR settings: `ocr_enabled`, `ocr_language`, `ocr_dpi`, `ocr_min_native_chars` (env `PHD_RAG_OCR_*`); `DEFAULT_OCR_LANGUAGE` changed `eng+ell` → **`eng`** |
| `app/parsing/models.py` | `TextSource` enum, `OcrPageResult` model, extended `ParsedPage` (`native_text`, `ocr_text`, `text_source`, `ocr_applied`, `ocr_language`, `ocr_error`), `ParsedBlock.text_source` |
| `app/parsing/pdf_parser.py` | Page-level OCR fallback, per-page OCR error handling, configurable threshold |
| `app/parsing/parser.py` | `parse_document(..., ocr=...)` passthrough |
| `app/parsing/parse.py` | `Native text pages` / `OCR pages` / `OCR failures` / `Pages with no text` lines, `--no-ocr`, `--ocr-info` |
| `tests/test_ocr.py` | New `test_default_ocr_language_is_english` locking the `eng` default |
| `RAG-005.md` | §3/§14 language configuration updated to the `eng` default |

## 3. Dependencies added

Installed into the project venv: **`pytesseract 0.3.13`** and **`Pillow 12.3.0`**. Both are imported lazily, so the rest of the system and the full test suite still run without Tesseract present.

## 4. External executable required

**Tesseract OCR** (offline) — **now installed and detected**. `python -m app.parsing.parse --ocr-info` reports:

```
Tesseract: available
Executable: C:\Program Files\Tesseract-OCR\tesseract.exe
Version: 5.4.0.20240606
Languages:
- eng
- osd
```

### Installation record

1. Python wrappers: `pytesseract 0.3.13`, `Pillow 12.3.0`.
2. Tesseract OCR for Windows (UB Mannheim build) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (detected via the built-in common-path probe; no `TESSERACT_CMD` needed).
3. Language packs present: **`eng`** and **`osd`** only. **`ell` (Greek) is NOT installed**, so `eng+ell` can no longer be the default.
4. Verified: `--ocr-info` prints `Tesseract: available`, the version, and the installed languages.

**Default language change:** `DEFAULT_OCR_LANGUAGE` is now **`eng`** (was `eng+ell`). Greek OCR is not required for this corpus; `PHD_RAG_OCR_LANGUAGE` can still be set explicitly (e.g. `eng+ell`) once `ell` data is installed. No language pack is downloaded automatically.

## 5. Final pytest result

```
163 passed, 1 warning in 2.24s
```

Includes RAG-001, RAG-002, RAG-003, RAG-004 (all unchanged), and RAG-005 (163 = 162 + `test_default_ocr_language_is_english`). Re-run after the language default change and again after the full-corpus OCR run — no regressions; the OCR suite mocks the engine, so the corpus run did not affect it.

## 6. OCR engine / version

Tesseract via the pytesseract wrapper (**`pytesseract 0.3.13`**). **Tesseract engine version: 5.4.0.20240606** (`C:\Program Files\Tesseract-OCR\tesseract.exe`).

## 7. Installed OCR languages detected

Detected: **`eng`, `osd`**. Greek (`ell`) is not installed. Configuration default is **`eng`**; language packs must be installed manually (never downloaded automatically).

## 8. Exact OCR eligibility rule

OCR runs **only** when OCR is enabled **and**:

```
page.text_extraction_status == POSSIBLY_SCANNED
  OR (page has images AND native_character_count < ocr_min_native_chars)
```

- `ocr_min_native_chars` defaults to **50** and is **configurable** (settings/env), used both for page classification and eligibility.
- Ordinary `TEXT` pages and text-only pages are never OCR-ed. A page is a plausible scan only if it contains images.

## 9. Exact rendering resolution

`ocr_dpi = 300` default. Rendering uses PyMuPDF `Matrix(zoom, zoom)` with `zoom = dpi / 72`, producing an in-memory PNG (`page.get_pixmap(...).tobytes("png")`). Pages are rendered **one at a time**; no temp image files are written; the pixmap is released immediately.

## 10. Exact text-source provenance representation

- `TextSource` enum: `NATIVE_PDF` (`"native_pdf"`) / `OCR` (`"ocr"`).
- `ParsedBlock.text_source`: every PDF block is `NATIVE_PDF` or `OCR` (DOCX/Markdown blocks leave it `None`).
- `ParsedPage`: `native_text` (original PyMuPDF text, never destroyed), `ocr_text`, `text_source` (effective), `ocr_applied`, `ocr_language`, `ocr_error`.
- Effective page text (`raw_text`) = OCR text when OCR succeeded and produced non-empty text, otherwise native text. Native and OCR text are **never concatenated**.
- OCR debug blocks are retained alongside native blocks; `source_locator` format `page:{n}:ocr:1`.

## 11. OCR failure handling

- Each eligible page is wrapped in its own `try/except Exception`. Failures (including `OcrUnavailableError` and `OcrError`) are recorded in `ParsedPage.ocr_error` (`"<ExceptionType>: <message>"`) and parsing **continues**.
- Native extraction is preserved; the page falls back to `text_source = NATIVE_PDF`.
- This is intentionally different from native parse failure, which still raises `DocumentParseError` for the whole document.

Demonstration of the OCR failure path on a real 2-page scanned PDF (OCR enabled, Tesseract binary still absent at the time). Kept as the recorded baseline for this behaviour:

```
Pages: 2
Native text pages: 0
OCR pages: 0
OCR failures: 2
Pages with no text: 2
Possibly scanned pages: 2
OCR note: OcrUnavailableError: OCR unavailable: Tesseract executable not found. ...
```

With Tesseract now installed, the same file recovers normally — see §12 (`FundamentalVR_ValidationFeb2019_BOTA.pdf`: 2/2 pages recovered, 0 failures).

### CLI metric correction (regression test added)

The `Native text pages` metric was misleading: it counted any page whose
*effective source* was native, including genuinely image-only pages that
produced **zero** text after an OCR failure. It now counts only pages where
native extraction produced usable text (`text_source == NATIVE_PDF` **and**
non-empty text), and a new `Pages with no text: N` line makes the all-scan case
explicit.

- Before: `Native text pages: 2` (both pages had 0 characters).
- After: `Native text pages: 0`, `Pages with no text: 2`.
- Regression tests: `test_cli_native_text_pages_excludes_image_only_pages`,
  `test_cli_native_text_pages_counts_real_native_text`.
- **Extraction architecture unchanged** (reporting only).

## 12. Real-corpus OCR validation results

### 12.1 Four-file sample (pre-approval)

Read-only OCR validation, real corpus, **OCR language `eng`, DPI 300**, Tesseract 5.4.0. Timings are wall-clock for the full parse of each file (native extraction + eligibility + OCR).

| File | Total pages | Native text pages | Eligible OCR pages | OCR pages recovered | OCR failures | Pages still without usable text | Total OCR chars | Processing time |
|------|------------:|------------------:|-------------------:|--------------------:|-------------:|--------------------------------:|----------------:|----------------:|
| `Algorithms\catmull_thesis.pdf` | 84 | 0 | 84 | 83 | 0 | 1 | 78,227 | 56.94 s |
| `Teaching\Experiential_Learning_Experience_As_The_Source_Of_.pdf` | 23 | 3 | 20 | 20 | 0 | 0 | 56,278 | 23.25 s |
| `Algorithms\Lin-View of Design and Application of Intelligent Assisted Learning System for Higher Education Students.pdf` | 10 | 0 | 10 | 10 | 0 | 0 | 40,128 | 18.25 s |
| `VR\FundamentalVR_ValidationFeb2019_BOTA.pdf` | 2 | 0 | 2 | 2 | 0 | 0 | 8,334 | 4.32 s |
| **Total** | **119** | **3** | **116** | **115** | **0** | **1** | **182,967** | **102.76 s** |

Notes:

- `Native text pages` counts pages whose *native* extraction produced usable text; `Eligible OCR pages` counts pages OCR was attempted on (all of them were `POSSIBLY_SCANNED`).
- **Zero OCR failures** across all 116 eligible pages — Tesseract was reachable and every page rendered/OCR-ed successfully.
- The single "page still without usable text" is `catmull_thesis.pdf` **page 84**, which is genuinely **blank**: its scanned image is a near-uniform page (pixel min 237 / max 255, mean 255; 0 dark cells in a coarse render) and native text is empty. OCR ran and correctly returned no text — this is a real blank page, not an OCR failure.
- Effective provenance behaved as designed: 83 catmull pages became `text_source = OCR`; the 3 text pages of the Experiential document stayed `NATIVE_PDF` and were never OCR-ed.
- No document text is reproduced in full anywhere; only short snippets (below) are used to judge quality.

### `catmull_thesis.pdf` OCR quality inspection

Sampled the beginning (p.1), middle (p.43), and end (p.83; p.84 is blank) plus additional pages (2, 5, 12, 25, 60, 70, 80) and whole-document statistics.

Whole-document OCR statistics: 78,310 chars, 13,378 whitespace tokens, 10,560 all-alphabetic tokens (78.9%), letter ratio 0.755, digit ratio 0.014, 304 non-ASCII chars, average token length 4.79.

- **Beginning — p.1 (title page, 303 chars):** essentially clean, e.g. `A SUBDIVISION ALGORITHM FOR COMPUTER DISPLAY OF CURVED SURFACES ... Edwin Earl Catmull ... Department of Computer Science / University of Utah / December 1974` — one error: `fscuily` for `faculty`.
- **Middle — p.43 (prose, 1,637 chars):** readable continuous prose with recurring character-substitution errors, e.g. `is lese than` (less), `stored sicture` (picture), `Cvery time` (Every), `subdivicsd` (subdivided), `that crea` (area), `ie. pieces a, o, and c` (i.e. a, b, c), `If oniy one` (only). ≈8 substantive word errors out of ≈290 words (~3%).
- **Middle — p.60 (prose + matrix math, 1,121 chars):** prose fine (`univariate cass` = case, `nearky` = nearly, `soma four-by-four` = some, `b'variate` = bivariate), but the equations degrade badly: `r P; / bla MIPal / ¢ P,`.
- **End — p.83 (Vita, 279 chars):** poor — `Sirthplace` (Birthplace), `Desrses` (Degree), `Uteh` (Utah), `Ssierca` (Science), `Universily`, `Farkersburg` (Parkersburg), plus a garbled header `+e eee scent sme ame`.
- Other sampled pages: p.2 committee-approval page poor (`Chaiunan, Supervisory Commitice`, `fopefd`, `cead`, `Stepkénson`); p.12 and p.80 (diagrams/figure captions) poor (`Raster-elament`, `KET,`); p.5 TOC usable with noise (`Chapier 7`, `Suodivision`, `Genera!`); p.25/p.70 mixed prose + garbled formulas.

**Classification: `usable with noise`.** Prose pages are readable and semantically recoverable at roughly 95–97% word accuracy, with systematic Tesseract confusions (`c`↔`e`, `t`↔`c`, `r`↔`i`, `m`↔`un`/`rn`, `y`↔`i`, `!`↔`l`), consistent with a 1974 typewriter-typescript scan. Quality degrades to **poor** on mathematical equations, figure/diagram labels, cover-page rubber stamps, and the Vita page, where output is not usable without manual correction.

Practical implication: OCR text is good enough to make these scanned documents *findable* (topical/lexical retrieval), but downstream chunks built from the affected pages will contain character-level noise, and equations/figure labels should not be trusted verbatim.

### 12.2 Corpus-wide run (full corpus — approved)

Read-only run over every PDF under `data\` with the existing RAG-005 eligibility rule; no caching, no layout analysis, no text repair, no source file touched.

```
total PDFs inspected ................ 478
PDFs containing OCR-eligible pages .. 19   (18 unique documents: two copies are byte-identical duplicates)
total pages inspected ............... 11,490
OCR-eligible pages .................. 166   (1.4% of pages)
OCR pages successfully recovered .... 162
OCR failures ........................ 0
pages still without usable text ..... 24
total OCR characters ................ 271,137
total processing time ............... 933.03 s  (15 min 33 s; native extraction of all 478 PDFs + OCR)
total OCR time ...................... 149.30 s
average OCR time per page ........... 0.899 s   (min 0.23 s, max 2.55 s, 166 calls)
```

The eligible-page count (**19 PDFs / 166 pages**) reproduces the RAG-004 prediction exactly, and **no OCR call raised an exception** (the few `MuPDF error: could not parse color space` messages on stderr came from benign structure-element objects and did not affect parsing or OCR).

#### Eligible pages per PDF

| File (relative to `data\`) | Pages | Native text pages | Eligible | Recovered | Failures | No usable text | OCR chars |
|----------------------------|------:|------------------:|---------:|----------:|---------:|---------------:|----------:|
| `literature\AI\2021_UNESCO_AI-and-education-Guidande-for-policy-makers_EN.pdf` | 50 | 49 | 1 | 1 | 0 | 0 | 97 |
| `literature\AI\8-AI_and_VR_Enabled_Modern_LMS_for_Student.pdf` | 14 | 3 | 11 | 11 | 0 | 0 | 32,291 |
| `literature\AI\DEMOCRATISING_AI_final (2).pdf` | 28 | 26 | 2 | 1 | 0 | 1 | 61 |
| `literature\AI\Generative-ai-higher-education-book.pdf` | 287 | 282 | 5 | 5 | 0 | 0 | 369 |
| `literature\AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 362 | 342 | 13 | 12 | 0 | 8 | 9,340 |
| `literature\AI\View of Challenges with Gamification in Higher Education.pdf` | 6 | 0 | 6 | 6 | 0 | 0 | 39,941 |
| `literature\Algorithms\catmull_thesis.pdf` | 84 | 0 | 84 | 83 | 0 | 1 | 78,227 |
| `literature\Algorithms\Lin-View of Design and Application of Intelligent Assisted Learning System for Higher Education Students.pdf` | 10 | 0 | 10 | 10 | 0 | 0 | 40,128 |
| `literature\Algorithms\perlin.pdf` | 10 | 8 | 2 | 2 | 0 | 0 | 67 |
| `literature\Algorithms\sellers.pdf` | 86 | 85 | 1 | 1 | 0 | 0 | 126 |
| `literature\Teaching\-John_Biggs_and_Catherine_Tang-_Teaching_for_Quali-BookFiorg-.pdf` | 418 | 410 | 2 | 2 | 0 | 6 | 3,831 |
| `literature\Teaching\Experiential_Learning_Experience_As_The_Source_Of_.pdf` | 23 | 3 | 20 | 20 | 0 | 0 | 56,278 |
| `literature\Teaching\Formative_Assessment_and_Self-Regulated_Learning_A.pdf` | 22 | 21 | 1 | 1 | 0 | 0 | 290 |
| `literature\Teaching\LS_OverView.pdf` | 28 | 27 | 1 | 1 | 0 | 0 | 301 |
| `literature\VR\DEMOCRATISING_AI_final (2).pdf` | 28 | 26 | 2 | 1 | 0 | 1 | 61 |
| `literature\VR\FundamentalVR_ValidationFeb2019_BOTA.pdf` | 2 | 0 | 2 | 2 | 0 | 0 | 8,334 |
| `literature\VR\Maister_et_al_TICS_authors_version_Figures.pdf` | 27 | 26 | 1 | 1 | 0 | 0 | 42 |
| `rules\plagiarism.pdf` | 3 | 0 | 1 | 1 | 0 | 2 | 1,285 |
| `rules\Writing-a-Dissertation.pdf` | 67 | 66 | 1 | 1 | 0 | 0 | 68 |
| **Total (19 PDFs)** | **1,555** | **1,374** | **166** | **162** | **0** | **19** | **271,137** |

Mean recovery: 271,137 OCR characters over 162 recovered pages ≈ **1,673 chars/recovered page**. Several mostly-native PDFs contribute only a handful of small recovered pages (e.g. `perlin.pdf` 67 chars, `sellers.pdf` 126 chars) — i.e. isolated figure/scan pages inside otherwise native documents, exactly the intended selective behaviour.

#### Verification that no ordinary native-text page was OCR-ed

- The rule is structural: a page is eligible only if it is `POSSIBLY_SCANNED` (zero native text with images, or < 50 native chars with images) **or** has images with < 50 native chars. Ordinary `TEXT` pages can never qualify.
- Independent audit during the run: for every one of the 11,490 pages, any page where OCR ran despite having ≥ 50 native characters was recorded. **Result: 0 pages.**
- Consequently 11,324 pages (98.6%) were never rendered and never sent to Tesseract, and the 1,374 native-text pages inside the 19 eligible PDFs kept `text_source = NATIVE_PDF` untouched.

#### Every page with no usable text, and every OCR failure (24 pages, 0 failures)

`dark` = fraction of pixels below 200 in a 36 DPI grayscale render. No OCR failures occurred, so this table lists every page that ended without usable text.

| # | File | Page | Status | Native chars | OCR attempted | OCR chars | Images | dark | Verdict |
|---|------|-----:|--------|-------------:|--------------|----------:|-------:|-----:|---------|
| 1 | `AI\DEMOCRATISING_AI_final (2).pdf` | 4 | possibly_scanned | 0 | yes | 0 | 2 | 1.000 | **Genuinely problematic** — dark full-bleed image page |
| 2 | `VR\DEMOCRATISING_AI_final (2).pdf` | 4 | possibly_scanned | 0 | yes | 0 | 2 | 1.000 | **Genuinely problematic** — same document, byte-identical duplicate |
| 3 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 120 | possibly_scanned | 0 | yes | 0 | 9 | 0.000 | Effectively blank (only tiny decorative fragments) |
| 4 | `Algorithms\catmull_thesis.pdf` | 84 | possibly_scanned | 0 | yes | 0 | 1 | 0.000 | Blank (verified: pixel min 237 / max 255, mean 255) |
| 5 | `AI\eu-ethical guidelines … NC0722649ENN.pdf` | 40 | empty | 0 | no | 0 | 0 | 0.128 | **Genuinely problematic** — image nested inside Form XObjects ⇒ not OCR-eligible |
| 6 | `rules\plagiarism.pdf` | 2 | empty | 0 | no | 0 | 0 | 0.075 | **Genuinely problematic** — vector-only page (1,295 filled paths, no raster image) |
| 7 | `rules\plagiarism.pdf` | 3 | empty | 0 | no | 0 | 0 | 0.019 | **Genuinely problematic** — vector-only page (324 filled paths, no raster image) |
| 8 | `AI\EPRS_STU(2020)641530_EN.pdf` | 2 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 9 | `AI\EPRS_STU(2020)641530_EN.pdf` | 14 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 10 | `AI\EPRS_STU(2020)641530_EN.pdf` | 99 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 11 | `AI\eu-ethical guidelines … NC0722649ENN.pdf` | 39 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 12 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 4 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 13 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 8 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 14 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 9 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 15 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 10 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 16 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 11 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 17 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 304 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 18 | `AI\Turnitin_Plagiarism_AI_Similarity_Index.pdf` | 332 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 19 | `Teaching\-John_Biggs_and_Catherine_Tang-…BookFiorg-.pdf` | 17 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 20 | `Teaching\-John_Biggs_and_Catherine_Tang-…BookFiorg-.pdf` | 27 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 21 | `Teaching\-John_Biggs_and_Catherine_Tang-…BookFiorg-.pdf` | 137 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 22 | `Teaching\-John_Biggs_and_Catherine_Tang-…BookFiorg-.pdf` | 303 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 23 | `Teaching\-John_Biggs_and_Catherine_Tang-…BookFiorg-.pdf` | 305 | empty | 0 | no | 0 | 0 | 0.000 | Blank |
| 24 | `Teaching\-John_Biggs_and_Catherine_Tang-…BookFiorg-.pdf` | 417 | empty | 0 | no | 0 | 0 | 0.000 | Blank |

Summary: **19 blank** pages (including `catmull_thesis.pdf` p.84, verified directly) and **5 genuinely problematic** pages. 19 of the 24 lie inside OCR-eligible PDFs; the other 5 (EPRS ×3, eu-ethical ×2) belong to documents with no eligible pages at all. The 20 `EMPTY` pages match RAG-004's corpus baseline (**20 empty pages**) exactly, and the 4 remaining no-text pages are OCR-eligible image-only pages where OCR legitimately recovered nothing.

**OCR failures: none.** Every one of the 166 eligible pages was rendered and OCR-ed successfully; `ocr_error` is `None` for all 11,490 pages.

##### The 5 genuinely problematic pages in detail

1. **`DEMOCRATISING_AI_final (2).pdf` p.4 (both copies).** OCR *was* attempted and returned zero characters. The page is not blank — it carries two large images (3186×2255 DeviceCMYK and 2490×3517 `Indexed(36,DeviceCMYK)`) inside a Form XObject — but it renders as an essentially black page: at 72/150/300 DPI the mean luminance is ~20.8/255 and **every pixel is ≤ 31** (histogram: 7.02 M pixels in 0–15, 1.68 M in 16–31, nothing above 31). There is no light-on-dark text in the rendering, and RAG-005 applies no thresholding/inversion/pre-processing, so Tesseract correctly finds nothing. This is a dark design/graphic page, not recoverable text; recovering it would require image pre-processing, which is out of scope.
2. **`eu-ethical guidelines … p.40`.** Classified `EMPTY` and therefore *not* OCR-eligible, because the picture is not exposed as a page image XObject: `page.get_images()` returns 0, yet `page.get_image_info()` reports one 455×736 image and the page carries 60 filled vector paths inside four Form XObjects (`Fm0`–`Fm3`), with 12.8% ink coverage. This is the known conservative-rule gap (architectural concern 3): content exists and would likely be OCR-recoverable if the page were rendered on the strength of `get_image_info()`/ink coverage rather than `get_images()`.
3. **`rules\plagiarism.pdf` p.2 and p.3.** Also classified `EMPTY` and skipped: no raster image at all, but heavy vector artwork (1,295 and 324 filled paths; 7.5% and 1.9% ink). If that artwork is text converted to outlines, it is unrecoverable under the current rule; if it is decorative, skipping is correct. p.1 of the same document (1 image + 953 paths) *was* eligible and recovered 1,285 characters, which shows the rule works when an image XObject exists.

None of these five pages is silently wrong: each is reported as having no usable text, and nothing was invented, concatenated, or heuristically repaired.

## 13. Full-corpus OCR workload / runtime

Eligibility is computed purely from **native** extraction, so it can be enumerated without OCR:

- **Eligible PDFs: 19** (18 unique documents)
- **Eligible pages: 166** (of 11,490 = ≈1.4%)

**Estimate (made before the run):** ≈150 s of OCR (≈0.9 s/page), i.e. ~2.5 minutes, allowing 2–6 minutes.

**Actual (full-corpus run, approved 2026-10-10):**

| Metric | Value |
|--------|-------|
| Pages OCR-ed | 166 (exactly the predicted count) |
| OCR time (render + Tesseract only) | **149.30 s** — the estimate was accurate to within 3 s |
| Average OCR time per page | **0.899 s** (min 0.23 s, max 2.55 s) |
| Whole-run wall clock (native extraction of all 478 PDFs + eligibility + OCR) | **933.03 s (15 min 33 s)** |
| OCR share of the run | ≈16% |

Full-corpus OCR is **complete** for this task; the results are in §12. Re-running it would repeat the same work — the OCR call remains a pure function of `file_hash + page_number + ocr_language + ocr_dpi`, so a cache can be added later without restructuring (not implemented here).

## 14. OCR quality issues observed

1. **Systematic character substitutions** on the 1974 typewritten thesis: `c`↔`e` (Cvery/Every, crea/area, cass/case), `t`↔`c` (lese/less, subdivicsd/subdivided), `r`↔`i` (oniy/only, Ssierca/Science), `m`↔`un`/`rn` (fscuily/faculty, Chaiunan/Chairman, Commitice/Committee), `!`↔`l` (Genera!, trivial!), `y`↔`i` (nearky/nearly, Desrses/Degree), and `y`↔`e` (Universily/University).
2. **Equations and matrix expressions are destroyed** (`f(t) = Shoat!`, `bla MIPal`, `rin` brackets). Affected pages are not usable as-is for formula-bearing retrieval.
3. **Figure labels and diagram annotations are poor** (p.12, p.80) and occasionally contain stray quote marks (`“KET,`).
4. **Rubber stamps / decorative headers garble** (`+e eee scent sme ame`, `ry`, `37,`).
5. **Layout is flattened**: OCR emits a single page-level paragraph block; multi-column or figure-heavy pages interleave text and labels.
6. **Blank pages produce no text** (catmull p.84) — expected and correctly reported, not a failure.
7. **Greek is unavailable** (`ell` not installed), so any Greek spans in the corpus would be transcribed by `eng` (Latin) and come out wrong. No Greek was observed in the 166 OCR-ed pages.
8. **Modern-born-digital scans OCR far better.** The Experiential Learning (23 pp.), Lin (10 pp.), and FundamentalVR (2 pp.) documents produced clean, high-yield text (56,278 / 40,128 / 8,334 chars). Spot checks show only light noise — Experiential p.23: `characterist:cally` (characteristically), `af inquiry` (of), `cottespond ta` (correspond to); Lin p.3 near-perfect, but Lin p.10 reference numbering is reordered by the flattened two-column layout (`1. N 10. ll. 13.`); FundamentalVR prose is clean while its branding/logo garbles (`FUNDAMENTAL gig a / SV2GERY`) and narrow side-column text interleaves out of order.
9. **24 pages still have no usable text, 0 OCR failures** (full list in §12). Nineteen are genuinely blank pages (17 `EMPTY` with no image at all, plus `catmull_thesis.pdf` p.84 and `Turnitin…` p.120 which render fully white). Five are genuinely problematic rather than blank:
   - `DEMOCRATISING_AI_final (2).pdf` p.4 in **both** the `AI\` and `VR\` copies (byte-identical duplicates, SHA-256 `5C7694…577F`): a dark full-bleed image page (mean luminance 20.8/255, **every** pixel ≤ 31) — OCR ran and correctly returned nothing; recovering it would need image pre-processing, which is out of scope.
   - `eu-ethical guidelines … p.40`: real content (a 455×736 image plus 60 vector paths inside Form XObjects) but **not** OCR-eligible because the picture is not exposed as a page image XObject.
   - `rules\plagiarism.pdf` p.2 and p.3: vector-only pages (1,295 and 324 filled paths, no raster image), also not OCR-eligible.
   The last three are the known conservative-rule gap (concern 3 below) and are the only places where the corpus may still hide recoverable text.
10. **No heuristic repair was applied** — garbled equations, figure text, stamps, and dark pages are reported as-is with OCR provenance; nothing was inverted, thresholded, de-noised, or LLM-corrected.

## 15. Architectural concerns

1. **`content_items` retains both native and OCR blocks** for a page where OCR was applied (provenance-first). Consumers building retrieval content must filter on `text_source` (use the OCR block where `ocr_applied`, else native) or read `ParsedPage.raw_text`.
2. **Two OCR dependencies**: `pytesseract` (thin wrapper) and `Pillow` (image decode). Both are lightweight; no heavyweight OCR framework was added.
3. **Eligibility requires a page image XObject — a confirmed, quantified gap.** The full-corpus run found **3 pages** whose content is present but not OCR-eligible because `page.get_images()` returns no image: `eu-ethical guidelines … p.40` (picture nested in Form XObjects) and `rules\plagiarism.pdf` p.2/p.3 (vector-only pages, 1,295/324 filled paths). Extending eligibility to `page.get_image_info()` and/or an ink-coverage threshold would cover these; it was **not** changed here because the brief fixed the existing selective rule. This is the only place where recoverable text may still be missing.
4. **Caching is designed but not implemented**: the OCR call is a pure function of `file_hash + page_number + ocr_language + ocr_dpi`. The full corpus costs only **149 s** of OCR, so caching is an optimisation, not a necessity; no Qdrant/DB was added.
5. **Privacy**: rendering and OCR are fully local; nothing leaves the machine.
6. **OCR language is now English-only by default.** `ell` is not installed in the local Tesseract, so `eng+ell` would fail at OCR time; the default is `eng`. If Greek material is ever added, install the `ell` traineddata and set `PHD_RAG_OCR_LANGUAGE=eng+ell`.
7. **Single page-level OCR block.** Quality inspection confirms the flattening cost: multi-column pages and figure-heavy pages interleave text, labels, and equation fragments. A layout-aware OCR pass was explicitly out of scope, so consumers should treat multi-column scanned pages as noisy.
8. **OCR preserves noise by design.** No inversion, thresholding, de-noising, or LLM correction is applied — garbled equations, figure labels, stamps, and even a fully dark page are passed through unchanged with OCR provenance, so the noise is visible and auditable rather than hidden.

---

## 16. Rule carried forward into chunking (RAG-006 input)

Provenance must keep its meaning downstream — do not flatten it away when chunking:

```
OCR prose                        → usable evidence, tagged with OCR provenance
OCR equation / figure text       → retrieval aid only, NOT verbatim-authoritative evidence
```

Why this rule, from the evidence in §12/§14:

- On `catmull_thesis.pdf`, OCR **prose** is readable at roughly 95–97% word accuracy (`usable with noise`) — good enough to be cited as evidence, provided it carries `TextSource.OCR`.
- OCR output for **equations, diagrams, figure labels, and flattened multi-column pages** is degraded (`f(t) = Shoat!`, `bla MIPal`, `Raster-elament`, `1. N 10. ll. 13.`) and must not be quoted as the author's verbatim words.
- RAG-005 does not attempt to separate the two: one OCR page becomes one `ParsedBlock` (`text_source = OCR`, locator `page:{n}:ocr:1`) that may contain both. Doing better needs layout analysis, which is out of scope.

Consequences to carry into chunking (to be designed in RAG-006 — **nothing implemented here**):

1. Chunks built from OCR text must retain `TextSource.OCR`; OCR text must never be silently upgraded to authoritative native text.
2. Because a single OCR block can mix prose with equation/figure text, OCR-derived chunks should be treated as **lower-confidence retrieval aid** by default — ranked, filtered, or flagged accordingly — rather than quoted as verbatim source, unless a later task adds layout-aware classification.
3. No heuristic repair of OCR equations/figures (RAG-005 rule). Noise is preserved and reported, not "fixed".
4. The 24 no-text pages (19 blank + 5 problematic) and the 3 known missed-eligibility pages should be treated as **no evidence**, not empty evidence.

---

## Scope confirmation

- The full-corpus OCR run executed **only** the existing selective rule: 166 of 11,490 pages, 0 pages with ≥ 50 native characters touched. Ordinary native-text pages were never rendered or OCR-ed.
- No cloud OCR, multimodal/vision LLM, layout-aware OCR, handwriting, table-OCR, chunking, embeddings, Qdrant, Ollama, caching, or metadata enrichment implemented; no heuristic OCR repair.
- No corpus source file was modified; the run and all inspections were read-only.
- Provenance preserved end-to-end: `page_number`, `native_text`, `ocr_text`, `text_source`, `ocr_applied`, `ocr_language`, `ocr_error`, `source_locator`; native and OCR text are never concatenated.
- Native PDF extraction behaves exactly as before on normal PDFs; `GET /health`, RAG-002 scanning, DOCX/Markdown parsing, and RAG-004 PDF native parsing are unchanged.
- Final `pytest`: **163 passed, 1 warning** (§5).
- RAG-006 has **not** been started. §16 records the prose-vs-equation/figure rule as input for chunking only.
