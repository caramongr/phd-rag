# RAG-014 — Local web interface

A local interface over the RAG-013 mode service: choose a mode, type a question (or paste a
paragraph of your thesis), run it, read the answer, and inspect everything it rested on. It
starts with one command:

```powershell
.\.venv\Scripts\python.exe -m app.web        # http://127.0.0.1:8000/
```

No retrieval, generation, citation, discovery or mode logic was added, moved or duplicated.
The backend owns one RAG-013 `ModeService`, exposes it as three JSON endpoints, and serves a
static page; the frontend renders the `ModeResult` it receives and decides nothing.

## 1. Files created

| File | Lines | Role |
|---|---|---|
| `app/api/modes.py` | 220 | `GET /api/modes` (the mode list and its UI copy) and `POST /api/mode` (run one mode); the advanced options, as a pydantic model, with the ones that do not apply to a mode left out |
| `app/api/status.py` | 220 | `GET /api/status`: Ollama availability/version/model, index availability/collection/points, configured models, locality — all without loading a model |
| `app/api/errors.py` | 291 | Friendly notices (`ModeNotice`) derived from a result, and the JSON error handlers so no failure ever leaves as HTML |
| `app/web/service.py` | 129 | `ModeRunner`: one lazily built `ModeService` per process, one run at a time, closed on shutdown, with a probe lock that keeps a status check from colliding with a run |
| `app/web/__init__.py` | 37 | `STATIC_DIR`, the localhost defaults and `create_web_app()` |
| `app/web/__main__.py` | 64 | `python -m app.web`: argument parsing, the banner, uvicorn |
| `app/web/static/index.html` | 134 | The page: mode selector, input, Run/Clear, loading, answer panel, copy row, collapsed Sources / External / Validation / Audit panels, collapsed advanced options |
| `app/web/static/app.js` | 826 | The renderers: modes, status, notices, source cards, findings, labelled excerpts, external cards, references, warnings, validation, audit, copy, options |
| `app/web/static/style.css` | 440 | Readable academic styling, light and dark via `prefers-color-scheme` |
| `tests/web_support.py` | 103 | The web test doubles: a real `ModeService` over the RAG-013 fakes, a recording factory, fake Ollama/Qdrant clients, the static-file reader |
| `tests/test_web_api.py` | 397 | 27 tests — the page, the mode list, running each mode, the options, the notice mapping, the error shapes, escaping |
| `tests/test_web_status.py` | 282 | 16 tests — Ollama and index probing, reuse of the pipeline's client, the probe never building the pipeline, the timeout |
| `tests/test_web_server.py` | 273 | 20 tests — the localhost default, the runner's lifetime and serialisation, the served controls, the escaping contract |
| `docs/rag014-ui-validation.txt` | 231 | Real validation: the five modes in a real browser against the real pipeline (§20) |
| `docs/manual.md` | 397 | The user manual: what each mode is for, how to read a result, the options, recipes, troubleshooting, limits, privacy |
| `docs/manual-images/*.png` | 13 files | The manual's screenshots, captured from real runs |
| `docs/rag014-screenshots/*.png` | 5 files | One full-page screenshot per mode (optional per §23, included because they cost nothing) |
| `RAG-014-report.md` | this file | |

## 2. Files modified

| File | Change |
|---|---|
| `app/main.py` | Wires the two new routers, installs the JSON error handlers, sets `app.state.mode_runner`, and mounts the static UI at `/` after the routers so `/api/...` and `/health` win |
| `app/modes/factory.py` | **New** (§1): one `build_mode_service()` used by the CLI *and* the web app, so the pipeline is assembled in exactly one place |
| `app/modes/inspect.py` | `build_service()` now delegates to that factory (behaviour unchanged) |
| `app/modes/service.py` | Two additions the UI needs and nothing else changed: a `retrieval_service` property that reports the RAG-009 service without building it, and RESEARCH now reports `CONFLICTS` as its own section instead of folding it into `POSSIBLE MATCHES` (§9) |
| `app/modes/models.py` | `ExternalSourceView` gained the fields the external card needs (`abstract`, `open_access`, `fulltext_status`, `identity_confidence`, `local_evidence_eligible`, `publisher`) — all carried through from RAG-010's record, none invented |
| `app/modes/__init__.py` | Exports `build_mode_service` |
| `app/generation/ollama.py` | `available_models()` takes an optional timeout so the status probe can be short (default unchanged: 30 s) |
| `pyproject.toml` | Declares `uvicorn`, which the "one command" requirement depends on (it was already installed) |
| `README.md` | A "Web interface" section with the exact command, and the RAG-014 prose paragraph |
| `tests/test_modes_service.py`, `tests/test_modes_cli.py` | Updated for the five research sections, plus a new test that a provider conflict lands in `CONFLICTS` and not in `POSSIBLE MATCHES` |

## 3. Architecture

```
python -m app.web
      |
      v
app/web/__main__.py  ──> app/web/__init__.create_web_app()
                              |
                              v
                        app/main.create_app(mode_runner=ModeRunner(...))
                              |                       |
                              |                       └─ one ModeService, lazily built,
                              |                          one run at a time, closed on shutdown
                              |
        ┌─────────────────────┼─────────────────────────────┐
        |                     |                             |
   /health              /api/modes, /api/mode          /api/status
   (RAG-001)            app/api/modes.py               app/api/status.py
                              |                             |
                              v                             v
                        ModeRunner.run(...)           HTTP probe + point count
                              |                       (never builds the pipeline)
                              v
                    RAG-013 ModeService.run_mode(...)
                              |
                              v
                        ModeResult  ──> result.to_json()  (+ transport-only "notice")
                              |
                              v
        static index.html + app.js (renderers, textContent only) + style.css
```

Three properties fall out of this shape:

* **The frontend cannot invent a mode.** `GET /api/modes` is built from `MODES` — the RAG-013
  presets — so the selector, the descriptions and the applicable advanced options all come
  from the same objects the CLI uses. A test asserts every mode in `MODES` has UI copy.
* **The response is the ModeResult.** `POST /api/mode` returns `result.to_json()` unchanged;
  the only extra key is `notice`, a transport-only sentence the UI shows above the result, and
  it is absent when there is nothing to say. A test asserts the payload's keys are exactly the
  `ModeResult` fields plus that one key.
* **One pipeline per process.** The models and the Qdrant client are built on the first
  request and reused for the life of the server (a CLI process pays that cost per invocation).
  Runs are serialised with a lock because the service holds two local models and one local
  Qdrant store.

## 4. The API (§14)

`POST /api/mode`

```json
{ "mode": "evidence", "query": "What is AI governance in higher education?", "options": {} }
```

| Option | Type | Applies to |
|---|---|---|
| `length` | `concise` \| `normal` \| `detailed` | evidence, writing |
| `candidate_k` | 1–200 | all modes |
| `final_k` | 1–50 | all modes |
| `rerank` | boolean | all modes |
| `year_min`, `year_max` | 1500–3000 | research |

`GET /api/modes` returns each mode's `name`, `label`, `description` (the copy from §5),
`purpose`, `retrieval_mode`, `output_style`, `generates`, `validates_citations`,
`uses_external_discovery` and the option names that apply to it, plus the option metadata the
form is built from (`advanced_options`) and `max_query_chars`.

Unknown fields are rejected rather than ignored: `{"options": {"temperature": 0.9}}` is a 422
with `options.temperature: Extra inputs are not permitted`, because silently ignoring a
setting the user believes they changed is worse than refusing it.

## 5. Status (§15)

`GET /api/status` answers two questions cheaply and never loads a model:

```json
{
  "ollama": {"available": true, "version": "0.35.1", "model": "qwen3.5:35b",
             "model_present": true, "base_url": "http://127.0.0.1:11434"},
  "index":  {"available": true, "collection": "phd_rag_chunks_v1", "points": 11469,
             "target": ".qdrant_data", "collection_present": true,
             "reused_client": true, "busy": false, "error": null},
  "generation_model": "qwen3.5:35b",
  "embedding_model": "BAAI/bge-m3",
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "locality": "localhost only (127.0.0.1) unless explicitly configured otherwise"
}
```

* Ollama is probed over HTTP with a 2-second timeout (`/api/version`, `/api/tags`); the model
  is never loaded, and a model that is not installed is reported rather than discovered later
  as a failed run.
* The index is counted with the client the pipeline already owns when a run has happened, and
  with a temporary client (opened, counted, closed) when it has not — because a local Qdrant
  store may be opened once per process. If a run is in flight, the probe takes the runner's
  lock; if it cannot, it reports `busy` instead of opening a second client and breaking the
  run that is already using it.
* `points` is `null` (with an explanation in `error`) when the collection is missing; a locked
  store is reported as locked. Status never raises.

## 6. Failure handling (§16)

Failures are turned into sentences on the server, so they are testable without a browser and
cannot drift between the CLI and the UI. `describe_result(result)` reads the result's status,
warnings and audit and returns at most one notice:

| Situation | Notice |
|---|---|
| model missing / Ollama unreachable / timeout (from the failure detail) | error — "The configured model is not installed", "Ollama is not reachable", "The model took too long" |
| `model_error` with no recognisable cause | error — "The model could not answer" |
| `no_citable_context` | warning — "Nothing citable was retrieved" |
| `insufficient_evidence` | warning — "The retrieved evidence was not enough to answer" |
| `no_results` | warning — "Nothing matched this query" |
| claims withheld as unsupported | warning — "N claim(s) were withheld as unsupported" |
| provider outage, retrieval failure, context budget, non-citable context | warning — one sentence each |

Invalid requests never reach the handler: FastAPI's validation errors are converted to
`{"error": {"kind": "invalid_request", "title", "detail", "fields"}}` with a readable detail
line, and anything unexpected becomes `{"error": {"kind": "internal_error", ...}}` — JSON, so
the UI can show it, with the traceback only when `debug` is on.

## 7. Security and local-only operation (§17)

| Requirement | How |
|---|---|
| local UI only by default | `DEFAULT_HOST = "127.0.0.1"`; `--host 0.0.0.0` is the only way off the machine and the banner prints a warning when it is used |
| no authentication needed | nothing but localhost can reach the default binding; there is no user model and no session state |
| escape rendered source/user text | every value is written with `textContent`; `app.js` contains no `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write` or `eval` — asserted by a test that reads the file |
| never render document HTML as trusted markup | there is no server-side templating of document content at all: the page is a static file, and document text arrives as JSON |
| no built-in page is trusted either | `index.html` contains no inline handlers (`onclick=`, `onerror=`), so a Content-Security-Policy can be added later without rewriting the page |

Verified in the browser, not only in tests: a query containing
`<img src=x onerror="window.__xss=1"><script>window.__xss=2</script>` rendered as visible
text, `window.__xss` stayed `undefined`, and the page contained no injected `img` or `script`
element.

## 8. The interface (§4–§12)

* **Main screen** — mode selector with the §5 description under it, large input box that
  accepts a question, a paragraph or a thesis section, Run, Clear, and a loading indicator
  that names the mode and what it is doing ("evidence mode is working — retrieval, then the
  model, then validation…"). The Run button is disabled while a run is in flight.
* **Evidence / Writing** — the validated answer with Harvard citations, the references, the
  warnings, and the original RAG-011 answer in a collapsed panel. Sources and validation
  details are collapsed by default and hold the full claim-by-claim audit.
* **Source cards** — title, author, year, pages, section, DOI, corpus role, the role it was
  allowed to play, evidence yes/no, OCR flag, content type and a text preview; expanding a
  card shows the retrieved text, dense and rerank score, chunk id, document id, file, corpus
  role, evidence eligibility, retrieval mode and page/section provenance (§11).
* **Consistency** — findings grouped into terminology, duplication, unsupported claims,
  numeric discrepancies and style, each with its description, the draft excerpt, the
  literature excerpt, the method and the chunks it rests on. The unsupported-claims group
  carries the caveat that "no match in the retrieved literature" is not "false", and nothing
  is rewritten.
* **Supervisor** — a banner saying the material is not scholarly evidence, then excerpts
  labelled `[supervision]`, `[future_chapters]`, `[guidance]`, `[rules]` with file, section,
  pages and text; no citations and no evidence badges.
* **Research** — all five sections in the mode's order (LOCAL EVIDENCE, NEW EXTERNAL PAPERS,
  ALREADY IN CORPUS, POSSIBLE MATCHES, CONFLICTS) with empty sections shown as `(none)`;
  every external card is badged `NOT IN LOCAL CORPUS` (or `ALREADY IN CORPUS`) and `metadata
  only — never evidence`, with authors, year, venue, publisher, DOI, providers, verification
  status, open-access and full-text status, an abstract preview and a provenance panel.
* **Copy** — "Copy answer", "Copy answer + references", "Copy references"; formatting is
  preserved because the text is copied verbatim, and reference-only buttons are disabled when
  there are no references.

## 9. Real validation (§20)

`docs/rag014-ui-validation.txt` records the five required queries run through the real
interface in a real browser, against the real index (11,469 points), the real model
(`qwen3.5:35b`) and the real providers:

| Mode | Status | Seconds | Local | External | References | Verdict |
|---|---|---|---|---|---|---|
| EVIDENCE | ok | 12.11 | 8 | 0 | 3 | §2/§6 met — validated Harvard-cited answer, 12/12 claims supported, all 8 sources evidence |
| WRITING | ok | 2.52 | 8 | 0 | 2 | §3/§6 met — 6 evidence + 2 non-citable sources, neither cited |
| CONSISTENCY | ok | 1.38 | 15 | 0 | 0 | §4/§7 met — 19 findings in three groups, caveat shown, draft never evidence |
| SUPERVISOR | ok | 0.63 | 12 | 0 | 0 | §5/§8 met — 12 excerpts labelled `[supervision]` etc., 0 evidence badges, 0 references |
| RESEARCH | ok | 1.08 | 6 | 20 | 0 | §6/§9 met — five sections, 20 records badged "NOT IN LOCAL CORPUS" and "metadata only" |

Plus the cross-cutting checks: the status line before and after a run, a status probe that
reuses the pipeline's client (`reused_client: true`) in 0.02 s, advanced options reaching the
pipeline (`candidate_k: 8` visible in the audit), "copied" from the copy button, an empty
submission handled without a request, hostile input rendered as text, and the four live 422
shapes. Screenshots: [docs/rag014-screenshots/](docs/rag014-screenshots) (one per mode).

## 10. Test result (§19)

```
1054 passed, 2 warnings in 9.89s
```

990 pre-existing tests plus **64 new** ones (27 API, 16 status, 20 server, +1 RAG-013 test for
the new CONFLICTS section), all passing. The two warnings are pre-existing (a Starlette/httpx
deprecation and Qdrant's local payload-index notice).

| §19 required case | Test |
|---|---|
| home page loads | `test_web_api.py::test_home_page_loads`, `::test_home_page_loads_its_frontend_files` |
| all five modes listed | `::test_all_five_modes_are_listed`, `::test_every_mode_carries_the_required_description`, `::test_every_mode_has_ui_copy_in_the_backend` |
| `POST /api/mode` calls ModeService | `::test_post_mode_calls_the_mode_service`, `::test_post_mode_runs_the_second_mode_too` |
| evidence response renders references | `::test_evidence_response_renders_references` (+ the browser run) |
| writing response preserves citations | `::test_writing_response_preserves_citations_and_non_citable_context` |
| research keeps local/external separate | `::test_research_response_keeps_local_and_external_separate`, `::test_research_external_records_carry_the_card_fields` |
| consistency findings render | `::test_consistency_response_carries_findings_by_kind` + `test_web_server.py::test_the_frontend_renders_the_sections_the_modes_produce` |
| supervisor labels render | `::test_supervisor_response_labels_the_material` |
| errors return safe JSON/messages | `::test_unknown_mode_is_rejected_with_a_readable_error`, `::test_empty_query_is_rejected_with_a_readable_error`, `::test_a_backwards_year_range_is_rejected`, `::test_a_model_failure_becomes_a_friendly_notice`, `::test_no_results_is_reported_as_a_warning_not_an_error`, `test_web_server.py::test_an_unexpected_failure_stays_json` |
| user/source HTML is escaped | `::test_user_html_arrives_as_data_and_is_never_markup`, `test_web_server.py::test_the_frontend_never_interprets_data_as_markup` |
| localhost is the documented default | `test_web_server.py::test_the_default_binding_is_localhost`, `::test_the_command_line_defaults_to_localhost`, `::test_exposing_the_interface_requires_the_explicit_host_flag`, `::test_the_banner_says_so_when_the_interface_is_exposed` |

## 11. Acceptance criteria (§21)

| Criterion | Status |
|---|---|
| all previous 990 tests pass | **met** — 990 + 64 = 1054 passed |
| all new UI/API tests pass | **met** — 64/64 |
| UI starts with one command | **met** — `python -m app.web`, documented in the README |
| all five modes work | **met** — five browser runs, 5/5 `ok` |
| validated answers and references render correctly | **met** — Harvard citations in the answer, 3 deduplicated references, claim audit expandable |
| sources are inspectable | **met** — role, evidence flag, scores, chunk id, document id, provenance, text preview |
| external/local separation is visually clear | **met** — five sections, per-card "NOT IN LOCAL CORPUS" and "metadata only — never evidence" badges |
| no evidence-policy leakage | **met** — 0 non-citable sources cited, 0 external records usable as evidence, 0 evidence badges in SUPERVISOR |
| localhost-only default | **met** — `127.0.0.1`, with an explicit warning if changed |
| no new architecture duplicated | **met** — one `ModeService`, one `build_mode_service`, no mode logic in the API or the frontend |

## 12. Known limitations

1. **One process holds the local index.** A local Qdrant store can be opened by one client
   per process, so the UI and a CLI run cannot both use `.qdrant_data` at the same time
   (observed during validation: a CLI test failed on the lock while the server was running,
   and passed as soon as it stopped). Status reports this as "in use" when a run is in flight;
   use a Qdrant server if you need both at once.
2. **No conversation state.** Each run is independent: there is no history, no follow-up, no
   "continue this paragraph". That is RAG-015 territory (sessions/workspaces) and was
   explicitly out of scope.
3. **No document upload** (§13 permits skipping it). The input box takes pasted text, which
   covers a paragraph or a section.
4. **The first request is slow.** The pipeline (BGE-M3, the cross-encoder, the Qdrant client)
   is built on the first run: the first EVIDENCE run took 12–17 s, later ones 2–3 s. The
   loading indicator names the mode while it happens, but there is no progress percentage.
5. **No streaming.** The answer appears when the whole run finishes, so a detailed answer
   keeps the spinner for longer.
6. **Escaping is a property of the renderer, not of the payload.** The API returns document
   text verbatim (which is right: it must not double-escape), so any *new* frontend must keep
   to the `textContent` rule. A test reads `app.js` and fails if that rule is broken.
7. **Screenshots are manual.** §23 made them optional; they were captured by hand through the
   browser integration, so a future UI change will not update them automatically.

## 13. Defects found and fixed during this stage

Browser validation and the live API checks found three real bugs that the unit tests had
missed — which is why the validation step exists:

| Defect | Symptom | Fix |
|---|---|---|
| `ModeRunner.retrieval_service()` called the attribute instead of reading it | The status probe never reused the pipeline's Qdrant client, so once a run had happened the probe tried to open the store a second time and reported "index unavailable — locked by another process" | Read the attribute as a value when it is not callable; a test now drives the *real* `ModeService` (whose attribute is a property) rather than a double that agreed with the bug |
| The status probe assumed `point_count` was an attribute | With the client correctly reused, the count raised `int() argument must be ... not 'method'` and status fell back to "unavailable" | Accept either a value or a callable, and a test asserts the shape of RAG-009's real `DenseRetriever` |
| A status probe and a run could race for the store | A run started while the probe held a temporary client would fail with "already accessed"; the user sees a broken run for no reason | The probe takes the runner's lock and reports `busy` instead of opening a second client |
| RESEARCH folded provider conflicts into POSSIBLE MATCHES | §9 asks for five sections; the page could not show a conflict separately | `CONFLICTS` is its own section (its note says the providers contradict each other and are never silently resolved), with a test |
| The `[supervision]`-style role label rendered as bare text | §8's labels were visible but unstyled, unlike every other badge | Rendered as badges with the non-evidence style |
| Empty research sections disappeared | A reader could not tell "no verified-new papers" from "the section was not rendered" | Empty sections render as `(none)`, matching the CLI |
| Stale content stayed in the DOM behind hidden panels | Metrics taken from the page mixed a previous mode's cards into the current one | Panels are cleared when hidden (no user-visible change; it keeps the DOM honest) |

## 14. Recommendation for RAG-015

1. **A session/workspace object** shared by the CLI and the UI: a named current chapter that
   CONSISTENCY compares against and WRITING continues, instead of pasting text each time.
2. **Streaming** generation through the same lock, so a long answer appears as it is written
   rather than after the run.
3. **A warm start**: build the pipeline in the background at server start (it is idle anyway)
   so the first real request is as fast as the rest, and expose its readiness in the status
   area.
4. **A UI-level integration test** that runs the five modes against the test doubles in a
   headless browser, so the renderers are covered by the suite and not only by manual runs.

## RAG-014 COMPLETE

One command, one pipeline, one result shape. The interface renders what the mode service
produces, keeps the corpus policy visible (evidence, non-citable context, metadata-only
records), never interprets a document as markup, and binds to localhost unless it is
explicitly told not to. 1,054 tests pass and the five modes were verified in a real browser
against the real corpus, model and providers.
