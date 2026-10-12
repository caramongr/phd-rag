# RAG-015-REPAIR — report

**Task:** RAG-015-REPAIR — Simple persistent chat with attachments and full-context Qwen
**Date:** 2026-10-12
**Status:** implemented, verified against the real local stack, awaiting architectural review.

---

## 1. Summary

The default interface is now a simple ChatGPT-style chat: **New chat**, a saved history in a
left sidebar, reopen / **Rename** / **Delete**, ordinary conversation, **Attach** for PDF, DOCX,
TXT and MD, and persistent storage in local SQLite. The conversation, the working documents and
the corpus are composed into one context that asks Ollama for the model's **full 262,144-token
window** instead of the 8,192-token evidence window.

Nothing was removed. The RAG-009 retrieval pipeline, Qdrant, BGE-M3, the reranker, the
bibliography guard, RAG-010 Internet discovery, RAG-011 generation, RAG-012 citation validation,
the five RAG-013 modes and the RAG-014 interface are all still there, in their files, reachable
from the **Advanced** button in the top bar; `/api/mode` is untouched and behaves exactly as
before.

The chat is *not* a sixth mode: it is a separate, small path that reuses the project's existing
pieces rather than adding a second stack —

| Reused | Where the chat uses it |
|---|---|
| `app.parsing.parser.parse_document` (RAG-003) | PDF, DOCX and Markdown attachment text |
| `app.generation.context.estimate_tokens` (RAG-011) | the one deterministic token estimator |
| `app.generation.models.SourceBlock`, `evidence_block` / `context_block` (RAG-011) | corpus excerpts keep the project's own role wording |
| `app.generation.ollama.OllamaClient`, `GenerationOptions`, `GenerationModel` (RAG-011) | the one Ollama HTTP implementation |
| `app.retrieval.service.RetrievalService` (RAG-009) | corpus retrieval, through the mode runner, so one Qdrant client per process |
| `app.web.service.ModeRunner` (RAG-014) | the shared pipeline and its lock |

## 2. Previous limitations confirmed

Read from the source as it was before this repair:

| Limitation | Actual value in the code |
|---|---|
| Previous generation `num_ctx` | `DEFAULT_GENERATION_NUM_CTX = 8192` — [app/config.py](app/config.py) (generation defaults), sent at [app/generation/service.py:370](app/generation/service.py#L370) and [app/generation/service.py:393](app/generation/service.py#L393), with `GenerationOptions.num_ctx = 8192` as the dataclass default in [app/generation/ollama.py:45](app/generation/ollama.py#L45) |
| Previous RAG context budget | `DEFAULT_GENERATION_CONTEXT_TOKENS = 6000` — [app/config.py](app/config.py), plus `DEFAULT_GENERATION_RESERVED_NON_EVIDENCE_TOKENS = 1500` held back for `my_work`/guidance |
| Previous web/query-size restriction | `MAX_QUERY_CHARS = 20_000` — [app/api/modes.py:102](app/api/modes.py#L102), enforced as `max_length` on the mode request |
| Mode-driven default UI | `python -m app.web` served one page, [`app/web/static/index.html`](app/web/static/index.html): a mode selector, a query box and **Run**; the RAG-013 mode service was the only thing behind it, and the only entry point was `/api/mode` |

The report of the brief matches the code; nothing had to be corrected.

## 3. Files created

| File | Purpose |
|---|---|
| [app/chat/__init__.py](app/chat/__init__.py) | the package's statement of purpose |
| [app/chat/models.py](app/chat/models.py) | `Chat`, `ChatMessage`, `ChatAttachment`, `ContextReport`, `ChatAnswer` and the typed errors the API turns into sentences |
| [app/chat/store.py](app/chat/store.py) | the SQLite store: schema, create / list / get / rename / delete, message ordering, attachment metadata |
| [app/chat/attachments.py](app/chat/attachments.py) | attachment validation, safe storage paths, text extraction, deletion |
| [app/chat/context.py](app/chat/context.py) | deterministic context budgeting for one turn, with recorded omissions |
| [app/chat/prompts.py](app/chat/prompts.py) + [app/chat/templates/system_v1.txt](app/chat/templates/system_v1.txt) (superseded by [system_v2.txt](app/chat/templates/system_v2.txt) in FIX2), [user_v1.txt](app/chat/templates/user_v1.txt) | the chat prompts, in files, with a content-addressed version |
| [app/chat/service.py](app/chat/service.py) | the chat service: orchestration, model call, optional corpus retrieval |
| [app/api/chats.py](app/api/chats.py) | the nine chat endpoints and their error envelope |
| [app/web/static/chat.js](app/web/static/chat.js) | the chat frontend (sidebar, conversation, composer, attachments, context panel), plus FIX1's **Retry this turn** action on a failed or unanswered turn |
| [tests/chat_support.py](tests/chat_support.py) | the chat test doubles and temporary-storage settings |
| [tests/test_chat_store.py](tests/test_chat_store.py), [tests/test_chat_attachments.py](tests/test_chat_attachments.py), [tests/test_chat_context.py](tests/test_chat_context.py), [tests/test_chat_service.py](tests/test_chat_service.py), [tests/test_chat_api.py](tests/test_chat_api.py), [tests/test_chat_retry.py](tests/test_chat_retry.py) | the focused RAG-015 suite (118 tests; `test_chat_retry.py` was added by FIX1) |
| [docs/rag015-validation.txt](docs/rag015-validation.txt) | the transcript of the real end-to-end validation |
| [docs/rag015-fix1-validation.txt](docs/rag015-fix1-validation.txt) | the FIX1 evidence: the captured failing request, the isolation matrix, the server log, and the three-turn validation after the fix |
| `RAG-015-REPAIR-report.md` | this report |

## 4. Files modified

| File | Why |
|---|---|
| [app/config.py](app/config.py) | the RAG-015 chat defaults (window, budgets, shares, limits, storage paths) and their `PHD_RAG_CHAT_*` environment variables. FIX1 lowers the chat answer cap to 4 096 and adds `chat_retry_on_repeat_abort`. The RAG-011 generation defaults are untouched |
| [app/main.py](app/main.py) | mounts the chat router, creates the `ChatRunner` next to the `ModeRunner`, closes both on shutdown |
| [app/web/service.py](app/web/service.py) | adds `ModeRunner.retrieve` (one RAG-009 retrieval through the shared pipeline, under the same lock), `ChatRunner` (one lazily built chat service) and `make_corpus_provider` |
| [app/generation/ollama.py](app/generation/ollama.py) | adds `OllamaClient.unload()`, the one way to ask Ollama for a fresh runner. FIX1's single retry uses it |
| [app/generation/prompts.py](app/generation/prompts.py) | `_load` became the public `load_template`, so the chat prompt loader reuses the same hashing/versioning instead of copying it. No behaviour change |
| [app/web/static/index.html](app/web/static/index.html) | the chat shell (sidebar, conversation, composer) plus the RAG-014 interface moved, unchanged, into a hidden `#advanced-view` |
| [app/web/static/style.css](app/web/static/style.css) | styles for the shell, chat list, messages, chips and composer, and one correctness rule — `[hidden] { display: none !important }` — so the attribute a component uses to hide itself is not overridden by that component's own `display`. Both screens are toggled with it |
| [docs/manual.md](docs/manual.md) | the manual now opens with the chat: a new §3 (the screen, chatting, the advanced screen), the Advanced route from §4 onwards, where conversations live in §11, and FIX1's troubleshooting rows for the repeat abort and the Retry action |
| [docs/manual-images/](docs/manual-images) | `15-chat-screen.png`, `16-advanced-screen.png`, `17-context-panel.png`, captured from the running interface |

`pyproject.toml` is unchanged: SQLite is in the standard library, and attachment uploads are
base64 in JSON, so no new dependency was needed (see §13).

## 5. Persistent chat

**Storage.** `data/chat/chat.sqlite3` (overridable with `PHD_RAG_CHAT_DB_PATH`). Three tables:

```
chats        (id, title, created_at, updated_at)
messages     (id, chat_id → chats, role, content, created_at, ordering, context_report)
attachments  (id, chat_id → chats, filename, stored_path, content_type, extension,
              size_bytes, text_status, parsed_text, char_count, token_estimate,
              parse_error, created_at, ordering)
```

`ordering` gives a stable order inside a chat; `context_report` keeps the JSON context summary
of the assistant turn that produced it, so the Context panel survives a restart. Timestamps are
RFC 3339 in UTC with microseconds, so two chats created in the same second still order
predictably. Writes are serialised behind one lock and the connection is shared by the server's
threads.

**Behaviour.** *New chat* inserts an empty conversation. The sidebar lists conversations by last
use. Opening one replays its messages and attachment metadata. A chat with no explicit name is
named from the first user message (a single line, ~60 characters) and never renamed afterwards
without asking. **Rename** is explicit and refuses an empty title. **Delete** removes the rows,
the messages and the stored files for that chat.

**Persistence.** Verified with the real server: a conversation with four messages and one
attachment was written, the server process was stopped and started again, and
`GET /api/chats/{id}` returned the same four messages and the same parsed attachment (see §12).

## 6. Attachments

**Formats.** `.pdf`, `.docx`, `.txt`, `.md` (`PHD_RAG_CHAT_ALLOWED_EXTENSIONS`). The extension,
the size (default 25 MiB) and the per-chat count (default 20) are validated before anything is
written, and each refusal is a sentence: what the file is, what the limit is, what to do.

**Storage path.** `data/chat/attachments/<chat id>/<attachment id><extension>`
(`PHD_RAG_CHAT_ATTACHMENT_DIR`). The path is built from identifiers this application generates;
the user's filename is stored as metadata for display only. Every path component is checked
against `^[0-9a-f]{8,64}(\.[a-z0-9]{1,8})?$` and the resolved path is checked to be inside the
attachment root, so `..\..\..\Windows\System32\evil.txt` becomes a displayed name
(`evil.txt`) and a file inside the chat's own directory.

**Parser reuse.** `app.parsing.parser.parse_document` handles PDF, DOCX and Markdown exactly as
the ingestion pipeline does. `.txt` is the one exception, and it is deliberate: RAG-003 decided
that `.txt` is not a corpus document format and a test asserts that `parse_document` still
rejects it ([tests/test_parser_dispatcher.py](tests/test_parser_dispatcher.py)), so the chat
decodes a text file directly into the same `ParsedDocument` shape instead of changing that
decision. No second parsing architecture was built.

**Parsed text.** Stored in `attachments.parsed_text` next to the metadata, so reopening a chat
needs no re-upload. A file that cannot be parsed is still stored, with
`text_status = failed` and the parser's own message; a file with no text is `empty`. Both are
shown in the chip and in the Context panel.

**Not corpus material.** The chat writes only under `data/chat/`; it never touches
`data/literature`, `data/my_work` or any other corpus role directory, and it never chunks,
embeds, indexes or classifies an attachment. A test asserts that the corpus tree is byte-for-byte
untouched by attaching a file ([tests/test_chat_attachments.py](tests/test_chat_attachments.py)).

## 7. Context assembly
**Configured window.** `chat_num_ctx = 262144` ([app/config.py:123](app/config.py#L123)); the
evidence path's 8192 is untouched and unused here. The value actually sent to Ollama is
asserted in a test (`GenerationOptions.num_ctx == 262144`) and confirmed live: the reported
`prompt_eval_count` from Ollama for a chat turn was 631 with `num_ctx` 262144.

**Model.** The chat uses `settings.generation_model` (`qwen3.5:35b`), or the model named by
`python -m app.web --model ...`, which now applies to both the chat and the modes
(`ModeRunner.model` is what `ChatRunner` receives). Confirmed live: `--model qwen3.8:27b` made
`GET /api/chat/info` report `qwen3.8:27b`. Temperature, top-p, seed, timeout, keep-alive and the
output cap all come from the same configuration the evidence path uses; they are not duplicated.

**Budget.** `chat_context_tokens = 236544` of input, `chat_max_output_tokens = 4096` of answer
headroom (4 096 since FIX1; see the FIX1 section) and a `chat_context_margin_tokens = 4096` safety
margin, so the input budget plus the answer and the margin can never exceed the window
(236544 + 4096 + 4096 = 244736 ≤ 262144). The
chat does not try to fill the window on every turn; it fills it in priority order and stops.

**Priority** (documented in [app/chat/context.py](app/chat/context.py)):

1. system instructions — never trimmed;
2. the current user message — never trimmed; if instructions plus message do not fit, the
   request is refused with a readable 413 rather than silently cut;
3. recent conversation — the newest turns, kept as a contiguous window (no gaps in the middle
   of a thread), within 25% of the budget;
4. attached working documents — whole documents wherever they fit, within 60% of the budget,
   fairly shared between several documents;
5. corpus material, only when it was asked for — whole excerpts in rank order, within 25%.

**Full documents.** An attachment is sent whole when it fits, and the leftover budget after the
other shares is handed back to it, so a large window is actually used. A document that must be
shortened is cut **at the end, at a paragraph (or sentence) boundary**, never in the middle, and
the cut says so inside the prompt: `[the rest of this document did not fit the context window
and was omitted: N tokens]`.

**Conversation history.** Previous turns are replayed as real `user`/`assistant` messages, so
the model sees the conversation, not a summary. The current turn is one final `user` message:
working documents, then corpus material, then the user's own message.

**Trimming is recorded.** Everything left out appears in `ContextReport.omitted` with its kind
(`history` / `attachment` / `corpus`), a label, the reason, the tokens and a sentence of
explanation, and is shown under **Context and debug**. After rendering, the assembled prompt is
measured again and shrunk (corpus first, then document text) if framing pushed it over the
budget, so the invariant "the request never exceeds the budget" is enforced, not assumed.

## 8. Simple UI

`python -m app.web` now opens the chat. Left sidebar: **+ New chat**, then the saved chats (title,
message and file counts), most recently used first, with the current one highlighted. Main area:
the conversation, the attached-file chips (name, size, how much text was read, `×` to detach),
the composer (**Attach** · message box · **Send**), a *Consult the PhD corpus* checkbox, one
unobtrusive line naming the model and the window, and a collapsed **Context and debug** panel.
`Ctrl+Enter` sends. The chat header carries **Rename** and **Delete**. Status stays in the top
bar; retrieval internals never appear in the chat.

**Advanced** in the top bar swaps the chat for the RAG-014 screen (and back), which is the same
markup and the same `app.js` as before.

## 9. Advanced compatibility

Unchanged and still callable: [app/modes/](app/modes), [app/retrieval/](app/retrieval),
[app/citations/](app/citations), [app/discovery/](app/discovery),
[app/generation/](app/generation), [app/api/modes.py](app/api/modes.py),
[app/web/static/app.js](app/web/static/app.js). The chat does not modify, wrap or bypass any of
them; it borrows the RAG-009 retrieval service through `ModeRunner.retrieve` so that the Qdrant
store is still opened once per process.

Confirmed live against the real stack (§12): `/api/modes` lists all five modes, a real
`/api/mode` **Evidence** run returned `status: ok` with 7 local sources and 6 validated Harvard
citations, and the chat's *consult the corpus* path reached the same pipeline (6 excerpts, 4,803
tokens). The pre-existing web, mode, retrieval, citation and discovery test files pass unchanged.

## 10. Internet and corpus behaviour

Internet discovery (RAG-010) is untouched and is never invoked by a chat turn: the chat has no
code path to `DiscoveryService`, and the only external call remains Research mode's metadata
query, reachable from Advanced and from the CLI. External records stay metadata-only and are
never merged into the local corpus.

The permanent corpus is consulted only when asked: the checkbox, or a message that asks in words
("what does the literature say about …", "search my corpus for …"). The trigger list is fixed
data in [app/chat/service.py](app/chat/service.py) — a phrase list, not a classifier — and an
ordinary conversational message never triggers retrieval (asserted in tests, and confirmed live:
"Give me three observations." produced `corpus_invoked: false` with no retrieval call at all).
Corpus excerpts are labelled with the RAG-011 wording: `evidence` is citable, `my_work` is the
author's own draft, `guidance` is supervision/rules material, and neither of the last two can be
presented as evidence.

## 11. Automated test results

```
.\.venv\Scripts\python.exe -m pytest -q
1173 passed, 2 warnings in 16.37s
```

* pre-existing tests: **1055 passed** (unchanged, none modified);
* RAG-015 tests: **118 passed** — store 11, attachments 17, context 20, service 25, API 31,
  retry 14 ([tests/test_chat_retry.py](tests/test_chat_retry.py), added by FIX1);
* total: **1173 passed, 0 failed, 0 errors**;
* warnings (all pre-existing, none caused by this repair): FastAPI's `TestClient` httpx
  deprecation notice, and qdrant-client's client/server compatibility check in `test_web_status`.
* the numbers above are the current ones; FIX1 added 20 tests to the 1153 reported when the
  repair first landed (see the FIX1 section for what changed).

## 12. Technical validation

The full transcript is in [docs/rag015-validation.txt](docs/rag015-validation.txt) (real server
on port 8015, real Ollama, real `qwen3.5:35b`, real Qdrant corpus).

| Requirement | Evidence |
|---|---|
| `num_ctx = 262144` on the new chat path | `GET /api/chat/info` → `num_ctx: 262144, context_tokens: 236544`; a real turn reported Ollama `prompt_eval_count = 631` and stored `context_report.num_ctx = 262144`; unit test asserts the sent `GenerationOptions.num_ctx` |
| The old 8192/6000 limits are not inherited | `chat_num_ctx != DEFAULT_GENERATION_NUM_CTX`, `chat_context_tokens > 20 × DEFAULT_GENERATION_CONTEXT_TOKENS`, `max_message_chars = 2 000 000 > MAX_QUERY_CHARS = 20 000` (tests) |
| A message beyond the old 20 000-character web limit is accepted | test posts 29 600 characters to `/api/chats/{id}/messages` → 200; the same text to `/api/mode` → 422, because the mode API is left as it was |
| Previous conversation turns reach later requests | follow-up turn: `history_messages_included: 2`; after restart: `4`; test asserts the exact `system,user,assistant,user,assistant,user` sequence |
| Attachment text reaches Qwen | DOCX attached (283 characters parsed); the model answered "Paragraph two argues that governance mechanisms shape the adoption of AI in organisations."; after a restart, with no re-upload, it answered which paragraph mentions the research questions |
| Persisted chat reload works | server stopped and restarted; `GET /api/chats` and `GET /api/chats/{id}` returned 4 messages and 1 parsed attachment; the stored report still reported `num_ctx: 262144` |
| Advanced mode still works | real `/api/mode` evidence run: `status: ok`, 7 local sources, 6 citations/references, Harvard-cited answer (2.4 s warm, 44.5 s cold) |
| Corpus bridge into the chat | `consult_corpus: true` → `corpus_invoked: true`, 6 excerpts, 4 803 corpus tokens, with attachment (71) and history (312) still present; the answer carried real citations |
| Deletion cleans up | `DELETE /api/chats/{id}` → `{"deleted": true}`; the chat list became empty and the chat's attachment directory was removed |
| The interface works in a real browser | driven with a real browser against the running server: a message was typed and sent, the answer rendered, an attachment was uploaded through **Attach** (138 B, "133 characters read", chip and sidebar count updated), detached again with `×`, and the conversation deleted with its files. Three screenshots in [docs/manual-images/](docs/manual-images) come from that session, and the **Context and debug** panel showed the stored report: window 262,144, input budget 236,544, headroom 8,192, instructions 562 |
| Output headroom preserved | `output_headroom_tokens = 4096` in every report; tests assert `input_budget + headroom ≤ num_ctx` and `total_tokens ≤ input_budget` |
| Trimming is recorded | tests force a small budget and assert `omitted` entries with kind, reason, tokens and a paragraph-boundary cut, and that the document's start survives while its end does not |

## 13. Known limitations

* **One turn at a time, shared with the Advanced pipeline.** A long chat answer holds the runner
  lock, so a mode run waits for it and vice versa. This is the existing RAG-014 rule (one Qdrant
  client per process); it is a wait, not a failure.
* **No streaming.** The answer appears when the model finishes; a long turn shows "the model is
  thinking…" rather than tokens.
* **The corpus trigger is a fixed phrase list.** A wording the list does not know is not
  detected; the checkbox is the reliable way. A false trigger costs one retrieval.
* **Uploads are base64 in JSON, not multipart**, so a 25 MiB attachment is ~33 MiB on the wire.
  That was chosen over adding `python-multipart` as a dependency; the payload is identical.
* **TXT is read as text here, not by the parser dispatcher**, because RAG-003 deliberately
  rejects `.txt` and that decision was left intact.
* **The model's thinking output is discarded**, per the brief: a reasoning trace is not part of
  the conversation.
* **Attachment parsing follows the project's OCR configuration.** A scanned PDF with OCR
  unavailable attaches with `text_status: failed` or `empty` rather than being refused.
* **Token counts in the report are the deterministic RAG-011 estimate**, not the model's
  tokenizer. Ollama's real `prompt_eval_count` is returned with each answer as well.
* **No cross-chat search and no export.** Deliberately out of scope for this repair.

---

## FIX1 — token repeat failure

### Reproduced or not

**Reproduced, exactly.** With `qwen3.5:35b` through real Ollama, corpus consultation enabled,
and the message "hi, can you please tell me about lattice?", one chat turn returned HTTP 500
with `{"error":"prediction aborted, token repeat limit reached"}`, not a context-window problem
(5 796 tokens sent into a 262 144-token window). The request was captured with a recording proxy
in front of Ollama, so the options in the transcript are what the application sent, not what it
intended: [docs/rag015-fix1-validation.txt](docs/rag015-fix1-validation.txt).

### Root cause

**Ollama's runner for that model was emitting one token forever, and Ollama's own repeat guard
aborted the prediction.** The chat request was not the cause, and neither was its size:

| Variable changed | Result |
|---|---|
| the chat's own prompt (4 800 tokens) | abort, content = 101 × `0` |
| a one-line system prompt, 20-token question | **OK** — "A quick brown fox leaps over a lazy dog." |
| the same one-line system prompt, 110-token prompt | abort |
| 380-token prompt, no system message | abort |
| `/api/generate` with `raw=true` (Ollama's chat template bypassed) | abort |
| `think: true` instead of `false` | abort |
| `temperature: 0.7` instead of `0.1` | abort |
| `num_ctx: 8192` instead of `262144` | abort |
| the **same long prompt** on `qwen3.8:27b` and on `qwen3-coder:30b` | **OK**, correct summaries |

So the failure was per-model and independent of prompt content, roles, templating, sampling,
`think`, or the context size: the `qwen3.5:35b` runner produced a constant token (the guard
stopped it after ~100 repeats), while other models on the same server answered the same prompt
correctly. The same app code and the same request had produced correct answers earlier the same
evening (18:53, and the FIX0 validation), and the model blob on disk is unchanged since
2026-10-06 — the variable was the **Ollama server process** (it had been restarted at 21:34).
Its log shows the engine taking an inconsistent initialisation path per load of that model:
some loads report "speculative decoding context initialized" and "the context supports bounded
partial sequence removal", most report "the context does not support partial sequence removal"
with no speculative context.

**Recovery that worked:** restarting the Ollama server process. Reloading the model did not help
(each reload produced a fresh, still-degenerate runner), unloading every model did not help, and
freeing 36 GB of VRAM (an orphaned `llama-server` holding 19 GB on GPU0 and 17 GB on GPU1) did
not help. After the server restart, the byte-identical failing request answered correctly.

### Exact request difference from RAG-011

Both paths failed at the same moment, so the comparison is like-for-like from the capture:

| | simple chat | RAG-011 evidence (`/api/mode`) |
|---|---|---|
| endpoint | `POST /api/chat` | `POST /api/chat` |
| model | `qwen3.5:35b` | `qwen3.5:35b` |
| messages | `system` + `user` (one turn) | `system` + `user` (one turn) |
| system prompt | chat prompts, 2 246 chars | RAG-011 prompts, 3 130 chars |
| think | `false` | `false` |
| temperature / top_p / seed | 0.1 / 0.9 / 0 | 0.1 / 0.9 / 0 |
| stop sequences | none | none |
| keep_alive | `30m` | `30m` |
| **num_ctx** | **262144** | **8192** |
| **num_predict** | **8192** | **900** |
| result | abort | abort |

The only differences were `num_ctx` and `num_predict`. `num_ctx` stays 262 144 (that is the
point of the repair); `num_predict` is now 4 096.

### Was think previously true or unspecified?

**Neither: it was already explicitly `false`, on the wire.** The captured request carries
`"think": false` from `OllamaClient.chat`, because `ChatService` passes `think=False` per call
(`app/chat/service.py`) rather than relying on the client's constructor default, and the client
then puts it in the payload. The repair keeps that, and adds a test that asserts the *actual*
Ollama JSON: `think is False`, `num_ctx == 262144`, `num_predict == 4096`, with the client
deliberately configured with `think=True` so the test cannot pass by inheritance.

### Fix applied

1. **Thinking is disabled in the request itself** (§3): `think: false` is set on every chat call
   and asserted against the serialised request payload in a test.
2. **Output cap lowered to 4 096 tokens** (§4), configurable with
   `PHD_RAG_CHAT_MAX_OUTPUT_TOKENS`. The 262 144-token window is unchanged; input budget and
   answer length stay independent settings (236 544 input + 4 096 answer + 4 096 margin).
3. **One automatic retry on that exact error** (§5): matched on Ollama's own phrase
   ("token repeat limit reached"), so no other failure is ever retried. The retry asks Ollama to
   drop the model first (`keep_alive: 0`) so the second attempt gets a genuinely fresh runner,
   then sends the same request once. The user message is not written again and no extra
   conversation turn is added.
4. **The retry is visible, never hidden**: `ContextReport.generation_retries` and
   `retry_reason` are stored with the assistant turn, returned by the API, and shown in the
   **Context and debug** panel; the failure payload carries `retries` and `retry_reason` too.
5. **Failed-turn UX (§6)**: a failed generation leaves exactly one user turn.
   `POST /api/chats/{id}/retry` answers the existing last user message (409 when the last turn
   already has an answer, 404 for an unknown chat), and the UI offers a **Retry this turn**
   button both in the error notice and when a reopened chat ends with an unanswered message.
   Nothing in the conversation is duplicated.

### Retry behaviour

Tested deterministically (14 tests in [tests/test_chat_retry.py](tests/test_chat_retry.py)):
one retry and only one; the retry recorded in the report; a successful retry yields one
assistant message and no second user message; a second failure surfaces Ollama's own error with
`retries: 1`; unrelated errors (model not found, connection refused, invalid request) are never
retried; the retry can be switched off with `PHD_RAG_CHAT_RETRY_ON_REPEAT_ABORT=false`; the
recovery also works for a client that cannot unload.

Live: a chat left over from the failed reproduction (three user turns, the last one unanswered —
the very duplication §6 describes) was answered through `POST /api/chats/{id}/retry` against
real Ollama: 200, a 712-character answer written for the *existing* user turn, no new user
message, and a second retry correctly refused with 409 `nothing_to_retry`.

Live through the interface, driven in a browser with Ollama deliberately unreachable for one
turn (the transcript is at the end of
[docs/rag015-fix1-validation.txt](docs/rag015-fix1-validation.txt)): the failed turn showed the
error notice with a **Retry this turn** button and stored one user message; reloading the page
offered the same button under "This message has no answer yet"; pressing it produced the answer
for that existing turn (stored: `user, assistant, user, assistant`, `num_ctx` 262144, zero
retries) and the notice cleared. A transport failure was **not** auto-retried, which is the
required behaviour: only the repeat abort is.

Two interface defects were found by driving the real page and fixed in the same change: a notice
with no action button rendered a stray "null" (a null child passed to `replaceChildren`), and
opening a chat cleared its "no answer yet" notice immediately after rendering it, so the Retry
affordance existed only on the failed turn and not in a reopened chat.

### Real three-turn validation

Against real Ollama and real `qwen3.5:35b`, through the simple-chat API
([docs/rag015-fix1-validation.txt](docs/rag015-fix1-validation.txt)):

| Turn | Message | Corpus | Result |
|---|---|---|---|
| A | "hi, can you please tell me about lattice?" | consulted (6 excerpts) | 200, no abort, 2 041-character answer about the LATTICE framework, `num_ctx` 262144, prompt 4 802 tokens, 365 generated, `generation_retries` 0 |
| B | "What are the main principles?" | consulted (6 excerpts) | 200, 3 121-character answer, `history_messages_included` 2, prompt 5 194 tokens |
| C | "Give me three observations only." | deliberately not consulted | 200, 496-character answer with exactly three numbered observations, `corpus_invoked` false, `history_messages_included` 4 |

Stored conversation after the three turns: `user, assistant, user, assistant, user, assistant` —
**one user message per turn**, `num_ctx` 262 144 on every assistant turn, `generation_retries` 0
throughout. The captured post-fix request is `think: false`, `num_ctx: 262144`,
`num_predict: 4096`, and includes the earlier turns as real `user`/`assistant` messages.

### Test count

```
.\.venv\Scripts\python.exe -m pytest -q
1173 passed, 2 warnings
```

1055 pre-existing tests, 118 RAG-015 tests (FIX1 added 20 of them: the retry and failed-turn
behaviour, plus the request-shape and output-cap assertions added to the existing context and API
files). No failures.

### Remaining limitation

The trigger is inside Ollama's engine, not in this application: when its runner for
`qwen3.5:35b` degenerates, *every* real prompt fails — including the RAG-011 evidence path, which
this repair does not own — and no request-side setting avoids it. The one automatic retry gives
a poisoned runner a second chance (with a fresh load), but if the engine is deterministically
broken the second attempt fails too and the real error reaches the user unchanged. The reliable
remedy is to restart the Ollama server, which is documented in the manual's troubleshooting
section; on this workstation that restored healthy generation immediately, with the chat and the
Advanced modes both working afterwards.

## FIX2 — turn latency

Asked for after FIX1: "can you just make it faster". Not a new stage, and not a new feature —
the simple chat was measured, one cheap lever was pulled, and the rest was left alone.

### Where the time actually goes

Three real turns against real Ollama and `qwen3.5:35b` through the simple-chat API. The
"Ollama wall" column is the chat's own `wall_seconds`, which measures the HTTP request to
Ollama only ([app/generation/ollama.py:320](app/generation/ollama.py#L320)); "outer wall" is a
stopwatch around the whole API call, so it includes retrieval setup and anything else spent
before the model is asked.

| Turn | Prompt | Prompt tokens | Generated | Answer | Ollama wall | Outer wall |
|---|---|---|---|---|---|---|
| FIX1 turn A, `system_v1` | "hi, can you please tell me about lattice?" (corpus) | 4 802 | 365 | 2 041 chars | — | — |
| FIX2 probe 1, `system_v2` | the same question (corpus) | 4 832 | 294 | 1 820 chars | 6.2 s | 18.6 s — first request after a process restart |
| FIX2 probe 2, `system_v2` | "What are the main principles?" (corpus, warm) | 4 840 | 329 | 1 827 chars | 2.0 s | **2.7 s** |

What that says:

- **A warm turn is 2.7 seconds end to end**, ~165 tokens/s of generation, for a ~1 800-character
  answer. The chat is not slow once the model is loaded: `keep_alive` is `30m`
  ([config.py:116](app/config.py#L116)) and the chat client sends it, so consecutive turns never
  reload the model.
- **The first request after a process start is the slow one, and only ~6 of its 18.6 seconds were
  Ollama** (a cold model load plus prompt eval). The remaining ~12 seconds happened *before* the
  Ollama call: the first corpus turn on a freshly started process also pays retrieval setup.
  Probe 2, with the same prompt size, confirms it is a one-off.
- **Prompt eval is not the bottleneck**: the prompt is ~4 840 tokens (system ~590, corpus
  ~4 800 across 6 excerpts) and costs a few seconds at most.
- **Nothing here is a context-window problem**: `num_ctx` is 262 144 on every turn.

### Change applied

One lever, in two files plus a test, and nothing else:

- [app/chat/templates/system_v2.txt](app/chat/templates/system_v2.txt) — new system prompt: the
  v1 text plus a brevity rule ("be brief by default … do not restate the question, do not
  enumerate every possibility", while still following an explicit length, count or format).
- [app/chat/prompts.py](app/chat/prompts.py) — `SYSTEM_TEMPLATE` is now `system_v2`; the files are
  content-addressed, so the version string changed with it. `system_v1.txt` stays in the tree as
  the recorded iteration, exactly as RAG-011 kept its superseded prompt.
- [tests/test_chat_service.py](tests/test_chat_service.py) — the assertion that checks the system
  prompt no longer forces citation machinery follows the new wording, and a new test asserts the
  prompt asks for short answers.

Deliberately unchanged: `num_ctx` 262 144, the 4 096-token output cap, the 6 corpus excerpts,
the context budgeting and trimming, the Ollama request shape, and the whole retrieval stack.

**Measured effect of the brevity rule: about 10–20 %.** Answers went from 365 generated tokens
(2 041 characters) to 294 and 329 tokens (1 820 and 1 827 characters) on the same question. That
is a real but modest saving, and it corrects an estimate made while this work was in progress,
which claimed brevity had roughly halved generation time; the FIX1 token counts above show it did
not. The honest conclusion is that the felt slowness was never answer length — it was the cold
model load and the first-turn retrieval setup, both of which are one-off costs.

### Verification

```
.\.venv\Scripts\python.exe -m pytest tests -q -k chat
119 passed, 1055 deselected, 1 warning
```

The full suite was not re-run for FIX2 because only chat prompt files and one chat test file
changed; FIX1's full run was 1 173 passed, and FIX2 adds one chat test (118 → 119). Live checks
against the running app: `/api/chat/info` reports
`prompt_version: system_v2:8618b6c4+user_v1:da447090`, `num_ctx: 262144`,
`max_output_tokens: 4096`, `corpus_retrieval_available: true`. Both probe chats were deleted
afterwards, so no test conversation is left in the user's chat list.

### Considered and not done

- **Streaming** — would not make the model faster by a single millisecond; it only makes text
  appear as it is written, so a 2.7-second answer stops looking like a stall. It needs Ollama
  streaming, a server-sent-events endpoint and a rewrite of the chat front end, and it puts risk
  on a screen that currently works. Offered; declined for now.
- **`OLLAMA_SCHED_SPREAD` / flash attention** — tidying these at the Ollama level might give a
  large multiple on generation, or might be much slower if the model stops fitting: that
  262 144-token KV cache is tens of gigabytes, so splitting the 35B across both GPUs may be
  forced rather than chosen. It needs an Ollama restart and an untested guess, so it was left
  alone.
- **Warming retrieval at startup** — would remove the ~12-second first-corpus-turn cost recorded
  in probe 1. Offered, not implemented, because it was not asked for.
- **Lowering the output cap further** — pointless: the cap never binds on ordinary conversation
  (294–329 tokens generated against a 4 096 allowance).

### Remaining limitation

The figures above are single samples on one workstation, not a controlled A/B, and they were
taken with a freshly restarted Ollama server, so FIX1's precondition still holds. Brevity is an
instruction, not a switch: ask for depth and the answers get long and slow again. The one-off
cold-start cost (model load, plus retrieval setup on the first corpus turn) remains the largest
single wait a user meets, and only the startup warm-up listed above would remove it.

