# RAG-016 — report

**Task:** add the DeepSeek API as a manually selected cloud model for the existing simple chat,
so work can continue in the same locally saved conversation when local Ollama is unreliable.

Status: implemented, then extended by **RAG-016-FIX1** (both current DeepSeek models, and the
provider's real context window — see the last section). `1217 passed` (the whole suite, including
the tests added here).

---

## 1. What changed

One thing: the simple chat can now be answered by **two providers**, chosen per message.

| | Local (default) | Cloud |
|---|---|---|
| provider id | `ollama` | `deepseek` |
| model | `qwen3.5:35b` (configured) | `deepseek-flash` — *DeepSeek V4.1 Flash* (the default) and `deepseek-v4-pro` — *DeepSeek V4 Pro* (selected by hand) |
| sends data off the machine | no | only the context selected for that one message |

Nothing else moved. The chat's storage, context assembly, prompt, corpus retrieval, attachments
and the whole RAG-009 → RAG-014 Advanced path are untouched; there is no new architecture layer,
no provider abstraction for Advanced, no routing, no failover and no OpenAI.

## 2. Files created

| File | What it is |
|---|---|
| `app/generation/deepseek.py` | the DeepSeek client: one HTTP call, model-agnostic (FIX1: the same client answers with either model), failures mapped to readable sentences, secrets redacted from every error path |
| `app/chat/providers.py` | the provider registry: the two provider ids, the models each offers, their display names, and the privacy sentence for each |
| `tests/test_generation_deepseek.py` | 18 tests: request shape, normalised result, reasoning never read, every failure kind, key redaction, model switching |
| `tests/test_chat_providers.py` | 21 tests: the ten checks §23 asks for, plus the migration, the selector payload and the local retrieval path |

## 3. Files modified

| File | Change |
|---|---|
| `app/config.py` | DeepSeek settings and defaults; the documented `DEEPSEEK_*` names and the project's `PHD_RAG_DEEPSEEK_*` names are both read, prefix winning |
| `app/chat/store.py` | additive schema migration with a timestamped backup, `provider`/`model` columns, `ChatMigration` record |
| `app/chat/models.py` | `ChatMessage.provider`/`model`, provider fields on `ContextReport`, `ChatAnswer.provider`, two new error types |
| `app/chat/service.py` | per-turn provider selection, provider-aware context budget, provider metadata on the report, cloud requests never auto-retried |
| `app/api/chats.py` | `provider` on the message body and the retry body, `providers` in `/api/chat/info`, provider fields in the answer JSON |
| `app/generation/ollama.py` | four optional fields on `ChatResult` (`provider`, `cached_input_tokens`, `reasoning_tokens`, `request_id`) |
| `app/generation/models.py` | `ModelFailure.detail` — the provider's own words, redacted, for the debug panel only |
| `app/web/static/index.html`, `chat.js`, `style.css` | the model selector, the privacy line, the per-answer badge, the provider rows in **Context and debug** |
| `README.md`, `docs/manual.md` | configuration, the selector and the DeepSeek failure rows |
| `tests/chat_support.py` | the `FakeCloudModel` double and a `cloud_model` hook in the builders |

## 4. Database migration and backup

The chat database gained two nullable columns on `messages` — `provider` and `model`.

* **Detect:** on open, `PRAGMA table_info(messages)` is compared with the columns RAG-016 needs.
* **Back up:** only when a column is actually missing, and before the first `ALTER`, the file is
  copied with SQLite's own backup API to
  `data/chat/backups/chat-before-rag016-YYYYMMDD-HHMMSS.sqlite3` (next to whatever database
  `PHD_RAG_CHAT_DB_PATH` names).
* **Alter:** `ALTER TABLE messages ADD COLUMN …` in one transaction, wrapped so any failure
  rolls back, leaves the original database exactly as it was and raises
  `ChatMigrationError`, which stops startup with the reason and the backup path.
* **Never:** no table is recreated or dropped, no message or attachment is deleted, no chat id
  or conversation is regenerated, no history is cleared.

A database created by RAG-015 therefore opens and behaves exactly as before: its old messages
have no provider, load with `provider = None`, and the UI simply shows no badge for them. A
second start finds the columns present, changes nothing and copies nothing
(`test_legacy_chat_loads_after_an_additive_migration`).

## 5. DeepSeek provider

`app/generation/deepseek.py`, ~270 lines, `requests` only (no SDK, no framework). It takes the
same `ChatMessage` sequence and `GenerationOptions` the Ollama client takes, and returns the same
`ChatResult`, so the service does not have two shapes to handle.

* endpoint `{DEEPSEEK_BASE_URL}/chat/completions`, `stream: false`, `max_tokens` from the
  turn's budget, temperature/top_p from the shared generation settings;
* the key travels as an `Authorization` header and nowhere else — never in a body, a log, an
  error or a response;
* `reasoning_content` is **never read**; a reasoning-token *count* is carried in usage metadata
  and nowhere else, so no chain of thought can reach SQLite or the conversation;
* **two models, one client** (FIX1): `use_model()` points the same client at `deepseek-flash` or
  `deepseek-v4-pro` for the next call, so switching model costs nothing and no state is rebuilt;
* `available_models()` asks `/models` at most once per process, with a 5-second cap and no
  failure. Since FIX1 that answer is *reported* (`reported_models`) and never authoritative: it
  cannot hide a configured model, and it cannot add one either, so the selector always holds the
  models this installation is set up for;
* failures are mapped to one useful sentence each — authentication, insufficient balance, rate
  limit, invalid model, 5xx outage, invalid request, timeout, connection failure, malformed body,
  empty answer — with the provider's own (redacted) words kept in `detail` for the debug panel.

## 6. Model selector

One small row in the existing composer, built entirely from `/api/chat/info`, grouped by
provider and offering every configured model (FIX1):

```
Model   [ Qwen 3.5 35B — 🔒 Local ▼ ]
        🔒 Local - processing remains on this computer · Qwen 3.5 35B

        Local
          Qwen 3.5 35B          🔒 Local
        DeepSeek
          DeepSeek V4.1 Flash   ☁ Cloud     deepseek-flash — the default
          DeepSeek V4 Pro       ☁ Cloud     deepseek-v4-pro
```

Choosing a cloud model changes the line to
`☁ Cloud - this request's selected context will be sent to DeepSeek · DeepSeek V4 Pro`, and the
hint under the composer follows the selected provider's own context window (262,144 local,
1,000,000 cloud). No modal, no per-request dialogue, no settings dashboard, no page redesign.
Without a key the cloud entries are listed, greyed out and unselectable.

Each entry carries `provider:model`, and the label is the server's display name for that model id
(`DeepSeek V4.1 Flash`, `DeepSeek V4 Pro`, `Qwen 3.5 35B`); a configured id the table does not
know is shown by its own id rather than hidden.

## 7. Manual provider *and* model switching

The choice applies to the next assistant answer and is stored with it (`messages.provider`,
`messages.model`). A chat is never bound to a model: `local → flash → pro → local` inside one
conversation produces four answers, four badges and one chat, and every model sees the same
locally stored history. Changing model never starts a new chat — the handler only updates the
selection, the privacy line and the hint.

`/api/chat/info` describes the providers and their models; `POST /api/chats/{id}/messages` and
`POST /api/chats/{id}/retry` accept `{"provider": "ollama" | "deepseek", "model": "<id>"}`
(provider aliases `local`, `cloud` are accepted; `model: null` means the provider's default). An
unknown provider or model is a 400 with the offered ids listed, and is refused *before* the
message is stored.

## 8. Retry using another provider or model

RAG-015's failed-turn logic is extended, not replaced: the user message is written first, a
failure leaves exactly one unanswered user turn, and **Retry this turn** answers that existing
turn — now with whichever provider *and* model is selected.

* Ollama fails → the message stays saved, the notice offers Retry, selecting a DeepSeek model and
  pressing Retry writes one assistant answer for the existing user message;
* DeepSeek fails → the same in the other direction, including from Flash to Pro;
* the database ends as `user, assistant`, never `user, user, assistant`, and the chat is the same
  chat it was before.

**No automatic failover in either direction.** A failed local turn is never escalated to the
cloud (`cloud.calls == []` in the test), and a failed cloud turn is never retried automatically:
one request, one saved message, control back to the user. Ollama's existing token-repeat retry
still fires once, for Ollama only, because it is a local runner fault and not a provider choice.

While browser-testing this path a pre-existing display bug surfaced and was fixed: `applyAnswer`
re-appended the user message on a retry, so the retried turn showed the user's text twice on
screen even though the database held it once. `retryLast` now appends the answer only.

## 9. What stays local

Everything that was local before: the SQLite chat database and every message, attachments and
their parsed text, the Qdrant index and embeddings, the permanent PhD corpus, retrieved
literature, supervision material, writing rules, the BGE-M3 embedder and the reranker. DeepSeek
is a generation provider and nothing else — no file API, no remote threads, no remote vector
store, no cloud storage of any kind. A cloud turn is retrieved locally first, and only the
selected excerpts travel.

## 10. What is sent to DeepSeek

Exactly the context the local application already assembled for that one message, and nothing
more:

* the chat system prompt (unchanged from RAG-015, no provider-specific academic prompt);
* the relevant part of the conversation history, whichever model wrote it;
* the parsed text of the documents attached to the chat;
* the corpus excerpts retrieved for that message, when the corpus was consulted;
* the current user message.

The corpus, the index, the other chats and the original binary files are not sent. The context
builder is the same one (`app/chat/context.py`); a cloud turn simply budgets against the cloud
window by handing it those numbers, so there is no second context system.

## 11. API-key security

The key exists only in the server process. It is read from `DEEPSEEK_API_KEY` (or
`PHD_RAG_DEEPSEEK_API_KEY`), never written to SQLite, never returned by `/api/chat/info` or any
other endpoint, never placed in frontend JavaScript, never printed in a log and never included in
an error: every failure string passes through `redact()`, which removes the configured key and
any `Bearer …` token. A redacted `detail` is exposed for the debug panel and the logs, while the
user sees one readable sentence. The file contains no key and this report contains none.

## 12. Smoke-check results

Ollama was not running and no DeepSeek key exists in this environment, so the §24 checks were run
against the **real application** — real `OllamaClient`, real `DeepSeekClient`, real SQLite, real
endpoints — pointed at two local stand-in servers that speak each provider's wire protocol
(`/api/chat`, `/chat/completions`, `/models`, header authentication). No private material was
used. 17/17 checks passed:

| Check | Result |
|---|---|
| selector lists both providers and their models | PASS |
| **TEST 1** local `Reply exactly: LOCAL OK` | PASS — `LOCAL OK` |
| **TEST 2** DeepSeek `Reply exactly: DEEPSEEK OK` | PASS — `DEEPSEEK OK` |
| **TEST 3** local remembers `BLUE SHIP`, cloud answers the recall in the same chat | PASS — `BLUE SHIP`, and the request carried the local turn |
| **TEST 4** local fails, user selects DeepSeek, **Retry this turn** | PASS — one user message, one answer, no duplicate |
| no automatic cloud call on a failed local turn | PASS |
| key never returned by any endpoint | PASS |
| a rejected key gives a sentence, not a trace | PASS |

The same four scenarios were then driven **through the browser** against the real UI: the
selector renders with every entry, the privacy line switches with the selection, the badges name
the model that answered, `BLUE SHIP` came back from the cloud turn, the retry-after-failure path
produced one user turn plus one cloud answer, and **Context and debug** showed provider, model,
input/cached/output/reasoning tokens, latency and request id — only the fields that were
reported.

Not executed here: a live call to `api.deepseek.com` with a real key, and a live Ollama run. Both
need credentials or a local model this machine does not have.

## 13. Known limitations

* **Real-provider validation is outstanding.** The wire protocol, the normalisation and every
  failure mapping are tested, but the first live DeepSeek call (and its real token accounting)
  still has to be made once a key is configured.
* **The cloud window is a setting, not a query.** `PHD_RAG_DEEPSEEK_NUM_CTX` (default 1,000,000
  since FIX1) is what the builder budgets against; a model with a smaller real window would still
  be refused by the provider with a readable 400, and the input budget is what makes that unlikely
  (96,000 tokens, unchanged).
* **The model list is configuration.** The selector offers `DEEPSEEK_MODEL` and
  `DEEPSEEK_MODELS`; what the provider reports is shown but never selects or unselects a model,
  so a newly released id is added by naming it in the configuration, not automatically.
* **Usage, not billing.** Token counts are shown only as the API reports them; no price is
  computed and no account panel exists, by design.
* **Reasoning is a number.** A reasoning model's trace is deliberately never requested, read or
  stored; only the billed reasoning-token count can appear.
* **DeepSeek is for the chat only.** The Advanced modes still answer with the local model, as
  §21 requires.

---

# RAG-016-FIX1 — both DeepSeek models, and the real context window

## What changed

1. **Two DeepSeek models instead of one.** `DEEPSEEK_MODEL` now defaults to `deepseek-flash`
   (*DeepSeek V4.1 Flash*), and `DEEPSEEK_MODELS` defaults to `deepseek-v4-pro`
   (*DeepSeek V4 Pro*), so both are offered out of the box, in that order, with the default
   first. `deepseek-chat` appears nowhere: not as a default, not as a fallback, not in the
   configuration, not in the display-name table, not in the client's own default.
2. **The selection is provider *and* model.** The message and retry bodies gained `"model"`, the
   selector's entries became `provider:model`, and `ChatService.client(provider, model)` resolves
   and validates both. One client serves both models — `DeepSeekClient.use_model()` points it at
   the next one, so switching costs nothing.
3. **The selector reads as asked:**

   ```
   Local
     Qwen 3.5 35B          🔒 Local
   DeepSeek
     DeepSeek V4.1 Flash   ☁ Cloud
     DeepSeek V4 Pro       ☁ Cloud
   ```

   Display names come from a presentation table (`DeepSeek V4.1 Flash`, `DeepSeek V4 Pro`,
   `Qwen 3.5 35B`); an id that is not in it is shown by its id, never hidden. Switching between
   Flash and Pro updates only the privacy line and the hint — it does not create a chat, reload
   the conversation, or touch what is already stored.
4. **The context window is the provider's current maximum.** `PHD_RAG_DEEPSEEK_NUM_CTX` went from
   128,000 to **1,000,000**. Nothing else about the budgeting moved: the input budget for one
   turn is still `PHD_RAG_DEEPSEEK_CONTEXT_TOKENS` = 96,000 tokens, so the change removes an
   obsolete ceiling and does not make any request larger. The old value was never binding either
   — `budget = min(context_tokens, num_ctx − headroom − margin)` was 96,000 before and after —
   but the debug panel and the hint line were reporting a window the current models do not have.
5. **`/models` no longer decides the selector.** It is still asked (once per process, 5-second
   cap, failure tolerated) and its answer is reported as `reported_models`, but it can neither
   hide a configured model nor add one, so *DeepSeek V4 Pro* stays selectable even when a listing
   omits it — which is exactly what the stand-in provider in the checks below did.

Minimum tests only: five added, and the ones that named the old default updated
(25 in `tests/test_chat_providers.py`, 18 in `tests/test_generation_deepseek.py`).

## Files touched

| File | Change |
|---|---|
| `app/config.py` | `DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"`, `DEFAULT_DEEPSEEK_MODELS = ("deepseek-v4-pro",)`, `DEFAULT_DEEPSEEK_NUM_CTX = 1_000_000` |
| `app/chat/providers.py` | `ModelOption`, per-provider groups, display names, `cloud_models()` from configuration only, `resolve_model()` |
| `app/chat/models.py` | `UnsupportedModelError` (400) |
| `app/chat/service.py` | `model` through `send_message`/`retry_last`/`client`, `use_model` on the client, `reported_models` |
| `app/api/chats.py` | `model` on both request bodies; `/api/chat/info` returns grouped models with labels and `num_ctx` |
| `app/generation/deepseek.py` | default model `deepseek-flash`, `use_model()` |
| `app/web/static/chat.js` | grouped selector, `provider:model` values, display names in the badges and the hint |
| `README.md`, `docs/manual.md` | the new defaults, the grouped list, the window, two new troubleshooting rows |
| `tests/chat_support.py` | the fake records the model each call was made with, and supports `use_model` |

No schema change, so no migration and no new backup.

## Verification

`1217 passed` — the whole suite.

FIX1 checks, run against the **real application** (real clients, real SQLite, real endpoints) with
two stand-in providers — and the cloud stand-in deliberately reporting only *one* of the two ids
from `/models`:

| Check | Result |
|---|---|
| both models offered, display names, default first | PASS — `deepseek-flash` / `deepseek-v4-pro` |
| `deepseek-flash` is the default with no model named | PASS |
| Flash answers `Reply exactly: FLASH OK` | PASS — label `DeepSeek V4.1 Flash` |
| Pro answers `Reply exactly: PRO OK` in the **same** chat | PASS — label `DeepSeek V4 Pro`, still one chat, six messages |
| stored metadata per turn | PASS — `deepseek-flash`, then `deepseek-v4-pro` |
| selecting Pro starts no new chat | PASS — one entry in the chat list |
| the window is 1,000,000 while the input budget stays 96,000 | PASS — total sent 649 tokens |
| a model the provider does not offer is refused | PASS — 400, nothing stored |
| retry answers the failed local turn with Pro | PASS — one user message, one answer |

Browser checks on the real UI confirmed the grouped selector
(`Local / Qwen 3.5 35B`, `DeepSeek / DeepSeek V4.1 Flash`, `DeepSeek / DeepSeek V4 Pro`), the
privacy line and hint following the selection, the per-answer badges naming the model, and
**Context and debug** showing `deepseek-v4-pro` with a 1,000,000 window and a 96,000-token input
budget.

Still not executed: a live call to `api.deepseek.com` with a real key.
