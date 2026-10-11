# RAG-010 — Controlled Internet-based academic discovery

Status: **complete** — read-only external scholarly discovery (Crossref + OpenAlex,
optional Semantic Scholar) that normalizes, merges, verifies, audits and compares
every result against the local corpus, keeping external records strictly separate from
trusted local evidence. 143 new tests; 657 tests pass in total.

Scope guard: RAG-010 searches, verifies and reports. It does **not** download anything
into `data/`, chunk, embed, index, import or mark Internet results as literature
evidence, does not generate prose and does not begin RAG-011.

---

## 1. Files created

| Path | Purpose |
|------|---------|
| [app/discovery/__init__.py](app/discovery/__init__.py) | Package exports (with a lazy `research_discover` so `-m app.discovery.research` stays clean) |
| [app/discovery/models.py](app/discovery/models.py) | `DiscoveryStatus`, `FulltextStatus`, `LocalMatchKind`, `FieldConflict`, `ProviderFailure`, `LocalMatch`, `ExternalWork`, `DiscoveryFilters`, `ProviderQuery`, `DiscoveryAudit`, `DiscoveryResult`, `LocalDiscoveryResult`, `CombinedDiscovery` |
| [app/discovery/cache.py](app/discovery/cache.py) | `DiscoveryCache` — TTL over RAG-006's `JsonCache`, versioned keys, hit/miss counters |
| [app/discovery/providers/base.py](app/discovery/providers/base.py) | `ProviderResponse`, `AcademicDiscoveryProvider` protocol, `failure_from_http`, `record_missing_fields` |
| [app/discovery/providers/crossref.py](app/discovery/providers/crossref.py) | `CrossrefProvider`, `work_from_message` |
| [app/discovery/providers/openalex.py](app/discovery/providers/openalex.py) | `OpenAlexProvider`, `work_from_work`, inverted-abstract reconstruction |
| [app/discovery/providers/semanticscholar.py](app/discovery/providers/semanticscholar.py) | Optional `SemanticScholarProvider` that degrades cleanly on 429 |
| [app/discovery/providers/__init__.py](app/discovery/providers/__init__.py) | `build_provider` factory |
| [app/discovery/merge.py](app/discovery/merge.py) | `same_work`, `merge_records`, field precedence + provenance, material-conflict capture, `status_for`, `identity_confidence` |
| [app/discovery/ranking.py](app/discovery/ranking.py) | Deterministic external ranking (title 0.55 / abstract 0.20 / completeness 0.15 / recency 0.10) |
| [app/discovery/local.py](app/discovery/local.py) | `LocalDocument`, `LocalCorpusIndex`, payload adapters — DOI → title → fuzzy title+year+author matching against the corpus |
| [app/discovery/fulltext.py](app/discovery/fulltext.py) | Opt-in `verify_fulltext` with SSRF controls, manual redirect validation, size cap, content-type allowlist, identity evidence |
| [app/discovery/service.py](app/discovery/service.py) | `DiscoveryService.discover/lookup_doi/lookup_title/find_related/verify_fulltext/describe` |
| [app/discovery/render.py](app/discovery/render.py) | Grouped, human-readable rendering (`ALREADY IN LOCAL CORPUS` / `NEW-VERIFIED` / `DISCOVERY ONLY` / `CONFLICTS`) |
| [app/discovery/research.py](app/discovery/research.py) | `research_discover` combined service + `python -m app.discovery.research` CLI |
| [app/discovery/search.py](app/discovery/search.py) | `python -m app.discovery.search` CLI |
| [tests/discovery_support.py](tests/discovery_support.py) | Offline harness: queued/routing fake transports, fake binary fetcher, synthetic provider payloads, synthetic local corpus, isolated cache |
| [tests/test_discovery_providers.py](tests/test_discovery_providers.py) | Provider adapters, normalization, timeout/429/5xx/malformed handling (27 tests) |
| [tests/test_discovery_merge.py](tests/test_discovery_merge.py) | Dedup, provenance, conflicts, status, determinism (17) |
| [tests/test_discovery_local.py](tests/test_discovery_local.py) | Local-corpus matching and evidence safety (15) |
| [tests/test_discovery_cache.py](tests/test_discovery_cache.py) | TTL, refresh, key design (13) |
| [tests/test_discovery_service.py](tests/test_discovery_service.py) | Pipeline, audit trail, isolation, filters, lookups (20) |
| [tests/test_discovery_security.py](tests/test_discovery_security.py) | SSRF controls, redirects, oversize, identity rules (23) |
| [tests/test_discovery_combined.py](tests/test_discovery_combined.py) | Local/external separation and outage behaviour (9) |
| [tests/test_discovery_cli.py](tests/test_discovery_cli.py) | Rendering, grouping, CLI flags, JSON, verification display (19) |
| [docs/rag010-discovery.txt](docs/rag010-discovery.txt) | Real-query validation: 5 required + 2 recent queries, per-query audit |
| [docs/rag010-local-match-validation.txt](docs/rag010-local-match-validation.txt) | Known-paper identification, performance, local-vs-Internet test, live full-text verification |
| [docs/rag010-recent-literature.txt](docs/rag010-recent-literature.txt) | Recent-literature inspection (titles, venues, abstracts, local status) |
| [RAG-010-report.md](RAG-010-report.md) | This report |

## 2. Files modified

| Path | Change |
|------|--------|
| [app/config.py](app/config.py) | RAG-010 settings block (`discovery_*`, `openalex_email`, `openalex_api_key`) plus `_parse_csv`; the polite-pool contact is read from `PHD_RAG_CROSSREF_MAILTO` (also accepting the RAG-006 name) — never hard-coded |
| [app/metadata/external/http.py](app/metadata/external/http.py) | *Additive*: `HttpResult.headers`, `text_snippet`, `retry_after` and the response helpers behind them, so a provider failure can quote the provider instead of inventing a reason |
| [app/metadata/external/cache.py](app/metadata/external/cache.py) | *Additive*: `JsonCache.stored_at`, `age_seconds`, `delete` — the TTL layer needs them; existing callers unchanged |
| [README.md](README.md) | Discovery CLIs and the local/external separation principle |
| [.gitignore](.gitignore) | `.discovery_cache/` |

No file under `data/` was modified (newest corpus mtime is still 2026-10-10, i.e. before this stage), and no Qdrant collection was read-modified — discovery only reads payloads.

## 3. Dependencies added

**None.** RAG-010 reuses what the project already has: `requests` (through RAG-006's
HTTP client), `pydantic` models, `PyMuPDF` for identity text extraction from a fetched
PDF, and RAG-006's `normalize_doi` / `doi_cache_key` / `title_similarity` /
`clean_external_text`. No new provider SDK, no ORM, no HTTP framework.

## 4. Provider architecture

```
DiscoveryService
  ├── provider selection (names from settings or the request)
  ├── per-provider isolation     ← a failure is recorded, never raised
  ├── DiscoveryCache             ← provider + request version + query + filters
  └── AcademicDiscoveryProvider protocol
        search(query, limit, filters) -> ProviderResponse
        lookup_doi(doi)               -> ProviderResponse
        lookup_title(title, limit)    -> ProviderResponse
        provider_name
```

A `ProviderResponse` carries records **and** the reason it produced none (`failure`),
plus `cache_hit`, `http_requests` and `seconds`. That is the whole point: "the API said
no" and "the API refused us" must never look the same. Unknown provider names raise
`DiscoveryError` (a typo must not silently search nothing). Providers are built by
`build_provider(name, ...)`, and every HTTP call goes through RAG-006's
`HttpClient` (timeout, bounded retry with backoff, 429/5xx handling, rate limiter).

## 5. Normalized external record model

`ExternalWork` is the one internal shape every provider maps into:

```
title, authors, year, doi, url, venue, publisher, publication_type, abstract,
cited_by_count, open_access, open_access_status,
fulltext_url, fulltext_status, fulltext_verified, fulltext_sha256, fulltext_bytes,
fulltext_content_type, fulltext_fetched_at, fulltext_source_url, fulltext_note,
providers, raw_provider_ids, field_provenance, source_provenance,
conflicts, discovery_status, identity_confidence, relevance_score,
local_match, missing_fields
```

* `field_provenance` records **which provider supplied each field** (a preference, not
  an authority claim); `source_provenance` and `raw_provider_ids` keep the provider's
  own identifiers.
* `missing_fields` lists what no provider supplied. Nothing is inferred or filled in:
  Crossref has no open-access flag and no citation count here, so those stay `None`.
* `is_evidence` is hard-wired to `False`; there is no code path that turns an external
  record into corpus evidence.

## 6. Verification / status model

| Status | Meaning |
|--------|---------|
| `LOCAL_MATCH` | Reserved for the local half; external records never claim it |
| `EXTERNAL_DISCOVERY` | Found externally, identity not corroborated |
| `EXTERNAL_METADATA_VERIFIED` | Corroborated (≥ 0.80 deterministic identity confidence) |
| `EXTERNAL_FULLTEXT_VERIFIED` | The bytes were actually fetched and carried the record's DOI or ≥ 80 % of its title words |
| `UNAVAILABLE` | No title and no DOI — nothing to identify |
| `CONFLICT` | Providers materially disagree (title, year beyond ±1, first-author family) |

`identity_confidence` is a fixed, explainable sum: DOI +0.35, direct DOI lookup +0.15,
title +0.15, authors +0.10, year +0.05, venue/publisher/type +0.05, ≥ 2 providers
agreeing +0.15, capped at 1.0. Status is derived from it, never guessed; a conflict
wins over the threshold, and both conflicting values are preserved in `conflicts`.

`FulltextStatus` (`FULLTEXT_OPEN` / `LANDING_PAGE_ONLY` / `UNKNOWN`) says what the
provider *claimed*; `fulltext_verified` + `fulltext_sha256` say what was *retrieved*.
The two are never conflated.

## 7. Crossref implementation

* Search: `query.bibliographic`, `rows`, a `select` field list, `query.author`, a
  `filter` of `from-pub-date`/`until-pub-date`/`type`, `sort=published&order=desc` for
  recent searches, and `mailto` when a contact is configured.
* DOI lookup: `/works/{doi}` — **without** `select`, because Crossref rejects that
  parameter on this route (found by validation; see §17/§22).
* Title lookup: bibliographic search, then a local `title_similarity ≥ 0.90` filter.
* A full-text URL is offered only when Crossref also reports an open licence
  (`creativecommons.org` / `cc-by` / `cc0` / `openaccess`); otherwise the record is
  `LANDING_PAGE_ONLY`, because a publisher link is not an open full text.

## 8. OpenAlex implementation

* Search: `search`, `per-page`, `filter=from_publication_date`/`to_publication_date`/
  `type`/`raw_author_name.search`, `sort=publication_date:desc` for recent searches,
  plus `mailto` and `api_key` when configured (both sent as request parameters **and**
  in the polite `User-Agent`).
* Abstracts are reconstructed from `abstract_inverted_index` in position order.
* Open-access status, citation count, PDF/landing URLs come from `open_access` and the
  `best_oa_location`/`primary_location` pair; the raw OpenAlex/MAG/PMID/PMC ids are kept
  in `raw_provider_ids`.

## 9. Caching

`DiscoveryCache` is a TTL layer over RAG-006's `JsonCache` (one JSON file, inspectable).

* Key = `discovery:<provider>:<request-version>:<operation>:<normalized query|doi key|title>:…`
  including `limit` and the filters, so a provider, filter, limit or request-format
  change can never hit the wrong entry. `PROVIDER_REQUEST_VERSION` is the single source
  of truth for invalidating all entries at once (bumped to `rag010-v2` when the Crossref
  DOI route changed).
* TTL default 24 h (`PHD_RAG_DISCOVERY_CACHE_TTL`); `--refresh` bypasses a valid entry;
  `--no-cache` disables the cache entirely.
* `reset_counters()` per request means the audit reports *this* request's hits/misses.
* Provider **failures are deliberately not cached**: a temporary 429 must not be
  remembered as "nothing here". Consequence: every run re-attempts a currently
  rate-limited provider and waits for its retries (measured in §20).

## 10. Deduplication

Grouping is deterministic and identity-first: normalized DOI equality wins outright
(so `10.x/y`, `https://doi.org/10.x/y` and `doi:10.X/Y` are one paper); without a DOI, a
normalized-identical title or `title_similarity ≥ 0.95` is required. Two *different*
DOIs are never merged, whatever the titles say. Input is sorted before grouping, so
provider response order cannot change the grouping, the chosen field values or the
output order (tests §30/29). Merging keeps every provider in `providers` and every
supplied field in `field_provenance`; material disagreements become `CONFLICT` with both
values preserved rather than a silent winner.

## 11. Local-corpus matching

Every external record is compared with the corpus the index knows (465 documents):
DOI identity → exact normalized title → fuzzy title (≥ 0.97 near-identical, or ≥ 0.92
with year ±1 and first-author consistency), reusing RAG-006's normalization and
thresholds so the stages cannot drift apart. The **year guard also applies to an
exact-title match**, so an identical title nine years apart is reported as new instead
of being swallowed.

Evidence safety: a match against a non-evidence local role (`my_work`, `summaries`,
`supervision`, `rules`, `future_chapters`, `guidance`) is reported as `POSSIBLE_MATCH`
with `local_evidence_eligible = False`, even when the DOI matches exactly — an external
paper is never promoted to "already have it as scholarly evidence" because it resembles
the author's own work. Index insertion is sorted by `document_id`, so the same payloads
in a different order produce the same match.

## 12. Combined local/external discovery

`research_discover(query, local_k=8, external_k=20, …)` returns `CombinedDiscovery`
with `local` and `external` as separate objects; the CLI prints `LOCAL CORPUS` (RAG-009
chunks with role/evidence flags and a quoted preview) and `NEW EXTERNAL DISCOVERY`
(grouped by what it means) and never a fused list or a generated answer. A local
retrieval failure is recorded in `local.error` and does not remove the external half; an
external outage leaves `local` untouched and sets `external_available = False` with the
provider failures listed. Both halves share one Qdrant client (a local store allows only
one open client per process).

## 13. Full-text verification behaviour

Opt-in only (`--verify-fulltext` / `DiscoveryService.verify_fulltext`). It fetches an
open-access URL, verifies that the retrieved text really carries the record's DOI or
≥ 80 % of its title words, records the SHA-256, byte size, content type, source URL and
timestamp, and returns a **copy** of the record — metadata is never replaced and nothing
is stored in the corpus. A landing page (HTML) is explicitly not accepted as full text
(`not_fulltext`) unless `follow_landing_pages=True`, and an unconfirmed identity is
reported as `identity_unconfirmed` rather than assumed.

Run for real against live publishers (evidence in
[docs/rag010-local-match-validation.txt](docs/rag010-local-match-validation.txt)):

| DOI | Outcome |
|-----|---------|
| `10.1186/s41239-023-00434-1` | **Verified** — 1 926 765 bytes `application/pdf`, SHA-256 `316d1664a4ef7647…`, identity `doi:10.1186/s41239-023-00434-1` → `EXTERNAL_FULLTEXT_VERIFIED` |
| `10.1007/s10734-026-01706-1` | **Verified** — 962 294 bytes, SHA-256 `3757a0ce9490fb7f…`, identity by DOI |
| `10.3390/info16040296` | Fetch refused by the publisher (HTTP 403) → stays `EXTERNAL_METADATA_VERIFIED`, nothing claimed |
| `10.22364/htqe.2025.25` | Not fetched: the provider states no open full text (`not_open_access`) |

Two consequences of that run, both fixed: the publisher's `content-type: unspecified`
+ `/pdf` route is now recognised as a PDF link (MDPI-style records were previously
reported as `LANDING_PAGE_ONLY` with no URL at all, so an openly licensed PDF was never
offered), and the renderer now prints *why* a verification did not succeed rather than
leaving the reader to guess.

## 14. Security controls

* Only `http`/`https`; `file:`, `ftp:`, `data:`, `javascript:` are refused.
* `localhost`, `*.localhost`, `*.local`, `*.internal`, loopback, private, link-local,
  reserved, multicast and unspecified addresses are refused — both as literals and as
  resolved hostnames.
* Redirects are followed **manually, one hop at a time**, and every hop is
  re-validated, so a redirect cannot smuggle the fetch into a private network
  (`unsafe_redirect`, tested); redirect loops are bounded (`too_many_redirects`).
* A download size cap, a timeout and a content-type allowlist are enforced; nothing is
  executed, and only a hash plus a short identity excerpt are kept.
* No secret is hard-coded: the polite-pool contact comes from configuration, and when it
  is absent the tools say so instead of inventing an address.

## 15. CLI examples

```powershell
# external search, grouped output
python -m app.discovery.search "AI governance in higher education"

# recent work only, hiding what the corpus already has
python -m app.discovery.search "generative AI academic integrity" --year-min 2025 --new-only

# identity lookups and optional open-access verification
python -m app.discovery.search --doi 10.1016/j.compedu.2019.103778
python -m app.discovery.search --title "Artificial intelligence in higher education"
python -m app.discovery.search --doi 10.1016/j.compedu.2019.103778 --verify-fulltext

# machine-readable
python -m app.discovery.search "AI literacy higher education" --json

# local evidence + external discovery, two sections, no generated answer
python -m app.discovery.research "AI governance in higher education" --local-k 8 --year-min 2025
```

All flags from the brief are implemented: `--year-min`, `--year-max`, `--limit`,
`--provider` (repeatable), `--refresh`, `--show-abstract`, `--new-only`, `--local-only`,
`--json` (plus `--author`, `--sort`, `--no-cache`, `--verify-fulltext`, `--no-audit`).
Both entry points were run for real (§17).

## 16. Test result

```
python -m pytest -q
657 passed, 2 warnings in 7.44s
```

514 pre-existing tests (RAG-001…RAG-009) still pass unchanged; **143 new RAG-010 tests**
were added: providers 27, merge 17, local 15, cache 13, service 20, security 23,
combined 9, CLI 19. All 30 numbered checks in §30 are covered, and the suite is fully
offline (fake transports, fake fetcher, in-memory/disabled cache, synthetic corpus):
no test needs the Internet, a GPU or Tesseract.

## 17. Real-query results

Full detail: [docs/rag010-discovery.txt](docs/rag010-discovery.txt). Seven live queries
(5 required + 2 recent, `year_min=2025`), 20 records each:

| Query | Crossref | OpenAlex | failures | merged | already in corpus | new | conflicts |
|-------|----------|----------|----------|--------|-------------------|-----|-----------|
| artificial intelligence governance higher education | 20 | – | openalex 429 | 20 | 0 | 20 | 0 |
| generative AI academic integrity higher education | 20 | – | openalex 429 | 20 | 0 | 20 | 0 |
| virtual reality higher education student engagement | 20 | – | openalex 429 | 20 | 1 | 19 | 0 |
| AI literacy higher education | 20 | – | openalex 429 | 20 | 0 | 20 | 0 |
| design science research methodology education | 20 | – | openalex 429 | 20 | 0 | 20 | 0 |
| generative AI higher education (≥2025) | 20 | – | openalex 429 | 20 | 0 | 20 | 0 |
| AI governance universities (≥2025) | 20 | – | openalex 429 | 20 | 0 | 20 | 0 |

Manual inspection of the top 10 for three required queries:

* **AI governance higher education** — 8/10 clearly relevant (e.g. *Mapping artificial
  intelligence adoption, perceptions, and governance in Indian higher education*,
  *AI Governance in Higher Education: A Meta-Analytic Systematic Review*, *Higher
  Education in the Age of AI: Governance, Educational Values…*), 2 generic-but-related
  (*Artificial Intelligence in Higher Education* book chapters).
* **Generative AI academic integrity higher education** — 9/10 clearly relevant
  (systematic reviews, a Springer *Higher Education* article, an Information (MDPI)
  review), 1 partly off (*Information Integrity, Academic Integrity, and Generative AI*
  is about information integrity more broadly).
* **Virtual reality higher education student engagement** — 8/10 clearly relevant, 2
  shallow (a general *Virtual Reality Education* record and a mental-health treatment
  study that only matches on "student engagement").

Honest caveats from the same output: `duplicates_merged = 0` for every query because
OpenAlex was unavailable, so no cross-provider merge could occur live (cross-provider
merging is covered by the offline tests); and generic short titles (*Artificial
Intelligence in Higher Education*) rank high against a broad query, because ranking
rewards title similarity — see §22.

The two §32 checks were run for real:

* `python -m app.discovery.research "AI governance in higher education"` → 8 local
  evidence chunks (all `role=literature`, `evidence_eligible=True`, e.g. Chan (2024)
  *Generative AI in Higher Education; The ChatGPT Effect*) printed above 10 external
  candidates; separation check: **no external record was represented as local evidence**
  (`ok - none`), and no prose answer was produced.
* Known-paper and performance checks: see §18 and §20.

## 18. Known-local-paper matching result

Five corpus papers with DOI metadata (deterministically sampled across the corpus, no
filename-specific tuning), looked up by DOI and by title:

| # | Local paper (truncated) | DOI | DOI route | Title route |
|---|------------------------|-----|-----------|-------------|
| 1 | Exploring the potential of AI tools in educational measurement… | `10.29333/ejmste/13428` | ALREADY_IN_CORPUS | ALREADY_IN_CORPUS |
| 2 | A survey on surface reconstruction based on 3D Gaussian splatting | `10.7717/peerj-cs.3034` | ALREADY_IN_CORPUS | ALREADY_IN_CORPUS |
| 3 | Conversations between self and self as Sigmund Freud… | `10.1038/srep13899` | ALREADY_IN_CORPUS | ALREADY_IN_CORPUS |
| 4 | Providing Individual Student Feedback at Scale… | `10.1145/3491140.3528313` | ALREADY_IN_CORPUS | ALREADY_IN_CORPUS |
| 5 | An Educational Virtual Reality Game from an Embodied Cognitive Perspective | `10.54254/2753-7048/11/20230724` | ALREADY_IN_CORPUS | NO_RECORD |

**5/5 correctly recognized as `ALREADY_IN_CORPUS` by DOI** (acceptance criterion met),
each at `identity_confidence = 0.85` and `EXTERNAL_METADATA_VERIFIED`. The external
titles matched the local titles character-for-character, including the em dash in
record 3. The title route identified 4/5: for record 5 the local title is
*…from an Embodied Cognitive Perspective* while Crossref holds *…- Take the Game
"Poetry" as an Example*, so the `≥ 0.90` title filter kept it out — a deliberate
precision-over-recall choice, and the DOI route covers that paper. Local matches also
appear correctly for a paper the plain search found (the VR query → 1 already held).

## 19. Recent-literature usefulness assessment

Detail: [docs/rag010-recent-literature.txt](docs/rag010-recent-literature.txt).

For `generative AI higher education` (≥ 2025) and `AI governance universities` (≥ 2025),
20 records each were inspected. The searches are **genuinely useful for finding
literature missing from the corpus**: both returned 2025–2026 material with substantive
abstracts and no already-held items, e.g. *Where universities place generative AI
governance: a comparative document analysis of institutions in Korea, Japan, and China*
(2026), *Testing the applicability of a governance checklist for high-risk AI-based
learning outcome assessment in Italian universities under the EU AI Act Annex III*
(2025), *Mapping generative AI governance readiness in Kurdistan Region universities*
(2026), and *How Generative AI Fixes What Higher Education Broke* (2025, with a
detailed abstract). Relevance was judged from the abstract where one exists, not from
the title alone.

Two quality observations, both reported rather than fixed in this stage:

1. **Preprint version families** — `How Generative AI Fixes What Higher Education Broke`
   occupies 6 of the 20 slots: Crossref registers each OSF version (`osf.io/yh6rs_v1`
   … `_v7`) as its own DOI, and different DOIs are different works by design, so
   deduplication correctly does not merge them.
2. **Book-chapter series** — several *Artificial Intelligence in Higher Education* /
   *Navigating Generative AI in Higher Education* chapters appear with no abstract from
   Crossref, so they can only be judged by title.

Actionable conclusion: the recent-work route is useful now; the two observations are
candidate refinements (a display-level work cap, and OpenAlex abstracts) rather than
correctness defects.

## 20. Performance

Measured on the real corpus (465 documents, 409 with titles) and live APIs:

| Stage | Measurement |
|-------|-------------|
| Crossref search (cold, network) | 5.4 ms provider time, 20 records |
| Crossref search (cache hit) | 0.8 ms provider time |
| Rediscovery with only a cached provider | 1.029 s total (dominated by the local comparison) |
| OpenAlex search while rate-limited | 5.56 s (retries/backoff), HTTP 429 |
| Both providers, cold, no cache | 9.65 s total (OpenAlex's backoff dominates) |
| Merge + rank (20 records) | 4.3 ms |
| Local comparison (20 records) | 1.03 s (51.3 ms/record; single record 26 ms) |
| First request in a process | +≈3.0 s one-off local-index build (465 docs read from Qdrant) |

So: network latency dominates when a provider is healthy, the local comparison is the
dominant *CPU* cost (~1 s per 20-record request, because each unmatched title is
compared against 409 corpus titles with `title_similarity`), and a rate-limited provider
dominates when it is not. Per the brief this was measured before optimizing; the two
obvious, behaviour-preserving optimizations for a later stage are (a) an upper-bound
prefilter before `SequenceMatcher` in `LocalCorpusIndex.best_title_candidates` and
(b) caching a provider failure for its `Retry-After` so a known outage does not cost
~5 s per run.

## 21. Provider / API limitations (observed, not assumed)

| Provider | Status in this environment | Evidence |
|----------|---------------------------|----------|
| Crossref | **Working** — search, DOI lookup and title lookup all returned real 2025–2026 records | `docs/rag010-discovery.txt` |
| OpenAlex | **HTTP 429 on every request**: `"Insufficient budget. This request has no API key, so it counts against the free daily budget shared by everyone on your network"`, `Retry-After` ≈ 16 854–18 406 s, i.e. it resets at midnight UTC | recorded verbatim in the audit |
| Semantic Scholar | Implemented as an optional adapter; HTTP 429 without a key | adapter test only |

The 429 is the provider's request budget for this IP, reported as exactly that — never
as a cost to the user. A free `PHD_RAG_OPENALEX_API_KEY` (and/or a polite `mailto`)
removes it; the code path is implemented and tested, and no key is invented. No
polite-pool contact identity was configured in this environment, so the tools report
`polite_pool_identity_configured = false` and operate without one, as the brief requires.
Consequence for §17–§19: all live results are Crossref-only, and no live cross-provider
merge could be demonstrated (the offline tests cover it).

## 22. Known limitations

1. **OpenAlex unavailable in this environment** (§21) — dedup, conflict detection and
   open-access/full-text fields are exercised offline, not live.
2. **Ranking favours title-similarity**: generic titles such as *Artificial Intelligence
   in Higher Education* score highly against broad queries. A "all query terms present"
   gate would improve precision but is a ranking change, so it was not made here.
3. **Preprint version families occupy several slots** (§19.1) — a consequence of
   DOI-strict identity, working as designed but noisy for a reader.
4. **Local comparison cost** (§20): ~51 ms per unmatched record against 465 documents;
   measured, deliberately not optimized in this stage.
5. **Rate-limited providers cost retries** — failures are not cached (by design), so a
   known outage costs ~5 s per run until the budget resets.
6. **A year filter drops records with a missing year** (provider metadata gaps are
   common for book chapters); this is conservative but does discard otherwise usable
   records, and it is deliberate rather than accidental.
7. **`find_related` uses only the local year** (`base_year + 1`) when no explicit floor
   is given, so a badly extracted local year narrows the search; no model rewrites the
   query, as the brief requires.
8. **No import path** — by design there is no command that moves an external record into
   the corpus; that is explicitly a later, separate action.

## 23. Recommendation for the next RAG stage

1. Keep the separation exactly as it is: local evidence and external discovery are
   different knowledge domains, and no size of external result set should ever be ranked
   into `--mode evidence`.
2. Before widening discovery, configure the provider identity
   (`PHD_RAG_CROSSREF_MAILTO`, `PHD_RAG_OPENALEX_EMAIL`, and a free OpenAlex key) so
   OpenAlex agrees to answer: that single change unlocks cross-provider dedup, conflict
   detection, abstracts and open-access fields, all of which are already implemented.
3. If an import/acceptance flow is added later, make it explicit and per-record, and
   keep the corpus fingerprint and role resolution (RAG-007) as the gate — never a
   bulk sync.
4. If answer generation is the next stage, prompt from `--mode evidence` results only,
   and treat external records as clearly labelled pointers ("not in your corpus"),
   never as citable literature.
5. Cheap, measurable follow-ups when wanted: cache provider failures for their
   `Retry-After`; add the token-presence gate to external ranking; add the
   upper-bound prefilter to local title matching (§20).

---

## Validation defects found and fixed during this stage

Real validation did its job; the following were found by running the feature, not by
reasoning about it:

1. **`merge.py` crashed on any merge** (`NameError: merged_doi`) — caught by the first
   live search; the DOI normalization step is now explicit.
2. **Crossref DOI lookups returned HTTP 400 for every DOI** — Crossref rejects `select`
   on `/works/{doi}` (`"This route does not support select"`), which the provider's own
   error text revealed because failures preserve the provider's words. Fixed, with
   regression tests.
3. **The combined service opened two Qdrant clients in one process** — a local store
   allows one, so `python -m app.discovery.research` failed outright. Discovery now
   reuses the retrieval service's client read-only.
4. **`--local-only` silently kept the default providers** — an empty provider tuple was
   falsy and fell back to the defaults; now explicit.
5. **The exact-title local match ignored the year guard**, so an identical title from a
   different year could be swallowed as "already have it"; the ±1 guard now applies.
6. **Local matching depended on payload order** when two documents shared a normalized
   title; the index now resolves ties by `document_id`.
7. **`candidate.pages` did not exist** on the RAG-009 `Candidate`, so the combined
   renderer would have raised; it now derives the span from `page_start/page_end`.
8. **`-m app.discovery.research` warned about a double import**; the package export is
   lazy now.
9. **Openly licensed PDFs were missed** when a publisher declared `content-type:
   unspecified` (MDPI and others) — the PDF-route check now also accepts a `/pdf` path
   segment, and the renderer explains an unverified fetch instead of hiding the reason.

**RAG-010 COMPLETE** — RAG-011 has not begun.
