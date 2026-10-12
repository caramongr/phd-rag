"use strict";

/*
 * The RAG-014 frontend.
 *
 * Three rules keep this file honest:
 *
 * 1. It renders, it does not decide. Every mode, every section, every verdict and every
 *    label comes from the API (`/api/modes`, `/api/mode`, `/api/status`), which comes from
 *    the RAG-013 mode service. There is no mode logic here and no second result model.
 * 2. All text from a document, a question, a model or a provider is written with
 *    `textContent`. Nothing in this app interpolates data into HTML, so a chunk containing
 *    `<script>` is displayed as text and never becomes markup.
 * 3. What the brief calls "output" is built from the result's own `sections`, `findings`,
 *    `references`, `warnings` and `sources` arrays — the same arrays the CLI prints.
 */

const state = {
  modes: [],
  advanced: {},
  mode: "evidence",
  result: null,
};

/* ------------------------------------------------------------------ helpers */

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  Object.entries(props || {}).forEach(([key, value]) => {
    if (value === null || value === undefined || value === false) return;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "onclick") node.addEventListener("click", value);
    else if (key === "hidden") node.hidden = Boolean(value);
    else if (key === "open") node.open = Boolean(value);
    else node.setAttribute(key, value);
  });
  children.flat().forEach((child) => {
    if (child === null || child === undefined || child === false) return;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  });
  return node;
}

function lines(...items) {
  return items.filter((item) => item !== null && item !== undefined && item !== "");
}

function join(separator, items) {
  return lines(...items).join(separator);
}

function text(value, fallback = "") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function truncate(value, limit) {
  const clean = text(value);
  if (clean.length <= limit) return clean;
  return clean.slice(0, limit).replace(/\s+\S*$/, "") + " …";
}

function badge(label, kind, strong = false) {
  const classes = ["badge"].concat(kind ? [kind] : []).concat(strong ? ["strong"] : []);
  return el("span", { class: classes.join(" "), text: label });
}

function kv(pairs) {
  const list = el("dl", { class: "kv" });
  pairs.forEach(([key, value]) => {
    if (value === null || value === undefined || value === "") return;
    list.append(el("dt", { text: key }), el("dd", { text: String(value) }));
  });
  return list;
}

function details(summary, ...children) {
  return el("details", { class: "card-details" }, el("summary", { text: summary }), ...children);
}

function sectionTitle(name, note) {
  return el(
    "h3",
    { class: "section-title" },
    el("span", { text: name }),
    note ? el("span", { class: "card-line", text: "  " + note }) : null
  );
}

function paragraphs(value) {
  const clean = text(value);
  if (!clean.trim()) return el("p", { class: "card-line", text: "(nothing to show)" });
  return el(
    "div",
    {},
    clean.split(/\n\s*\n/).map((block) => el("p", { class: "paragraph", text: block.trim() }))
  );
}

async function requestJSON(url, options) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    const error = (payload && payload.error) || {};
    const failure = new Error(
      error.title || `The request failed (HTTP ${response.status})`
    );
    failure.detail = error.detail || "The server did not return a usable answer.";
    failure.debug = error.fields || error.traceback || null;
    failure.status = response.status;
    throw failure;
  }
  return payload;
}

/* ------------------------------------------------------------------- status */

function renderStatus(payload) {
  const target = document.getElementById("status");
  target.replaceChildren();
  const ollama = payload.ollama || {};
  const index = payload.index || {};

  const ollamaClass = ollama.available ? "ok" : "bad";
  const ollamaText = ollama.available
    ? `Ollama ${text(ollama.version, "reachable")} · ${text(ollama.model)}`
    : "Ollama unavailable";

  const indexClass = index.available ? "ok" : index.busy ? "" : "bad";
  const points =
    typeof index.points === "number" ? `${index.points.toLocaleString()} points` : "no index";
  const indexText = index.available
    ? `${index.collection} · ${points}`
    : index.busy
      ? "index in use — a run is in progress"
      : "index unavailable";

  target.append(
    el("span", { class: "status-line" }, el("span", { class: ollamaClass, text: ollamaText })),
    el("span", { text: " · " }),
    el("span", { class: "status-line" }, el("span", { class: indexClass, text: indexText }))
  );
  if (ollama.model_present === false) {
    target.append(
      el("span", {
        class: "status-line",
        text: ` · model ${text(ollama.model)} is not installed`,
      })
    );
  }
  const problems = lines(ollama.error, index.error);
  if (problems.length) {
    target.append(
      el("span", { class: "status-line" }, details("details", kv([["ollama", ollama.error], ["index", index.error]])))
    );
  }
}

async function loadStatus() {
  try {
    renderStatus(await requestJSON("/api/status"));
  } catch (error) {
    const target = document.getElementById("status");
    target.replaceChildren(el("span", { class: "bad", text: "status unavailable" }));
  }
}

/* -------------------------------------------------------------------- modes */

function optionKeyFor(element) {
  return element.id.replace(/^opt-/, "").replace(/-/g, "_");
}

function optionElements() {
  return Array.from(document.querySelectorAll(".advanced-grid .adv")).map((label) => ({
    label,
    input: label.querySelector("input, select"),
  }));
}

function applyModeOptions(mode) {
  const entry = state.modes.find((item) => item.name === mode);
  const allowed = new Set((entry && entry.options) || []);
  optionElements().forEach(({ label, input }) => {
    if (!input) return;
    const key = optionKeyFor(input);
    const applies = allowed.has(key) || key === "show_audit";
    label.hidden = !applies;
  });
}

function renderModeDescriptions() {
  const select = document.getElementById("mode");
  const description = document.getElementById("mode-description");
  select.replaceChildren(
    ...state.modes.map((mode) => el("option", { value: mode.name, text: mode.label }))
  );
  select.value = state.mode;
  const entry = state.modes.find((item) => item.name === state.mode);
  description.textContent = entry ? entry.description : "";
  applyModeOptions(state.mode);
}

function renderLengthChoices() {
  const select = document.getElementById("opt-length");
  const spec = state.advanced.length || {};
  const choices = spec.choices || ["concise", "normal", "detailed"];
  select.replaceChildren(
    ...choices.map((choice) =>
      el("option", {
        value: choice,
        text: choice.charAt(0).toUpperCase() + choice.slice(1),
      })
    )
  );
  select.value = spec.default || "normal";
}

async function loadModes() {
  const payload = await requestJSON("/api/modes");
  state.modes = payload.modes || [];
  state.advanced = payload.advanced_options || {};
  renderLengthChoices();
  renderModeDescriptions();
}

/* ------------------------------------------------------------------ options */

function collectOptions() {
  const options = {};
  const length = document.getElementById("opt-length").value;
  if (length) options.length = length;
  const candidate = document.getElementById("opt-candidate-k").value;
  if (candidate) options.candidate_k = Number(candidate);
  const finalK = document.getElementById("opt-final-k").value;
  if (finalK) options.final_k = Number(finalK);
  options.rerank = document.getElementById("opt-rerank").checked;
  const yearMin = document.getElementById("opt-year-min").value;
  if (yearMin) options.year_min = Number(yearMin);
  const yearMax = document.getElementById("opt-year-max").value;
  if (yearMax) options.year_max = Number(yearMax);
  return options;
}

/* ------------------------------------------------------------------ notices */

function renderNotice(notice) {
  const target = document.getElementById("notice");
  if (!notice) {
    target.hidden = true;
    target.className = "notice";
    target.replaceChildren();
    return;
  }
  const level = notice.level || "info";
  const children = [
    el("h3", { text: text(notice.title, "Note") }),
    el("p", { text: text(notice.detail) }),
  ];
  const debug = notice.debug && Object.keys(notice.debug).length ? notice.debug : null;
  if (debug) {
    children.push(details("Technical details", el("pre", { class: "audit", text: JSON.stringify(debug, null, 2) })));
  }
  target.className = `notice ${level}`;
  target.hidden = false;
  target.replaceChildren(...children);
}

function renderError(error) {
  renderNotice({
    level: "error",
    title: error.message || "Something went wrong",
    detail: error.detail || "The run failed before it produced a result.",
    debug: error.debug ? { status: error.status, fields: error.debug } : { status: error.status },
  });
}

/* ------------------------------------------------------------------ sources */

function roleLabel(source) {
  const role = text(source.role, "");
  const corpus = text(source.corpus_role, "");
  if (corpus && corpus !== "literature") return badge(`[${corpus}]`, "non-evidence");
  if (role === "subject") return badge("[your own work]", "non-evidence");
  if (role && role !== "evidence") return badge(`[${role.replace(/_/g, " ")}]`, "non-evidence");
  return badge("[evidence]", "evidence");
}

function sourceEvidenceBadge(source) {
  const label = source.evidence_eligible ? "evidence" : "not evidence";
  return badge(`${label} (role=${text(source.role, "-")})`, source.evidence_eligible ? "evidence" : "non-evidence");
}

function sourceCard(source, index) {
  const heading = join(" · ", [text(source.first_author, "unknown author"), text(source.year, "n.d.")]);  const provenance = [
    ["chunk id", source.chunk_id],
    ["document id", source.document_id],
    ["file", source.filename],
    ["corpus role", source.corpus_role],
    ["evidence eligible", source.evidence_eligible],
    ["role in this mode", source.role],
    ["retrieval mode", source.retrieval_mode],
    ["content type", source.content_type],
    ["dense score", source.dense_score],
    ["rerank score", source.rerank_score],
    ["pages", source.pages],
    ["section", source.section_title],
    ["doi", source.doi],
  ];
  return el(
    "article",
    { class: "card" },
    el("div", { class: "card-title", text: text(source.title, "(untitled)") }),
    el("div", { class: "card-line", text: join(" · ", [heading, source.pages ? `p.${source.pages}` : null]) }),
    el(
      "div",
      { class: "badges" },
      roleLabel(source),
      sourceEvidenceBadge(source),
      source.ocr_involved ? badge("OCR", "non-evidence") : null,
      source.content_type && source.content_type !== "body" ? badge(text(source.content_type)) : null
    ),
    el("div", { class: "card-line", text: join(" · ", [
      source.section_title ? `section: ${source.section_title}` : null,
      source.doi ? `doi:${source.doi}` : null,
    ]) }),
    el("p", { class: "excerpt", text: truncate(source.excerpt, 600) }),
    details("Retrieved text, scores and provenance", kv(provenance))
  );
}

function renderLocalCards(section) {
  const cards = (section.items || []).map((source, index) => sourceCard(source, index));
  return cards.length ? cards : [el("p", { class: "card-line", text: "(none)" })];
}

/* ----------------------------------------------------------------- external */

function externalCard(record) {
  const inCorpus = Boolean(record.already_in_corpus);
  const identity = inCorpus ? "ALREADY IN CORPUS" : "NOT IN LOCAL CORPUS";
  const openAccess = record.open_access === true
    ? "open access"
    : record.open_access === false
      ? "not open access"
      : "open access unknown";
  const fulltext = record.fulltext_verified
    ? "full text verified"
    : `full text: ${text(record.fulltext_status, "unknown")}`;
  return el(
    "article",
    { class: "card" },
    el("div", { class: "card-title", text: text(record.title, "(untitled record)") }),
    el(
      "div",
      { class: "card-line" },
      join(" · ", [
        (record.authors || []).slice(0, 4).join(", ") || "unknown authors",
        text(record.year, "n.d."),
        record.venue,
        record.publisher,
      ])
    ),
    el(
      "div",
      { class: "badges" },
      badge(identity, inCorpus ? "evidence" : "non-evidence", true),
      badge("metadata only — never evidence", "non-evidence"),
      badge(`status: ${text(record.discovery_status, "unknown")}`),
      badge((record.providers || []).join(" + ") || "no provider"),
      badge(openAccess),
      badge(fulltext, record.fulltext_verified ? "evidence" : null)
    ),
    el(
      "div",
      { class: "card-line" },
      join(" · ", [
        record.doi ? `doi:${record.doi}` : null,
        record.url ? text(record.url) : null,
      ])
    ),
    record.abstract ? el("p", { class: "excerpt", text: truncate(record.abstract, 480) }) : null,
    details(
      "Provenance and verification",
      kv([
        ["local match", record.local_match_kind],
        ["local document is evidence", record.local_evidence_eligible],
        ["identity confidence", record.identity_confidence],
        ["relevance score", record.relevance_score],
        ["usable as evidence", record.usable_as_evidence],
        ["fulltext verified", record.fulltext_verified],
      ])
    )
  );
}

/* ---------------------------------------------------------------- findings */

const FINDING_GROUPS = [
  ["terminology", "Terminology"],
  ["duplication", "Duplication"],
  ["unsupported_claim", "Unsupported claims"],
  ["numeric_discrepancy", "Numeric discrepancies"],
  ["style", "Style"],
];

function findingCard(finding) {
  const children = [
    el("div", { class: "card-title", text: text(finding.summary) }),
    finding.detail ? el("p", { class: "card-line", text: text(finding.detail) }) : null,
    finding.your_text ? el("p", { class: "excerpt", text: `your text: ${text(finding.your_text)}` }) : null,
    finding.other_text ? el("p", { class: "excerpt", text: `literature: ${text(finding.other_text)}` }) : null,
    el("div", { class: "card-line", text: `method: ${text(finding.method, "unknown")}` }),
    details(
      "Where this finding rests",
      kv([
        ["your chunk", finding.your_chunk_id],
        ["literature chunk", finding.other_chunk_id],
        ["evidence", (finding.evidence || []).join(", ")],
      ])
    ),
  ];
  return el("article", { class: "card" }, ...children);
}

function renderFindings(result) {
  const findings = result.findings || [];
  if (!findings.length) {
    return el("p", {
      class: "card-line",
      text: "No differences were found in what was retrieved. Repeat with a broader query before drawing a conclusion.",
    });
  }
  const groups = new Map();
  findings.forEach((finding) => {
    const key = finding.kind;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(finding);
  });
  const blocks = [];
  FINDING_GROUPS.forEach(([key, title]) => {
    const items = groups.get(key);
    if (!items || !items.length) return;
    blocks.push(sectionTitle(`${title} (${items.length})`));
    if (key === "unsupported_claim") {
      blocks.push(
        el("p", {
          class: "card-line",
          text:
            "“Unsupported” here means no match was found in the retrieved literature — it is not "
            + "a statement that the claim is false. Widen the retrieval before rewriting anything.",
        })
      );
    }
    items.forEach((finding) => blocks.push(findingCard(finding)));
  });
  return el("div", {}, ...blocks);
}

/* ------------------------------------------------------------------- answer */

function renderSourcesPanel(result) {
  const panel = document.getElementById("sources-panel");
  const target = document.getElementById("sources");
  const localSections = (result.sections || []).filter((section) => (section.local_indices || []).length);
  const children = [];
  localSections.forEach((section) => {
    children.push(sectionTitle(section.name, section.note));
    children.push(renderLocalCards(section));
  });
  if (!children.length) {
    children.push(el("p", { class: "card-line", text: "(no local sources)" }));
  }
  target.replaceChildren(...children.flat());
  panel.hidden = false;
}

function renderValidationPanel(result) {
  const target = document.getElementById("validation");
  const summary = result.validation_summary;
  if (!summary) {
    target.replaceChildren(
      el("p", { class: "card-line", text: "This mode does not validate citations." })
    );
    return;
  }
  const children = [
    kv([
      ["claims", summary.claims_total],
      ["supported", summary.supported],
      ["partially supported", summary.partially_supported],
      ["unsupported (withheld)", summary.unsupported],
      ["uncertain", summary.uncertain],
      ["markers replaced", summary.markers_replaced],
      ["markers removed", summary.markers_removed],
      ["quotations checked", summary.quotations_checked],
      ["figures checked", summary.numeric_mentions_checked],
      ["sources cited", summary.sources_cited],
      ["references", summary.references_total],
      ["references with metadata warnings", summary.references_with_metadata_warning],
    ]),
  ];
  const claims = result.claim_validations || [];
  if (claims.length) {
    children.push(sectionTitle(`Claim by claim (${claims.length})`));
    claims.forEach((claim) => {
      const verdictClass =
        claim.verdict === "SUPPORTED"
          ? "evidence"
          : claim.verdict === "UNSUPPORTED"
            ? "error"
            : "non-evidence";
      children.push(
        el(
          "article",
          { class: "card" },
          el(
            "div",
            { class: "badges" },
            badge(claim.verdict, verdictClass),
            badge(`method: ${text(claim.validator_method, "unknown")}`),
            badge(`confidence: ${text(claim.confidence)}`),
            claim.harvard_citation ? badge(`citation: ${claim.harvard_citation}`) : null,
            claim.elevated_validation_risk ? badge("elevated risk (OCR/table)", "non-evidence") : null
          ),
          el("p", { class: "excerpt", text: text(claim.sentence) }),
          claim.reason ? el("p", { class: "card-line", text: text(claim.reason) }) : null,
          (claim.warnings || []).length
            ? el("p", { class: "card-line", text: `warnings: ${claim.warnings.join("; ")}` })
            : null,
          details(
            "Sources and measurements",
            kv(
              (claim.per_source || []).flatMap((support) => [
                [`${support.source_id} verdict`, support.verdict],
                [`${support.source_id} reason`, support.reason],
                [`${support.source_id} single-sentence coverage`, support.best_sentence_coverage],
                [`${support.source_id} union coverage`, support.union_coverage],
              ])
            )
          )
        )
      );
    });
  }
  target.replaceChildren(...children);
}

function renderWarnings(result) {
  const target = document.getElementById("warnings");
  const warnings = result.warnings || [];
  if (!warnings.length) {
    target.replaceChildren();
    return;
  }
  target.replaceChildren(
    sectionTitle(`Warnings (${warnings.length})`),
    el("ul", { class: "warning-list" }, warnings.map((warning) => el("li", { text: warning })))
  );
}

function renderReferences(result) {
  const target = document.getElementById("references");
  const references = result.references || [];
  if (!references.length) {
    target.replaceChildren();
    return;
  }
  target.replaceChildren(
    sectionTitle(`References (${references.length})`),
    el("ol", { class: "reference-list" }, references.map((reference) => el("li", { text: reference })))
  );
}

function renderBody(result) {
  const target = document.getElementById("body");
  const mode = result.mode;
  const blocks = [];

  if (mode === "evidence" || mode === "writing") {
    const answer = text(result.validated_answer) || text(result.answer);
    blocks.push(el("pre", { class: "answer", text: answer || "(no answer)" }));
    if (result.validated_answer && result.answer && result.validated_answer !== result.answer) {
      blocks.push(details("Original generated answer (before validation)", el("pre", { class: "answer", text: result.answer })));
    }
  } else if (mode === "consistency") {
    blocks.push(
      el("p", {
        class: "card-line",
        text:
          "Measured against the retrieved literature and your rules. Nothing here is a model's "
          + "opinion, and nothing is rewritten for you.",
      })
    );
    blocks.push(renderFindings(result));
    if (result.answer) {
      blocks.push(details("Full comparison report", paragraphs(result.answer)));
    }
  } else if (mode === "supervisor") {
    blocks.push(
      el("p", {
        class: "card-line",
        text:
          "Supervision, guidance and rules material. This is your own scaffolding: it is not "
          + "scholarly evidence, it is never cited, and it cannot support a claim about the literature.",
      })
    );
    (result.sections || [])
      .filter((section) => (section.local_indices || []).length)
      .forEach((section) => {
        blocks.push(sectionTitle(section.name, section.note));
        blocks.push(renderLocalCards(section));
      });
  } else {
    // RESEARCH: every section, in the order the mode produced them, local and external kept
    // apart — and an empty section is shown as empty, because "nothing was verified" is a
    // result the reader needs to see.
    (result.sections || []).forEach((section) => {
      blocks.push(sectionTitle(section.name, section.note));
      const local = (section.local_indices || []).length;
      const external = (section.external_indices || []).length;
      if (!local && !external) {
        blocks.push(el("p", { class: "card-line", text: "(none)" }));
        return;
      }
      if (local) blocks.push(renderLocalCards(section));
      if (external) {
        blocks.push(...(section.items || []).map((record) => externalCard(record)));
      }
    });
    if (!blocks.length) {
      blocks.push(el("p", { class: "card-line", text: "(no local or external results)" }));
    }
  }
  target.replaceChildren(...blocks.filter(Boolean).flat());
}

/* ------------------------------------------------------------------- render */

function expandSections(result) {
  const localSources = result.local_sources || [];
  const externalSources = result.external_sources || [];
  const sections = (result.sections || []).map((section) => {
    const items = [];
    (section.local_indices || []).forEach((index) => items.push(localSources[index]));
    (section.external_indices || []).forEach((index) => items.push(externalSources[index]));
    return { ...section, items: items.filter(Boolean) };
  });
  return { ...result, sections: sections };
}

function renderResult(payload) {
  state.result = expandSections(payload);
  const result = state.result;
  const copyRow = document.getElementById("copy-row");

  const titles = {
    evidence: "Validated answer",
    writing: "Validated draft",
    consistency: "Consistency findings",
    supervisor: "Supervision and guidance (not evidence)",
    research: "Local and external results",
  };
  document.getElementById("result-title").textContent = titles[result.mode] || "Result";

  const summary = result.validation_summary;
  const metaBits = [
    `mode: ${result.mode}`,
    `status: ${result.status}`,
    `${result.seconds}s`,
    summary ? `${summary.claims_total} claims (${summary.supported} supported)` : null,
    `${(result.local_sources || []).length} local sources`,
    (result.external_sources || []).length ? `${result.external_sources.length} external records` : null,
  ];
  document.getElementById("result-meta").textContent = join(" · ", metaBits);

  copyRow.hidden = false;
  const hasReferences = ((result.references || []).length > 0);
  copyRow.querySelectorAll("button").forEach((button) => {
    button.disabled = !hasReferences && button.dataset.copy !== "answer";
  });
  renderBody(result);
  renderWarnings(result);
  renderReferences(result);

  const sourcesPanel = document.getElementById("sources-panel");
  const bodyAlreadyHasSources = result.mode === "supervisor" || result.mode === "research";
  sourcesPanel.hidden = bodyAlreadyHasSources;
  if (bodyAlreadyHasSources) {
    document.getElementById("sources").replaceChildren();
  } else {
    renderSourcesPanel(result);
  }

  const externalPanel = document.getElementById("external-panel");
  const external = (result.external_sources || []);
  const researchAlreadyShows = result.mode === "research";
  externalPanel.hidden = researchAlreadyShows || !external.length;
  if (externalPanel.hidden) {
    document.getElementById("external").replaceChildren();
  } else {
    document
      .getElementById("external")
      .replaceChildren(...external.map((record) => externalCard(record)));
  }

  renderValidationPanel(result);
  document.getElementById("audit").textContent = JSON.stringify(payload, null, 2);

  document.getElementById("result").hidden = false;
  if (document.getElementById("opt-show-audit").checked) {
    document.getElementById("audit-panel").open = true;
  }
}

/* ---------------------------------------------------------------------- run */

function answerText(result) {
  return text(result && (result.validated_answer || result.answer));
}

function referencesText(result) {
  return ((result && result.references) || [])
    .map((reference, index) => `${index + 1}. ${reference}`)
    .join("\n\n");
}

async function copy(kind) {
  const result = state.result;
  const status = document.getElementById("copy-status");
  if (!result) return;
  let value = "";
  if (kind === "answer") value = answerText(result);
  else if (kind === "references") value = referencesText(result);
  else {
    const answer = answerText(result);
    const references = referencesText(result);
    value = references ? `${answer}\n\nReferences\n\n${references}` : answer;
  }
  try {
    await navigator.clipboard.writeText(value);
    status.textContent = "copied";
  } catch (error) {
    status.textContent = "could not copy — select the text instead";
  }
  window.setTimeout(() => {
    status.textContent = "";
  }, 2500);
}

function setBusy(busy, mode) {
  document.getElementById("run").disabled = busy;
  document.getElementById("loading").hidden = !busy;
  document.getElementById("loading-text").textContent = busy
    ? `${text(mode)} mode is working — retrieval, then the model, then validation…`
    : "";
}

async function run(event) {
  event.preventDefault();
  const mode = document.getElementById("mode").value;
  const query = document.getElementById("query").value.trim();
  if (!query) {
    renderNotice({
      level: "warning",
      title: "Type a question or paste some text first",
      detail: "The input box can hold a question, a paragraph, or a whole section of your thesis.",
    });
    return;
  }
  renderNotice(null);
  document.getElementById("result").hidden = true;
  setBusy(true, mode);
  try {
    const payload = await requestJSON("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode, query: query, options: collectOptions() }),
    });
    renderResult(payload);
    renderNotice(payload.notice || null);
  } catch (error) {
    renderError(error);
  } finally {
    setBusy(false, mode);
  }
}

function clearAll() {
  document.getElementById("query").value = "";
  document.getElementById("result").hidden = true;
  document.getElementById("copy-status").textContent = "";
  state.result = null;
  renderNotice(null);
  document.getElementById("query").focus();
}

/* --------------------------------------------------------------------- boot */

async function boot() {
  document.getElementById("run-form").addEventListener("submit", run);
  document.getElementById("clear").addEventListener("click", clearAll);
  document.getElementById("mode").addEventListener("change", (event) => {
    state.mode = event.target.value;
    renderModeDescriptions();
  });
  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", () => copy(button.dataset.copy));
  });
  const queryBox = document.getElementById("query");
  queryBox.addEventListener("keydown", (event) => {
    // Enter is a newline inside a textarea, so running needs a modifier: Ctrl/Cmd+Enter.
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      document.getElementById("run-form").requestSubmit();
    } else if (event.key === "Escape") {
      clearAll();
    }
  });
  renderLengthChoices();
  await loadStatus();
  try {
    await loadModes();
  } catch (error) {
    renderError(error);
  }
}

document.addEventListener("DOMContentLoaded", boot);
