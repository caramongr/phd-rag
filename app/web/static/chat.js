"use strict";

/*
 * The simple chat frontend (RAG-015 §11, §15).
 *
 * The same three rules the RAG-014 frontend keeps, and for the same reason:
 *
 * 1. It renders, it does not decide. Chats, messages, attachments and the context summary all
 *    come from `/api/chats` and `/api/chat/info`; there is no conversation state here that the
 *    server does not have.
 * 2. Every piece of text — a message, a filename, a provider error — is written with
 *    `textContent`. Nothing interpolates data into HTML, so a document cannot become markup.
 * 3. The academic interface is not reimplemented: it is the RAG-014 page, shown by the
 *    Advanced button.
 */

const chatState = {
  chats: [],
  chat: null,
  info: null,
  busy: false,
  lastReport: null,
};

/* ------------------------------------------------------------------ helpers */

function chatEl(tag, props, ...children) {
  const node = document.createElement(tag);
  Object.entries(props || {}).forEach(([key, value]) => {
    if (value === null || value === undefined || value === false) return;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "onclick") node.addEventListener("click", value);
    else if (key === "hidden") node.hidden = Boolean(value);
    else if (key === "value") node.value = value;
    else if (key === "checked") node.checked = Boolean(value);
    else if (key === "disabled") node.disabled = Boolean(value);
    else node.setAttribute(key, value);
  });
  children.flat().forEach((child) => {
    if (child === null || child === undefined || child === false) return;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  });
  return node;
}

function chatText(value, fallback = "") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function node(id) {
  return document.getElementById(id);
}

async function chatJSON(url, options) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    const error = (payload && payload.error) || {};
    const failure = new Error(error.title || `The request failed (HTTP ${response.status})`);
    failure.detail = error.detail || "The server did not return a usable answer.";
    failure.kind = error.kind || "";
    failure.status = response.status;
    throw failure;
  }
  return payload;
}

function bytes(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function bytesToBase64(buffer) {
  const view = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < view.length; index += chunk) {
    binary += String.fromCharCode.apply(null, view.subarray(index, index + chunk));
  }
  return btoa(binary);
}

function notice(level, title, detail, action) {
  const box = node("chat-notice");
  box.dataset.busy = "false";
  if (!title) {
    box.hidden = true;
    box.replaceChildren();
    return;
  }
  box.hidden = false;
  box.className = `notice ${level || "info"}`;
  box.replaceChildren(
    ...[
      chatEl("strong", { text: title, class: "notice-title" }),
      detail ? chatEl("p", { text: detail, class: "notice-detail" }) : null,
      action || null,
    ].filter(Boolean)
  );
}

/* The "thinking" line is the one notice that disappears by itself; anything else stays until
   the next action replaces it, so a message about an attachment — or the Retry action for a
   failed turn — is not wiped by the spinner. */
function busyNotice() {
  notice("info", "The model is thinking…", "This can take a while on a long context.");
  node("chat-notice").dataset.busy = "true";
}

function clearBusyNotice() {
  if (node("chat-notice").dataset.busy === "true") notice(null, "");
}

/* One failed turn stays one failed turn: this asks the server to answer the *existing* last
   user message again, so nothing the user typed is ever sent twice (FIX1 §6). */
function retryButton(label) {
  return chatEl("button", {
    type: "button",
    class: "primary retry-button",
    text: label || "Retry this turn",
    onclick: () => {
      retryLast();
    },
  });
}

function showFailure(failure, retriable) {
  const detail = chatText(failure.message, "Ollama did not return an answer.");
  const retried = failure.retries
    ? ` It was tried a second time (the runner reported: ${chatText(failure.retry_reason)}).`
    : "";
  notice(
    "error",
    "The model could not answer",
    `Ollama said: ${detail}.${retried} Your message is saved; retrying answers that same message — it is never sent twice.`,
    retriable === false ? null : retryButton()
  );
}

function showError(error) {
  const detail = [error.detail, error.kind ? `(${error.kind})` : ""]
    .filter((part) => part)
    .join(" ");
  notice("error", error.message || "The request failed", detail);
}

function setBusy(busy) {
  chatState.busy = busy;
  node("send").disabled = busy;
  node("attach-input").disabled = busy;
  const label = node("chat-hint");
  label.dataset.busy = busy ? "true" : "false";
  if (busy) {
    busyNotice();
  } else {
    clearBusyNotice();
  }
}

/* ------------------------------------------------------------------ rendering */

function renderChatList() {
  const list = node("chat-list");
  list.replaceChildren();
  if (!chatState.chats.length) {
    list.append(chatEl("p", { class: "hint", text: "No saved chats yet." }));
    return;
  }
  chatState.chats.forEach((chat) => {
    const selected = chatState.chat && chatState.chat.id === chat.id;
    const button = chatEl(
      "button",
      {
        type: "button",
        class: selected ? "chat-item selected" : "chat-item",
        title: chat.title,
        onclick: () => openChat(chat.id),
      },
      chatEl("span", { class: "chat-item-title", text: chat.title }),
      chatEl("span", {
        class: "chat-item-meta",
        text: `${chat.message_count} message(s)${chat.attachment_count ? ` · ${chat.attachment_count} file(s)` : ""}`,
      })
    );
    list.append(button);
  });
}

function renderAttachments() {
  const box = node("attachments");
  box.replaceChildren();
  const attachments = chatState.chat ? chatState.chat.attachments || [] : [];
  if (!attachments.length) return;
  attachments.forEach((attachment) => {
    const status =
      attachment.text_status === "parsed"
        ? `${attachment.char_count.toLocaleString()} characters read`
        : attachment.text_status === "empty"
          ? "no text was found in this file"
          : `could not be read: ${chatText(attachment.parse_error, "unknown reason")}`;
    box.append(
      chatEl(
        "span",
        { class: "chip", title: `${attachment.filename} — ${status}` },
        chatEl("span", { class: "chip-name", text: attachment.filename }),
        chatEl("span", { class: "chip-meta", text: `${bytes(attachment.size_bytes)} · ${status}` }),
        chatEl("button", {
          type: "button",
          class: "chip-remove",
          text: "×",
          title: `Remove ${attachment.filename} from this chat`,
          onclick: () => detachFile(attachment.id),
        })
      )
    );
  });
}

function appendMessage(message) {
  const pending = node("conversation").querySelector(".empty-state");
  if (pending) pending.remove();
  const conversation = node("conversation");
  conversation.append(
    chatEl(
      "article",
      { class: `message ${message.role}` },
      chatEl("header", { class: "message-head" }, [
        chatEl("span", {
          class: "message-role",
          text: message.role === "user" ? "You" : "Qwen (local model)",
        }),
        chatEl("span", { class: "message-time", text: chatText(message.created_at) }),
      ]),
      chatEl("div", { class: "message-body", text: chatText(message.content) })
    )
  );
  conversation.scrollTop = conversation.scrollHeight;
}

function renderConversation() {
  const conversation = node("conversation");
  conversation.replaceChildren();
  const messages = chatState.chat ? chatState.chat.messages || [] : [];
  if (!messages.length) {
    conversation.append(
      chatEl("p", {
        class: "empty-state",
        text: "Ask anything, or attach a PDF, DOCX, TXT or Markdown file to work on it with the model.",
      })
    );
  } else {
    messages.forEach((message) => appendMessage(message));
  }
  node("chat-title").textContent = chatState.chat ? chatState.chat.title : "New chat";
  const lastReport = [...messages].reverse().find((item) => item.context_report);
  renderContext(lastReport ? lastReport.context_report : null);
  // A turn that failed to generate stays a single turn: offer to answer it again rather than
  // inviting the user to type the same message a second time.
  const last = messages[messages.length - 1];
  if (last && last.role === "user") {
    notice(
      "warning",
      "This message has no answer yet",
      "The last turn did not produce an answer. Retry answers this same message; it is never sent twice.",
      retryButton()
    );
  }
  conversation.scrollTop = conversation.scrollHeight;
}

function renderContext(report) {
  const box = node("context");
  box.replaceChildren();
  if (!report) {
    box.append(
      chatEl("p", {
        class: "hint",
        text: "No context summary yet. Send a message, or reopen a chat and look at its last answer.",
      })
    );
    return;
  }
  const rows = [
    ["Configured context window (num_ctx)", report.num_ctx.toLocaleString()],
    ["Input budget used", `${report.input_budget_tokens.toLocaleString()} tokens`],
    ["Output headroom kept", `${report.output_headroom_tokens.toLocaleString()} tokens`],
    ["Instructions", report.system_tokens.toLocaleString()],
    ["Conversation history", `${report.history_tokens.toLocaleString()} (${report.history_messages_included} message(s))`],
    ["Attached documents", report.attachment_tokens.toLocaleString()],
    ["Corpus material", `${report.corpus_tokens.toLocaleString()} (${report.corpus_sources} excerpt(s))`],
    ["This message", report.current_message_tokens.toLocaleString()],
    ["Total sent", report.total_tokens.toLocaleString()],
    ["Corpus consulted", report.corpus_invoked ? "yes" : report.corpus_requested ? "asked, nothing used" : "no"],
    [
      "Generation retries",
      report.generation_retries
        ? `yes — ${report.generation_retries} (${chatText(report.retry_reason)})`
        : "no",
    ],
  ];
  const list = chatEl("dl", { class: "kv" });
  rows.forEach(([key, value]) => {
    list.append(chatEl("dt", { text: key }), chatEl("dd", { text: String(value) }));
  });
  box.append(list);

  if ((report.attachments || []).length) {
    box.append(chatEl("h3", { class: "section-title", text: "Documents in this request" }));
    const documents = chatEl("ul", { class: "plain-list" });
    report.attachments.forEach((usage) => {
      documents.append(
        chatEl("li", {
          text: `${usage.filename}: ${usage.included_chars.toLocaleString()} of ${usage.total_chars.toLocaleString()} characters${usage.truncated ? " (truncated at a paragraph boundary)" : ""}`,
        })
      );
    });
    box.append(documents);
  }

  if ((report.omitted || []).length) {
    box.append(chatEl("h3", { class: "section-title", text: "Left out because of the context limit" }));
    const omitted = chatEl("ul", { class: "plain-list" });
    report.omitted.forEach((item) => {
      omitted.append(
        chatEl("li", {
          text: `${item.kind}: ${item.label} — ${item.reason}${item.tokens ? ` (${item.tokens} tokens)` : ""}. ${item.detail}`,
        })
      );
    });
    box.append(omitted);
  }
}

function renderInfo() {
  const info = chatState.info;
  if (!info) return;
  const hint = node("chat-hint");
  hint.textContent = `Model: ${info.model} · context window ${info.num_ctx.toLocaleString()} tokens · attaches ${info.accepted_formats.join(", ")} up to ${bytes(info.max_attachment_bytes)} · ${info.max_attachments_per_chat} files per chat`;
}

/* ------------------------------------------------------------------ actions */

async function loadChats() {
  const payload = await chatJSON("/api/chats");
  chatState.chats = payload.chats || [];
  renderChatList();
}

async function loadInfo() {
  chatState.info = await chatJSON("/api/chat/info");
  node("consult-corpus").parentElement.hidden = !chatState.info.corpus_retrieval_available;
  renderInfo();
}

async function openChat(chatId) {
  chatState.chat = await chatJSON(`/api/chats/${encodeURIComponent(chatId)}`);
  // Clear any notice from the previous chat *before* rendering, because rendering decides
  // whether this chat needs one (an unanswered last turn offers Retry).
  notice(null, "");
  renderChatList();
  renderAttachments();
  renderConversation();
  node("composer-input").focus();
}

async function newChat() {
  const chat = await chatJSON("/api/chats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  await loadChats();
  await openChat(chat.id);
}

async function renameChat() {
  if (!chatState.chat) return;
  const current = chatState.chat.title;
  const title = window.prompt("Name this chat", current);
  if (title === null) return;
  const trimmed = title.trim();
  if (!trimmed) {
    notice("warning", "A chat needs a name", "The title was left empty, so nothing changed.");
    return;
  }
  const updated = await chatJSON(`/api/chats/${encodeURIComponent(chatState.chat.id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: trimmed }),
  });
  chatState.chat = { ...chatState.chat, title: updated.title };
  await loadChats();
  renderConversation();
}

async function deleteChat() {
  if (!chatState.chat) return;
  const title = chatState.chat.title;
  if (!window.confirm(`Delete "${title}" and its attachments? This cannot be undone.`)) return;
  await chatJSON(`/api/chats/${encodeURIComponent(chatState.chat.id)}`, { method: "DELETE" });
  chatState.chat = null;
  await loadChats();
  if (chatState.chats.length) {
    await openChat(chatState.chats[0].id);
  } else {
    await newChat();
  }
}

async function uploadFile(file) {
  if (!chatState.chat) await newChat();
  const info = chatState.info || {};
  const extension = `.${(file.name.split(".").pop() || "").toLowerCase()}`;
  if (info.accepted_formats && !info.accepted_formats.includes(extension)) {
    notice(
      "error",
      `"${file.name}" cannot be attached`,
      `Attach ${info.accepted_formats.join(", ")} files.`
    );
    return;
  }
  if (info.max_attachment_bytes && file.size > info.max_attachment_bytes) {
    notice(
      "error",
      `"${file.name}" is too large (${bytes(file.size)})`,
      `The limit for one attachment is ${bytes(info.max_attachment_bytes)}.`
    );
    return;
  }
  const buffer = await file.arrayBuffer();
  const attachment = await chatJSON(
    `/api/chats/${encodeURIComponent(chatState.chat.id)}/attachments`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        content_base64: bytesToBase64(buffer),
      }),
    }
  );
  chatState.chat.attachments = [...(chatState.chat.attachments || []), attachment];
  renderAttachments();
  await loadChats();
  if (attachment.text_status === "parsed") {
    notice(
      "info",
      `"${attachment.filename}" is attached to this chat`,
      `${attachment.char_count.toLocaleString()} characters were read and stay with the conversation. It is not added to your permanent corpus.`
    );
  } else {
    notice(
      "warning",
      `"${attachment.filename}" is attached, but its text could not be read`,
      chatText(attachment.parse_error, "The file was stored; its content cannot be sent to the model.")
    );
  }
}

async function attachFiles(files) {
  const list = Array.from(files || []);
  if (!list.length) return;
  setBusy(true);
  try {
    for (const file of list) {
      await uploadFile(file);
    }
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function detachFile(attachmentId) {
  const attachment = (chatState.chat.attachments || []).find((item) => item.id === attachmentId);
  if (
    attachment &&
    !window.confirm(`Remove "${attachment.filename}" from this chat? The file is deleted.`)
  ) {
    return;
  }
  try {
    await chatJSON(
      `/api/chats/${encodeURIComponent(chatState.chat.id)}/attachments/${encodeURIComponent(attachmentId)}`,
      { method: "DELETE" }
    );
    chatState.chat.attachments = (chatState.chat.attachments || []).filter(
      (item) => item.id !== attachmentId
    );
    renderAttachments();
    await loadChats();
  } catch (error) {
    showError(error);
  }
}

async function sendMessage(event) {
  if (event) event.preventDefault();
  if (chatState.busy) return;
  const input = node("composer-input");
  const message = input.value.trim();
  if (!message) {
    notice("warning", "There is nothing to send", "Type a message first.");
    return;
  }
  if (!chatState.chat) await newChat();
  const consult = node("consult-corpus").checked;
  notice(null, "");
  setBusy(true);
  try {
    const payload = await chatJSON(
      `/api/chats/${encodeURIComponent(chatState.chat.id)}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          consult_corpus: consult ? true : null,
        }),
      }
    );
    input.value = "";
    applyAnswer(payload);
    await refreshTitles();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

/* Ask the server to answer the last user message again. The user message is not resent: the
   server regenerates the answer for the turn that is already in the conversation. */
async function retryLast() {
  if (chatState.busy || !chatState.chat) return;
  notice(null, "");
  setBusy(true);
  try {
    const payload = await chatJSON(
      `/api/chats/${encodeURIComponent(chatState.chat.id)}/retry`,
      { method: "POST" }
    );
    applyAnswer(payload);
    await refreshTitles();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

function applyAnswer(payload) {
  appendMessage(payload.user_message);
  if (payload.assistant_message) {
    appendMessage(payload.assistant_message);
  } else {
    showFailure(payload.failure || {}, true);
  }
  renderContext(payload.context);
}

async function refreshTitles() {
  await loadChats();
  const listed = chatState.chats.find((item) => item.id === chatState.chat.id);
  if (listed) {
    chatState.chat.title = listed.title;
    node("chat-title").textContent = listed.title;
    renderChatList();
  }
}

/* ------------------------------------------------------------------ views */

function showAdvanced(show) {
  node("chat-view").hidden = show;
  node("advanced-view").hidden = !show;
  const toggle = node("advanced-toggle");
  toggle.textContent = show ? "Back to chat" : "Advanced";
  toggle.setAttribute("aria-pressed", show ? "true" : "false");
}

/* --------------------------------------------------------------------- boot */

async function chatBoot() {
  node("new-chat").addEventListener("click", () => {
    newChat().catch(showError);
  });
  node("composer").addEventListener("submit", sendMessage);
  node("rename-chat").addEventListener("click", () => {
    renameChat().catch(showError);
  });
  node("delete-chat").addEventListener("click", () => {
    deleteChat().catch(showError);
  });
  node("attach-input").addEventListener("change", (event) => {
    attachFiles(event.target.files);
    event.target.value = "";
  });
  node("advanced-toggle").addEventListener("click", () => {
    // The chat is visible now, so the click shows the advanced screen.
    showAdvanced(!node("chat-view").hidden);
  });
  const input = node("composer-input");
  input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      node("composer").requestSubmit();
    }
  });

  try {
    await loadInfo();
    await loadChats();
    if (chatState.chats.length) {
      await openChat(chatState.chats[0].id);
    } else {
      await newChat();
    }
  } catch (error) {
    showError(error);
  }
}

document.addEventListener("DOMContentLoaded", chatBoot);
