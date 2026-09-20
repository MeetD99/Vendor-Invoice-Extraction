const state = { cases: [], selectedCase: null, selectedDocument: null };

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? "—")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(typeof error.detail === "string" ? error.detail : JSON.stringify(error.detail));
  }
  return response.json();
}

function status(value) {
  return `<span class="status ${escapeHtml(value)}">${escapeHtml((value || "UNKNOWN").replaceAll("_", " "))}</span>`;
}

async function refresh() {
  const [dashboard, cases] = await Promise.all([api("/api/dashboard"), api("/api/cases")]);
  state.cases = cases;
  $("#metrics").innerHTML = [
    ["Documents", dashboard.documents],
    ["Cases", Object.values(dashboard.cases).reduce((a, b) => a + b, 0)],
    ["Committed", dashboard.committed],
    ["Scheduled", dashboard.cases.PAYMENT_SCHEDULED || 0],
    ["Vendor flagged", dashboard.cases.VENDOR_FLAGGED || 0],
    ["Needs review", dashboard.cases.NEEDS_REVIEW || 0],
  ].map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`).join("");

  $("#case-list").innerHTML = cases.map(item => `
    <tr>
      <td><strong>${escapeHtml(item.case_key)}</strong></td>
      <td>${status(item.status)}</td>
      <td>${escapeHtml(item.vendor_name)}</td>
      <td>${escapeHtml(item.invoice_number)}</td>
      <td>${item.document_count}</td>
      <td>${escapeHtml(item.payment_decision)}</td>
      <td><button class="case-link" data-case-id="${item.id}">Review →</button></td>
    </tr>`).join("");
  $("#empty-state").hidden = cases.length > 0;
  document.querySelectorAll(".case-link").forEach(button => {
    button.addEventListener("click", () => openCase(Number(button.dataset.caseId)));
  });
}

async function openCase(caseId) {
  state.selectedCase = await api(`/api/cases/${caseId}`);
  state.selectedDocument = state.selectedCase.documents.find(doc => doc.document_type === "VENDOR_INVOICE") || state.selectedCase.documents[0];
  renderCase();
  if (!$("#case-dialog").open) $("#case-dialog").showModal();
}

function renderCase() {
  const detail = state.selectedCase;
  const draft = detail.draft?.data;
  const record = detail.committed?.data;
  const data = draft || record;
  const docs = detail.documents;
  const selected = state.selectedDocument;
  const extractionMessages = (detail.agent_messages || []).filter(item => item.agent_name === "extraction");
  const validationMessages = (detail.agent_messages || []).filter(item => item.agent_name === "validation");
  const payment = detail.committed?.payment;
  const fields = data ? [
    ["Vendor name", "vendor_name", data.vendor_name],
    ["Remit to", "remit_to", data.remit_to],
    ["Remit explanation", "remit_to_explanation", data.remit_to_explanation],
    ["Invoice number", "invoice_number", data.invoice_number],
    ["Invoice date", "invoice_date", data.invoice_date],
    ["Total due", "total_due", data.total_due],
    ["Total basis", "total_due_basis", data.total_due_basis],
    ["Due date", "due_date", data.due_date],
    ["Due-date basis", "due_date_basis", data.due_date_basis],
    ["Payment terms", "payment_terms", data.payment_terms],
    ["Line items", "line_items", data.line_items || []],
  ] : [];

  const renderField = ([label, name, value]) => {
    if (!draft) {
      const display = name === "line_items" ? `${value.length} extracted` : value;
      return `<div class="field"><label>${label}</label><strong>${escapeHtml(display)}</strong></div>`;
    }
    const serialized = name === "line_items" ? JSON.stringify(value, null, 2) : (value ?? "");
    const control = name === "line_items"
      ? `<textarea data-edit-field="${name}">${escapeHtml(serialized)}</textarea>`
      : `<input data-edit-field="${name}" value="${escapeHtml(serialized)}">`;
    return `<div class="field"><label>${label}</label><div class="field-edit">${control}<button class="save-field" data-save-field="${name}">Save</button></div></div>`;
  };

  $("#case-detail").innerHTML = `
    <div class="detail-head">
      <div><span class="subtle">Invoice case</span><h2>${escapeHtml(detail.case.case_key)}</h2></div>
      <button class="close" aria-label="Close">×</button>
    </div>
    <div class="detail-grid">
      <section class="document-panel">
        <div class="doc-tabs">${docs.map(doc => `
          <button class="doc-tab ${doc.id === selected?.id ? "active" : ""}" data-document-id="${doc.id}">
            ${escapeHtml(doc.original_filename)}${doc.is_current ? "" : " · superseded"}
          </button>`).join("")}</div>
        <pre class="source">${escapeHtml(selected?.raw_text || "No source text")}</pre>
      </section>
      <section class="record-panel">
        <div class="decision-summary">
          <header><strong>Current decision</strong>${status(detail.case.status)}</header>
          ${detail.case.review_reason ? `<p>${escapeHtml(detail.case.review_reason)}</p>` : ""}
          <div class="decision-grid">
            <div><span>Validation</span><strong>${escapeHtml(detail.draft?.validation_status || (record ? "COMMITTED" : "Waiting"))}</strong></div>
            <div><span>Payment</span><strong>${escapeHtml(payment?.decision || "Not scheduled")}</strong></div>
          </div>
        </div>
        ${detail.followups.length ? `<div class="timeline"><h3>Open follow-ups</h3>${detail.followups.map(item => `<div class="flag"><strong>${escapeHtml(item.followup_type.replaceAll("_", " "))}</strong><br>${escapeHtml(item.description)}</div>`).join("")}</div>` : ""}
        <h3>${draft ? "Extracted record" : record ? "Committed record" : "Awaiting extraction"}</h3>
        ${fields.map(renderField).join("")}
        ${(data?.flags || []).map(flag => `<div class="flag">${escapeHtml(flag.replaceAll("_", " "))}</div>`).join("")}
        ${data?.derivations?.length ? `<div class="evidence"><h3>Derivations</h3>${data.derivations.map(item => `<div class="evidence-item">${escapeHtml(item)}</div>`).join("")}</div>` : ""}
        ${detail.draft?.evidence?.length ? `<div class="evidence"><h3>Evidence</h3>${detail.draft.evidence.map(item => `<div class="evidence-item"><strong>${escapeHtml(item.field_name)}</strong> · ${escapeHtml(item.verification)}<br>${escapeHtml(item.excerpt)}</div>`).join("")}</div>` : ""}
        <div class="actions">
          ${detail.draft ? `<button id="commit-button" class="primary">Commit reviewed record</button>` : ""}
          <button id="reprocess-button" class="secondary">Reprocess</button>
        </div>
        <div class="timeline"><h3>Deterministic tool activity</h3>
          ${detail.tool_calls.map(call => `<div class="timeline-item"><strong>${escapeHtml(call.tool_name)}</strong><br><span class="subtle">${escapeHtml(call.status)} · ${escapeHtml(call.created_at)}</span></div>`).join("") || '<p class="subtle">No tool activity yet.</p>'}
        </div>
        ${detail.agent_messages?.length ? `<div class="agent-sections">
          ${renderAgentCard("Extraction agent", "Reads and structures source evidence", "extraction", extractionMessages)}
          ${renderAgentCard("Validation agent", "Checks evidence and makes the decision", "validation", validationMessages)}
        </div>` : ""}
      </section>
    </div>`;

  $(".close").addEventListener("click", () => $("#case-dialog").close());
  document.querySelectorAll(".doc-tab").forEach(button => button.addEventListener("click", () => {
    state.selectedDocument = docs.find(doc => doc.id === Number(button.dataset.documentId));
    renderCase();
  }));
  $("#reprocess-button").addEventListener("click", async () => {
    await api(`/api/cases/${detail.case.id}/reprocess`, { method: "POST" });
    $("#case-dialog").close();
    refresh();
  });
  $("#commit-button")?.addEventListener("click", async () => {
    try {
      await api(`/api/cases/${detail.case.id}/commit`, { method: "POST" });
      await openCase(detail.case.id);
      refresh();
    } catch (error) { alert(error.message); }
  });
  document.querySelectorAll("[data-save-field]").forEach(button => button.addEventListener("click", async () => {
    const fieldName = button.dataset.saveField;
    const control = document.querySelector(`[data-edit-field="${fieldName}"]`);
    let value = control.value.trim() || null;
    if (fieldName === "line_items") {
      try { value = JSON.parse(control.value); }
      catch { alert("Line items must be valid JSON."); return; }
    }
    button.disabled = true;
    try {
      await api(`/api/cases/${detail.case.id}/field`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ field_name: fieldName, value, reason: "Human review correction" }),
      });
      await openCase(detail.case.id);
      refresh();
    } catch (error) { alert(error.message); }
    finally { button.disabled = false; }
  }));
}

function renderAgentCard(title, subtitle, className, messages) {
  return `<section class="agent-card ${className}">
    <h3>${escapeHtml(title)} <span>${escapeHtml(subtitle)}</span></h3>
    ${messages.map(message => `<div class="timeline-item">
      <strong>${escapeHtml(message.role)}${message.tool_name ? ` · ${escapeHtml(message.tool_name)}` : ""}</strong><br>
      <span class="subtle">${escapeHtml((message.content || message.tool_calls_json || "").slice(0, 500))}</span>
    </div>`).join("") || '<p class="subtle">Not started.</p>'}
  </section>`;
}

$("#folder-input").addEventListener("change", event => {
  const files = [...event.target.files];
  $("#selection-summary").textContent = files.length ? `${files.length} files selected` : "No folder selected";
});

$("#upload-form").addEventListener("submit", async event => {
  event.preventDefault();
  const files = [...$("#folder-input").files];
  if (!files.length) return;
  const button = event.currentTarget.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "Uploading…";
  const formData = new FormData();
  files.forEach(file => formData.append("files", file, file.webkitRelativePath || file.name));
  const registry = $("#registry-input").files[0];
  if (registry) formData.append("registry", registry, registry.name);
  try {
    const result = await api("/api/imports/upload", { method: "POST", body: formData });
    const ingested = result.files.filter(item => item.status === "INGESTED").length;
    $("#upload-message").hidden = false;
    $("#upload-message").textContent = `${ingested} documents staged. Processing has started.`;
    await refresh();
  } catch (error) {
    $("#upload-message").hidden = false;
    $("#upload-message").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Upload and process";
  }
});

$("#refresh-button").addEventListener("click", refresh);
$("#case-dialog").addEventListener("click", event => {
  if (event.target === $("#case-dialog")) $("#case-dialog").close();
});

refresh();
setInterval(refresh, 5000);
