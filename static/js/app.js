/**
 * InsightPilot Frontend Engine
 * Real-time SSE Agent Orchestration & Interactive Knowledge Visualization
 */

// State
let currentReport = null;
let currentCitations = [];
let isRunning = false;
let startTime = null;
let timerInterval = null;
let kgAnimationId = null;
let indexedDocuments = [];
const selectedDocumentIds = new Set();
let activeRequest = null;
let lastAttempt = null;
let appConfig = { mode: "local", uploads_enabled: true, history_enabled: true, visitor_api_keys_enabled: true };

// DOM Elements
const queryInput = document.getElementById("queryInput");
const depthSelect = document.getElementById("depthSelect");
const checkUseDocs = document.getElementById("checkUseDocs");
const checkUseWeb = document.getElementById("checkUseWeb");
const btnRunAgent = document.getElementById("btnRunAgent");
const agentStatusBadge = document.getElementById("agentStatusBadge");
const agentStatusText = document.getElementById("agentStatusText");
const traceTimer = document.getElementById("traceTimer");
const planStepsContainer = document.getElementById("planStepsContainer");
const traceLogsContainer = document.getElementById("traceLogsContainer");

// Report Elements
const reportPanel = document.getElementById("reportPanel");
const reportEmptyState = document.getElementById("reportEmptyState");
const reportContent = document.getElementById("reportContent");
const reportHeaderTitle = document.getElementById("reportHeaderTitle");
const reportTimestamp = document.getElementById("reportTimestamp");
const reportDuration = document.getElementById("reportDuration");
const reportCitationsCount = document.getElementById("reportCitationsCount");
const reportModelBadge = document.getElementById("reportModelBadge");
const execSummaryContent = document.getElementById("execSummaryContent");
const findingsContainer = document.getElementById("findingsContainer");
const comparativeMatrixSection = document.getElementById("comparativeMatrixSection");
const matrixHead = document.getElementById("matrixHead");
const matrixBody = document.getElementById("matrixBody");
const strategicSection = document.getElementById("strategicSection");
const strategicList = document.getElementById("strategicList");
const citationsReferenceList = document.getElementById("citationsReferenceList");
const kgCanvas = document.getElementById("kgCanvas");

// Modals
const docsModal = document.getElementById("docsModal");
const settingsModal = document.getElementById("settingsModal");
const citationModal = document.getElementById("citationModal");
const historyModal = document.getElementById("historyModal");
const btnOpenDocs = document.getElementById("btnOpenDocs");
const btnCloseDocs = document.getElementById("btnCloseDocs");
const btnOpenSettings = document.getElementById("btnOpenSettings");
const btnCloseSettings = document.getElementById("btnCloseSettings");
const btnCloseCitation = document.getElementById("btnCloseCitation");
const btnOpenHistory = document.getElementById("btnOpenHistory");
const btnCloseHistory = document.getElementById("btnCloseHistory");
const docsCountPill = document.getElementById("docsCountPill");
const docsListContainer = document.getElementById("docsListContainer");
const historyCountPill = document.getElementById("historyCountPill");
const historyListContainer = document.getElementById("historyListContainer");
const fileDropzone = document.getElementById("fileDropzone");
const fileInput = document.getElementById("fileInput");
const apiKeyInput = document.getElementById("apiKeyInput");
const btnSaveSettings = document.getElementById("btnSaveSettings");
const btnExportMarkdown = document.getElementById("btnExportMarkdown");
const btnPrintReport = document.getElementById("btnPrintReport");
const sourceModal = document.getElementById("sourceModal");
const evaluationModal = document.getElementById("evaluationModal");
const searchAllDocuments = document.getElementById("searchAllDocuments");
const documentScopeList = document.getElementById("documentScopeList");
const runStatus = document.getElementById("runStatus");
const runStatusText = document.getElementById("runStatusText");
const btnCancelResearch = document.getElementById("btnCancelResearch");
const btnRetryResearch = document.getElementById("btnRetryResearch");

// Initialize on page load
document.addEventListener("DOMContentLoaded", async () => {
  setupEventListeners();
  try {
    const response = await fetch("/api/config");
    if (response.ok) appConfig = await response.json();
  } catch (error) {
    console.error("Could not load app configuration:", error);
  }
  if (appConfig.mode === "demo") {
    const demoNotice = document.getElementById("demoNotice");
    demoNotice.hidden = false;
    demoNotice.textContent = appConfig.model_enabled
      ? "Public demo: research uses the server's Gemini key and bundled sample documents. Uploads and saved history are available when running locally."
      : "Public demo: answers are selected from source excerpts. Broad comparisons and current events need stronger sources or a configured model. Uploads and saved history are available when running locally.";
    btnOpenHistory.style.display = "none";
    btnOpenSettings.style.display = "none";
    fileDropzone.style.display = "none";
  }
  loadDocuments();
  if (appConfig.history_enabled) loadHistory();
});

function setupEventListeners() {
  btnRunAgent.addEventListener("click", launchAgentResearch);
  btnCancelResearch.addEventListener("click", () => activeRequest?.abort());
  btnRetryResearch.addEventListener("click", () => {
    if (lastAttempt) launchAgentResearch(lastAttempt);
  });
  checkUseDocs.addEventListener("change", renderDocumentScope);
  searchAllDocuments.addEventListener("change", renderDocumentScope);
  documentScopeList.addEventListener("change", (event) => {
    const id = event.target.dataset.docId;
    if (!id) return;
    if (event.target.checked) selectedDocumentIds.add(id);
    else selectedDocumentIds.delete(id);
    renderDocumentScope();
    [...documentScopeList.querySelectorAll("input")].find(input => input.dataset.docId === id)?.focus();
  });

  // Quick topics chips
  document.querySelectorAll(".chip-tag").forEach(chip => {
    chip.addEventListener("click", () => {
      if (isRunning) return;
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      const source = indexedDocuments.find(doc => doc.filename === chip.dataset.document);
      if (source) {
        selectedDocumentIds.clear();
        selectedDocumentIds.add(source.id);
        searchAllDocuments.checked = false;
        checkUseDocs.checked = true;
        checkUseWeb.checked = false;
        renderDocumentScope();
      }
      queryInput.focus();
    });
  });

  // Enter to submit (Shift+Enter for newline)
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      launchAgentResearch();
    }
  });

  // Modals toggles
  btnOpenDocs.addEventListener("click", () => {
    docsModal.classList.add("open");
    loadDocuments();
  });
  btnCloseDocs.addEventListener("click", () => docsModal.classList.remove("open"));

  btnOpenHistory.addEventListener("click", () => {
    historyModal.classList.add("open");
    loadHistory();
  });
  btnCloseHistory.addEventListener("click", () => historyModal.classList.remove("open"));

  btnOpenSettings.addEventListener("click", () => settingsModal.classList.add("open"));
  btnCloseSettings.addEventListener("click", () => settingsModal.classList.remove("open"));
  btnCloseCitation.addEventListener("click", () => citationModal.classList.remove("open"));
  document.getElementById("btnCloseSource").addEventListener("click", () => {
    sourceModal.classList.remove("open");
    document.getElementById("sourcePdfImage").removeAttribute("src");
  });
  document.getElementById("btnOpenEvaluation").addEventListener("click", openEvaluation);
  document.getElementById("btnCloseEvaluation").addEventListener("click", () => evaluationModal.classList.remove("open"));

  // Close modals on backdrop click
  [docsModal, settingsModal, citationModal, historyModal, sourceModal, evaluationModal].forEach(modal => {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.remove("open");
    });
  });

  // File Upload Handling
  fileDropzone.addEventListener("click", () => fileInput.click());
  fileDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileDropzone.classList.add("dragover");
  });
  fileDropzone.addEventListener("dragleave", () => fileDropzone.classList.remove("dragover"));
  fileDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    fileDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Save Settings
  btnSaveSettings.addEventListener("click", () => {
    settingsModal.classList.remove("open");
  });

  // Exports
  btnExportMarkdown.addEventListener("click", exportReportMarkdown);
  btnPrintReport.addEventListener("click", () => window.print());
}

// ---------------- Document Ingestion ----------------
function renderDocumentScope() {
  const all = searchAllDocuments.checked;
  const count = all ? indexedDocuments.length : selectedDocumentIds.size;
  document.getElementById("sourceScopeSummary").textContent = checkUseDocs.checked
    ? `Choose documents · ${all ? "all " : ""}${count} selected`
    : "Document search is off";
  searchAllDocuments.disabled = isRunning || !checkUseDocs.checked;
  documentScopeList.innerHTML = indexedDocuments.length ? indexedDocuments.map(doc => `
    <label class="document-choice">
      <input type="checkbox" data-doc-id="${escapeHtml(doc.id)}"
        ${all || selectedDocumentIds.has(doc.id) ? "checked" : ""}
        ${all || !checkUseDocs.checked || isRunning ? "disabled" : ""}>
      <span>${escapeHtml(doc.filename)}<small>${doc.chunks_count} indexed passages</small></span>
    </label>`).join("") : '<p>No documents available. Add a document or enable web sources.</p>';
  document.querySelectorAll(".chip-tag[data-document]").forEach(chip => {
    chip.hidden = !indexedDocuments.some(doc => doc.filename === chip.dataset.document);
    chip.disabled = isRunning;
  });
}

async function loadDocuments() {
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) throw new Error("Could not load documents");
    const docs = await res.json();
    indexedDocuments = docs;
    const available = new Set(docs.map(doc => doc.id));
    for (const id of selectedDocumentIds) if (!available.has(id)) selectedDocumentIds.delete(id);
    renderDocumentScope();
    docsCountPill.textContent = docs.length;

    if (docs.length === 0) {
      docsListContainer.innerHTML = '<div style="font-size: 0.85rem; color: var(--text-muted);">No documents indexed yet. Upload a PDF or TXT to get started.</div>';
      return;
    }

    docsListContainer.innerHTML = docs.map(doc => `
      <div class="file-item">
        <div class="file-info">
          <span>📄</span>
          <div>
            <strong>${escapeHtml(doc.filename)}</strong>
            <div style="font-size: 0.72rem; color: var(--text-muted);">
              ${doc.file_type} • ${Math.round(doc.size_bytes / 1024)} KB • ${doc.chunks_count} indexed ${doc.chunks_count === 1 ? "chunk" : "chunks"}
            </div>
          </div>
        </div>
        ${appConfig.uploads_enabled ? `<button class="delete-file-btn" onclick="deleteDoc('${doc.id}')" title="Delete document">✕</button>` : ""}
      </div>
    `).join("");
  } catch (err) {
    document.getElementById("sourceScopeSummary").textContent = "Could not load documents · open Documents to retry";
    console.error("Error loading documents:", err);
  }
}

async function handleFileUpload(file) {
  if (!appConfig.uploads_enabled) return;
  const formData = new FormData();
  formData.append("file", file);

  fileDropzone.innerHTML = `<div class="dropzone-icon">⏳</div><h4>Processing and chunking ${escapeHtml(file.name)}...</h4>`;

  try {
    const res = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData
    });

    if (res.ok) {
      await loadDocuments();
      fileDropzone.innerHTML = `
        <div class="dropzone-icon">✅</div>
        <h4>${escapeHtml(file.name)} indexed successfully!</h4>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem;">Click or drop more files to index.</p>
      `;
    } else {
      const err = await res.json();
      alert(`Upload failed: ${err.detail || "Server error"}`);
      fileDropzone.innerHTML = `<div class="dropzone-icon">📄</div><h4>Drag & Drop research PDFs, Markdown, or TXT</h4>`;
    }
  } catch (err) {
    alert(`Upload error: ${err.message}`);
    fileDropzone.innerHTML = `<div class="dropzone-icon">📄</div><h4>Drag & Drop research PDFs, Markdown, or TXT</h4>`;
  }
}

window.deleteDoc = async function(docId) {
  if (!confirm("Are you sure you want to remove this document from the vector store?")) return;
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
    if (res.ok) {
      loadDocuments();
    }
  } catch (err) {
    console.error("Error deleting doc:", err);
  }
};

// ---------------- Agent Research Execution (SSE Stream) ----------------
const reportTrail = [];

document.getElementById("followupForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = document.getElementById("followupInput").value.trim();
  if (currentReport && query) launchAgentResearch({ followupQuery: query, parentReport: currentReport });
});
document.getElementById("btnPreviousReport").addEventListener("click", () => {
  if (isRunning) return;
  if (!reportTrail.length) {
    if (appConfig.history_enabled && currentReport?.metrics?.parent_report_id) window.loadReportById(currentReport.metrics.parent_report_id);
    return;
  }
  currentReport = reportTrail.pop();
  renderReport(currentReport);
});

async function launchAgentResearch(options = {}) {
  if (isRunning) return;
  const query = (options.requestBody?.query || options.followupQuery || queryInput.value).trim();
  const body = options.requestBody || {
    query,
    depth: depthSelect.value,
    api_key: appConfig.visitor_api_keys_enabled ? apiKeyInput.value.trim() : null,
    use_documents: checkUseDocs.checked,
    search_web: checkUseWeb.checked,
    document_ids: searchAllDocuments.checked ? [] : [...selectedDocumentIds],
    parent_report_id: options.parentReport?.id || null
  };
  let validation = "";
  if (query.length < 3) validation = "Enter a question with at least 3 characters.";
  else if (!body.parent_report_id && !body.use_documents && !body.search_web) validation = "Choose documents or web sources before starting.";
  else if (!options.requestBody && !body.parent_report_id && body.use_documents && !searchAllDocuments.checked && !body.document_ids.length) validation = "Select at least one document, or switch to searching all documents.";
  else if (!options.requestBody && !body.parent_report_id && body.use_documents && !indexedDocuments.length && !body.search_web) validation = "Add a document or enable web sources before starting.";
  if (validation) {
    runStatus.hidden = false;
    runStatus.dataset.state = "error";
    runStatusText.textContent = validation;
    btnRetryResearch.hidden = true;
    queryInput.focus();
    return;
  }

  lastAttempt = { requestBody: body, parentReport: options.parentReport };
  const previousReport = currentReport;
  let receivedReport = false;
  let completed = false;
  isRunning = true;
  activeRequest = new AbortController();
  runStatus.hidden = false;
  runStatus.dataset.state = "running";
  runStatusText.textContent = `Researching: ${query}`;
  btnCancelResearch.hidden = false;
  btnRetryResearch.hidden = true;
  reportContent.style.display = "none";
  reportEmptyState.style.display = "none";
  reportPanel.setAttribute("aria-busy", "true");
  [queryInput, depthSelect, checkUseDocs, checkUseWeb, ...document.querySelectorAll(".chip-tag")].forEach(input => input.disabled = true);
  renderDocumentScope();
  btnRunAgent.disabled = true;
  btnOpenHistory.disabled = true;
  document.getElementById("btnFollowUp").disabled = true;
  document.getElementById("btnPreviousReport").disabled = true;
  btnRunAgent.innerHTML = `<span>⏳ Researching...</span>`;
  agentStatusText.textContent = "Research in progress";
  agentStatusBadge.style.color = "var(--accent-cyan)";
  agentStatusBadge.style.borderColor = "var(--accent-cyan)";

  // Reset UI
  currentCitations = [];
  traceLogsContainer.innerHTML = "";
  startTime = Date.now();
  startTimer();

  try {
    const response = await fetch("/api/research/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: activeRequest.signal,
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(typeof error.detail === "string" ? error.detail : `Server returned status ${response.status}`);
    }

    await ResearchStream.consume(response.body, (eventPayload) => {
      if (eventPayload.event === "report") {
        if (options.parentReport) reportTrail.push(options.parentReport);
        else reportTrail.length = 0;
        receivedReport = true;
      }
      handleStreamEvent(eventPayload);
    });
    completed = true;
    runStatus.dataset.state = "complete";
    runStatusText.textContent = `Research complete: ${query}`;
  } catch (err) {
    const stopped = err.name === "AbortError";
    const message = stopped ? "Research stopped. You can change the question or retry." : err.message;
    addTraceLog(stopped ? "observation" : "error", stopped ? "Research stopped" : "Research interrupted", message);
    runStatus.dataset.state = stopped ? "stopped" : "error";
    runStatusText.textContent = `${message}${previousReport && !receivedReport ? " Your previous report is shown below." : ""}`;
    btnRetryResearch.hidden = false;
    if (!receivedReport && previousReport) {
      currentReport = previousReport;
      renderReport(previousReport);
    } else if (!receivedReport) reportEmptyState.style.display = "flex";
  } finally {
    stopTimer();
    isRunning = false;
    activeRequest = null;
    btnCancelResearch.hidden = true;
    reportPanel.setAttribute("aria-busy", "false");
    [queryInput, depthSelect, checkUseDocs, checkUseWeb, ...document.querySelectorAll(".chip-tag")].forEach(input => input.disabled = false);
    renderDocumentScope();
    btnRunAgent.disabled = false;
    btnOpenHistory.disabled = false;
    document.getElementById("btnFollowUp").disabled = false;
    document.getElementById("btnPreviousReport").disabled = false;
    btnRunAgent.innerHTML = `<span>Research question</span>`;
    agentStatusText.textContent = completed ? "Report ready" : "Ready to retry";
    agentStatusBadge.style.color = "var(--accent-emerald)";
    agentStatusBadge.style.borderColor = "rgba(16, 185, 129, 0.25)";
  }
}

function handleStreamEvent(eventObj) {
  const event = eventObj.event;
  const data = eventObj.data;

  switch (event) {
    case "status":
      addTraceLog("observation", "Research status", data.message);
      runStatusText.textContent = data.message;
      break;
    case "plan":
      renderPlanSteps(data);
      break;

    case "step_start":
      markStepActive(data.step_id);
      break;

    case "thought":
      addTraceLog("thought", "Research step", data.thought);
      break;

    case "action":
      const toolLabel = data.tool_name.replace("_", " ").toUpperCase();
      const inputStr = JSON.stringify(data.input, null, 2);
      addTraceLog("action", `Action: Tool [${toolLabel}]`, `Executing tool with parameters:\n${inputStr}`, true);
      break;

    case "observation":
      const latencyStr = data.latency_ms ? ` (${data.latency_ms}ms)` : "";
      addTraceLog("observation", `Observation${latencyStr}`, data.observation);
      break;

    case "citation_added":
      currentCitations.push(data);
      break;

    case "step_complete":
      markStepCompleted(data.step_id);
      break;

    case "report":
      currentReport = data;
      renderReport(data);
      break;

    case "done":
      addTraceLog("observation", "Synthesis Complete", `Report generated in ${data.elapsed_seconds}s with ${data.total_citations} grounded citations.`);
      loadHistory();
      break;

    case "error":
      addTraceLog("error", "Error", data.message);
      break;
  }
}

// ---------------- Trace Logging Helpers ----------------
function renderPlanSteps(steps) {
  planStepsContainer.innerHTML = steps.map(s => `
    <div class="step-card" id="step_card_${s.id}">
      <div class="step-badge">${s.id}</div>
      <div class="step-info">
        <h4>${escapeHtml(s.title)}</h4>
        <p>${escapeHtml(s.description)}</p>
      </div>
    </div>
  `).join("");
}

function markStepActive(stepId) {
  document.querySelectorAll(".step-card").forEach(c => c.classList.remove("active"));
  const card = document.getElementById(`step_card_${stepId}`);
  if (card) {
    card.classList.add("active");
  }
}

function markStepCompleted(stepId) {
  const card = document.getElementById(`step_card_${stepId}`);
  if (card) {
    card.classList.remove("active");
    card.classList.add("completed");
  }
}

function addTraceLog(type, header, content, isCode = false) {
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;

  const headerDiv = document.createElement("div");
  headerDiv.className = `log-header ${type}-header`;
  headerDiv.textContent = header;
  entry.appendChild(headerDiv);

  if (isCode) {
    const pre = document.createElement("pre");
    pre.className = "log-code";
    pre.textContent = content;
    entry.appendChild(pre);
  } else {
    const p = document.createElement("p");
    p.textContent = content;
    entry.appendChild(p);
  }

  traceLogsContainer.appendChild(entry);
  traceLogsContainer.scrollTop = traceLogsContainer.scrollHeight;
}

function startTimer() {
  traceTimer.textContent = "0.0s";
  timerInterval = setInterval(() => {
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    traceTimer.textContent = `${elapsed}s`;
  }, 100);
}

function stopTimer() {
  if (timerInterval) clearInterval(timerInterval);
}

// ---------------- Report Rendering ----------------
function renderReport(report) {
  reportEmptyState.style.display = "none";
  reportContent.style.display = "block";
  document.getElementById("followupInput").value = "";
  document.getElementById("btnPreviousReport").hidden = reportTrail.length === 0 && !(appConfig.history_enabled && report.metrics?.parent_report_id);
  const notices = [];
  const scope = report.metrics?.source_scope;
  if (scope) {
    const documents = scope.documents.length ? `${scope.documents.length} document${scope.documents.length === 1 ? "" : "s"} (${scope.documents.slice(0, 3).join(", ")}${scope.documents.length > 3 ? ", …" : ""})` : "no documents";
    notices.push(`Sources requested: ${documents}; web ${scope.web_enabled ? "enabled" : "off"}.`);
  }
  if (report.metrics?.web_fallback_used || report.metrics?.model_rate_limited) notices.push("Google grounding is unavailable. This report uses available source excerpts.");
  if (report.metrics?.web_cache_hit) notices.push("Using cached web sources; their retrieval dates appear with the evidence.");
  if (report.metrics?.retrieval_mode) notices.push(report.metrics.retrieval_mode === "hybrid" ? "Document search used meaning and keywords." : "Document search used keywords.");
  if (report.metrics?.parent_report_id) notices.push("Follow-up based on the previous report's sources.");
  if (["extractive_fallback", "semantic_excerpts"].includes(report.metrics?.mode)) notices.push("Showing extracted passages. They may not answer every part of your question; inspect the sources for context.");
  const notice = document.getElementById("researchNotice");
  notice.textContent = notices.join(" ");
  notice.hidden = notices.length === 0;

  reportHeaderTitle.textContent = report.query;
  reportTimestamp.textContent = new Date((report.created_at || Date.now() / 1000) * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  reportDuration.textContent = report.metrics?.total_duration_seconds != null ? `${report.metrics.total_duration_seconds}s` : "—";
  reportCitationsCount.textContent = `${report.citations?.length || 0} Citations`;
  const reportModeLabels = { extractive_fallback: "Source excerpts", no_evidence: "No evidence", verified_quotes: "Verified quotes", grounded_synthesis: "Cited answer" };
  reportModelBadge.textContent = report.metrics?.model || reportModeLabels[report.metrics?.mode] || "Evidence report";

  // Executive summary
  execSummaryContent.innerHTML = formatMarkdownWithCitations(report.executive_summary);

  // Key Findings
  findingsContainer.innerHTML = report.key_findings.map((f, index) => `
    <div class="finding-item">
      <h4>${escapeHtml(f.heading)}</h4>
      <p>${formatMarkdownWithCitations(f.content)}</p>
      <button class="btn btn-secondary inspect-evidence" data-finding-index="${index}">Inspect evidence for finding ${index + 1}</button>
      ${f.key_takeaway ? `<div class="key-takeaway">💡 Takeaway: ${escapeHtml(f.key_takeaway)}</div>` : ""}
    </div>
  `).join("");
  findingsContainer.querySelectorAll(".inspect-evidence").forEach(button => {
    button.addEventListener("click", () => showFindingEvidence(Number(button.dataset.findingIndex)));
  });

  // Comparative Matrix
  if (report.comparative_matrix && Object.keys(report.comparative_matrix).length > 0) {
    comparativeMatrixSection.style.display = "block";
    const headers = Object.keys(report.comparative_matrix);
    matrixHead.innerHTML = `<tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;

    const rowCount = report.comparative_matrix[headers[0]]?.length || 0;
    let rowsHtml = "";
    for (let r = 0; r < rowCount; r++) {
      rowsHtml += `<tr>${headers.map(h => `<td>${escapeHtml(String(report.comparative_matrix[h][r] || ""))}</td>`).join("")}</tr>`;
    }
    matrixBody.innerHTML = rowsHtml;
  } else {
    comparativeMatrixSection.style.display = "none";
  }

  // Strategic Implications
  if (report.strategic_implications && report.strategic_implications.length > 0) {
    strategicSection.style.display = "block";
    strategicList.innerHTML = report.strategic_implications.map(imp => `<li>${escapeHtml(imp)}</li>`).join("");
  } else {
    strategicSection.style.display = "none";
  }

  // Grounded Citations Reference List
  citationsReferenceList.innerHTML = report.citations.map(c => `
    <div class="citation-card" id="cite_ref_${c.id}">
      <h4>[${c.id}] ${escapeHtml(c.title)}</h4>
      <p>${c.evidence_type === "grounded_claim" ? "Search-grounded claim" : "Source excerpt"}: "${escapeHtml(c.snippet)}"</p>
      <div style="font-size: 0.72rem; color: var(--accent-cyan); margin-top: 0.35rem;">
        Type: ${c.source_type.toUpperCase()}${c.source_type === "document" ? ` • Retrieval score: ${Number(c.relevance_score || 0).toFixed(2)}` : ""}
        ${c.url_or_page ? ` • Reference: ${escapeHtml(c.url_or_page)}` : ""}
      </div>
      ${c.doc_id && c.chunk_id ? `<button class="btn btn-secondary source-open-btn" data-cite-id="${c.id}" style="margin-top: 0.65rem;">Open source</button>` : ""}
    </div>
  `).join("");

  // Attach click listeners to all inline [1], [2] tags
  document.querySelectorAll(".cite-tag").forEach(tag => {
    tag.addEventListener("click", (e) => {
      e.preventDefault();
      const citeId = parseInt(tag.getAttribute("data-cite-id"), 10);
      showCitationModal(citeId);
    });
  });
  document.querySelectorAll(".source-open-btn").forEach(button => {
    button.addEventListener("click", () => {
      const cite = currentReport?.citations?.find(c => c.id === Number(button.dataset.citeId));
      if (cite) openSourceViewer(cite);
    });
  });

  // Render Concept Knowledge Graph on Canvas
  const graphSection = document.getElementById("knowledgeGraphSection");
  if (graphSection) graphSection.style.display = report.knowledge_graph?.nodes?.length ? "block" : "none";
  if (report.knowledge_graph?.nodes?.length) {
    renderKnowledgeGraph(report.knowledge_graph);
  }
}

function formatMarkdownWithCitations(text) {
  if (!text) return "";
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\n\n/g, "<br><br>");
  formatted = formatted.replace(/\n/g, "<br>");

  // Replace [1], [2] with clickable tags
  formatted = formatted.replace(/\[(\d+)\]/g, (match, p1) => {
    return `<a href="#cite_ref_${p1}" class="cite-tag" data-cite-id="${p1}">[${p1}]</a>`;
  });

  return formatted;
}

function showCitationModal(citeId) {
  const cite = currentReport?.citations?.find(c => c.id === citeId);
  if (!cite) return;

  const modalTitle = document.getElementById("citationModalTitle");
  const modalBody = document.getElementById("citationModalBody");
  let webLink = "";
  if (cite.source_type === "web" && cite.url_or_page) {
    try {
      const parsed = new URL(cite.url_or_page);
      if (["https:", "http:"].includes(parsed.protocol)) {
        webLink = `<a href="${escapeHtml(parsed.href)}" target="_blank" rel="noopener noreferrer">Open cited page ↗</a>`;
      }
    } catch (_) { /* Keep the citation text when the URL is malformed. */ }
  }

  modalTitle.textContent = `Citation [${cite.id}]: ${cite.title}`;
  modalBody.innerHTML = `
    <div style="margin-bottom: 1rem;">
      <span class="status-badge" style="display: inline-flex;">${cite.source_type.toUpperCase()} SOURCE</span>
      ${cite.source_type === "document" ? `<span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.5rem;">Retrieval score: ${Number(cite.relevance_score || 0).toFixed(2)} (not answer confidence)</span>` : ""}
    </div>
    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: var(--radius-sm); border-left: 3px solid var(--accent-cyan); font-style: italic; color: #e2e8f0; font-size: 0.92rem; line-height: 1.6; margin-bottom: 1rem;">
      ${cite.evidence_type === "grounded_claim" ? "Search-grounded claim" : "Source excerpt"}: "${escapeHtml(cite.snippet)}"
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted);">
      <strong>Source Location:</strong> ${escapeHtml(cite.url_or_page || "Document Context")}
    </div>
    ${webLink ? `<div style="margin-top: 0.8rem;">${webLink}</div>` : ""}
    ${cite.doc_id && cite.chunk_id ? `<button class="btn btn-primary" id="btnViewSource" style="margin-top: 1rem;">Open source</button>` : ""}
  `;

  modalBody.querySelector("#btnViewSource")?.addEventListener("click", () => {
    citationModal.classList.remove("open");
    openSourceViewer(cite);
  });

  citationModal.classList.add("open");
}

function evidenceDates(cite) {
  const retrieved = cite.retrieved_at ? new Date(cite.retrieved_at * 1000).toLocaleString() : "Not recorded";
  return `Published: ${cite.published_at || "Unknown"} · Retrieved: ${retrieved}`;
}

function showFindingEvidence(index) {
  const finding = currentReport?.key_findings?.[index];
  if (!finding) return;
  const body = document.getElementById("citationModalBody");
  document.getElementById("citationModalTitle").textContent = `Evidence for finding ${index + 1}`;
  body.innerHTML = `<p class="evidence-claim">${escapeHtml(finding.content)}</p>` + (finding.citations || []).map(id => {
    const cite = currentReport.citations.find(c => c.id === id);
    if (!cite) return "";
    const link = (finding.evidence || []).find(e => e.citation_id === id);
    const label = cite.evidence_type === "grounded_claim" ? "Google Search annotation · not independently verified" : link ? "Copied support checked against the source excerpt" : "Citation excerpt · claim support not recorded";
    let webLink = "";
    try {
      const url = new URL(cite.url_or_page);
      if (["https:", "http:"].includes(url.protocol)) webLink = `<a href="${escapeHtml(url.href)}" target="_blank" rel="noopener noreferrer">Open cited page ↗</a>`;
    } catch (_) { /* Document source or absent URL. */ }
    return `<div class="evidence-card"><h4>[${id}] ${escapeHtml(cite.title)}</h4>
      <p>${escapeHtml(label)}</p><blockquote>${escapeHtml(link?.quote || cite.snippet)}</blockquote>
      <p class="evidence-date">${escapeHtml(evidenceDates(cite))}</p>
      ${cite.doc_id && cite.chunk_id ? `<button class="btn btn-secondary evidence-source" data-citation="${id}">Open original passage</button>` : webLink}</div>`;
  }).join("");
  body.querySelectorAll(".evidence-source").forEach(button => button.addEventListener("click", () => {
    const id = Number(button.dataset.citation);
    const cite = currentReport.citations.find(c => c.id === id);
    const quote = (finding.evidence || []).find(e => e.citation_id === id)?.quote;
    citationModal.classList.remove("open");
    openSourceViewer(cite, quote);
  }));
  citationModal.classList.add("open");
}

async function openSourceViewer(cite, supportQuote = null) {
  const status = document.getElementById("sourceViewerStatus");
  const textBox = document.getElementById("sourceViewerText");
  const pdfPanel = document.getElementById("sourcePdfPanel");
  const image = document.getElementById("sourcePdfImage");
  document.getElementById("sourceViewerTitle").textContent = cite.title;
  status.textContent = "Loading source…";
  textBox.textContent = "";
  image.removeAttribute("src");
  pdfPanel.style.display = "none";
  sourceModal.classList.add("open");
  try {
    const url = `/api/documents/${encodeURIComponent(cite.doc_id)}/chunks/${encodeURIComponent(cite.chunk_id)}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error("Source excerpt is unavailable. It may have been deleted.");
    const source = await response.json();
    const finding = currentReport?.key_findings?.find(item => item.citations?.includes(cite.id));
    const selectedQuote = supportQuote || finding?.evidence?.find(e => e.citation_id === cite.id)?.quote || finding?.content?.replace(/\s*\[\d+\]\s*$/, "").replace(/\.\.\.$/, "").trim();
    const sourceSnippet = cite.snippet.replace(/\.\.\.$/, "").trim();
    const excerpt = selectedQuote && source.content.includes(selectedQuote) ? selectedQuote : sourceSnippet;
    const offset = source.content.indexOf(excerpt);
    textBox.innerHTML = offset >= 0
      ? `${escapeHtml(source.content.slice(0, offset))}<mark>${escapeHtml(excerpt)}</mark>${escapeHtml(source.content.slice(offset + excerpt.length))}`
      : escapeHtml(source.content);
    status.textContent = `Page ${source.page_number || 1} · ${offset >= 0 ? "Highlighted indexed passage" : "Original indexed passage"}`;
    if (source.pdf_available) {
      pdfPanel.style.display = "block";
      image.src = `/api/documents/${encodeURIComponent(cite.doc_id)}/pages/${source.page_number || 1}/preview?quote=${encodeURIComponent(excerpt.slice(0, 240))}`;
    } else if (cite.source_type === "document" && cite.title.toLowerCase().includes(".pdf")) {
      status.textContent += " · Original PDF unavailable; re-upload to enable PDF preview";
    }
  } catch (error) {
    status.textContent = error.message;
  }
}

async function openEvaluation() {
  const container = document.getElementById("evaluationContent");
  container.textContent = "Running the bundled benchmark…";
  evaluationModal.classList.add("open");
  try {
    const response = await fetch("/api/evaluation");
    if (!response.ok) throw new Error("Benchmark could not run.");
    const data = await response.json();
    const metrics = data.metrics;
    const answer = data.answer_quality;
    const cards = [
      ["Hit @ 1", `${(metrics.hit_at_1 * 100).toFixed(0)}%`],
      ["Hit @ 3", `${(metrics.hit_at_3 * 100).toFixed(0)}%`],
      ["Source @ 1", `${(metrics.source_at_1 * 100).toFixed(0)}%`],
      ["Abstention", `${(metrics.abstention_rate * 100).toFixed(0)}%`],
      ["MRR @ 3", metrics.mrr_at_3.toFixed(3)],
      ["Mean search time", `${metrics.mean_latency_ms.toFixed(1)} ms`]
    ];
    container.innerHTML = `
      <p class="evaluation-description">Keyword candidate benchmark · ${escapeHtml(data.dataset)} · ${data.questions} questions · ${data.documents} documents · ${data.chunks} chunks</p>
      <div class="evaluation-metrics">${cards.map(([name, value]) => `
        <div class="evaluation-metric"><strong>${value}</strong><span>${name}</span></div>
      `).join("")}</div>
      <div style="overflow-x: auto;"><table class="evaluation-table">
        <thead><tr><th>Question</th><th>Expected source</th><th>Top source</th><th>Rank</th><th>Result</th></tr></thead>
        <tbody>${data.cases.map(row => `<tr>
          <td>${escapeHtml(row.query)}</td>
          <td>${escapeHtml(row.expected_source || "No match")}</td>
          <td>${escapeHtml(row.top_source || "None")}</td>
          <td>${row.rank || "—"}</td>
          <td class="${row.success ? "pass" : "fail"}">${row.success ? "Pass" : "Miss"}</td>
        </tr>`).join("")}</tbody>
      </table></div>
      <p class="evaluation-description">Answer quality · ${answer.questions} offline cases</p>
      <div class="evaluation-metrics">
        <div class="evaluation-metric"><strong>${(answer.metrics.answer_match_rate * 100).toFixed(0)}%</strong><span>Answer match</span></div>
        <div class="evaluation-metric"><strong>${(answer.metrics.citation_validity_rate * 100).toFixed(0)}%</strong><span>Citation validity</span></div>
        <div class="evaluation-metric"><strong>${(answer.metrics.answer_abstention_rate * 100).toFixed(0)}%</strong><span>Answer abstention</span></div>
      </div>
      <div style="overflow-x: auto;"><table class="evaluation-table">
        <thead><tr><th>Question</th><th>Answer</th><th>Citations</th></tr></thead>
        <tbody>${answer.cases.map(row => `<tr>
          <td>${escapeHtml(row.query)}</td>
          <td class="${row.answer_match ? "pass" : "fail"}">${row.answer_match ? "Pass" : "Miss"}</td>
          <td class="${row.citations_valid ? "pass" : "fail"}">${row.citations_valid ? "Valid" : "Invalid"}</td>
        </tr>`).join("")}</tbody>
      </table></div>
    `;
    if (data.semantic) {
      const semantic = data.semantic;
      container.insertAdjacentHTML("beforeend", `<h4>Keyword vs semantic retrieval · ${semantic.cases.length} independent cases</h4>
        <p class="evaluation-description">A small synthetic corpus with direct questions, paraphrases, and unrelated questions. These results do not establish general accuracy.</p>
        <table class="evaluation-table"><thead><tr><th>Method</th><th>Hit @ 1</th><th>Hit @ 3</th><th>Abstention</th></tr></thead><tbody>
        ${Object.entries(semantic.metrics).map(([mode, scores]) => `<tr><td>${escapeHtml(mode)}</td><td>${(scores.hit_at_1 * 100).toFixed(1)}%</td><td>${(scores.hit_at_3 * 100).toFixed(1)}%</td><td>${(scores.abstention * 100).toFixed(1)}%</td></tr>`).join("")}</tbody></table>
        <details><summary>Inspect each query</summary><table class="evaluation-table"><thead><tr><th>Question</th><th>Expected</th><th>Keyword</th><th>Hybrid</th></tr></thead><tbody>
        ${semantic.cases.map(row => `<tr><td>${escapeHtml(row.query)}</td><td>${escapeHtml(row.expected || "No match")}</td><td>${escapeHtml(row.keyword.top || "No match")}</td><td>${escapeHtml(row.hybrid.top || "No match")}</td></tr>`).join("")}</tbody></table></details>`);
    }
  } catch (error) {
    container.textContent = error.message;
  }
}

// ---------------- Knowledge Graph Canvas ----------------
function renderKnowledgeGraph(kg) {
  if (!kgCanvas) return;
  const ctx = kgCanvas.getContext("2d");
  const width = kgCanvas.parentElement.clientWidth;
  const height = 280;
  kgCanvas.width = width;
  kgCanvas.height = height;

  if (kgAnimationId) cancelAnimationFrame(kgAnimationId);

  const nodes = kg.nodes || [];
  const edges = kg.edges || [];

  if (nodes.length === 0) return;

  // Position nodes radially around the center
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.35;

  const positionedNodes = nodes.map((node, i) => {
    if (i === 0) {
      return { ...node, x: centerX, y: centerY, vx: 0, vy: 0, r: 24, color: "#38bdf8" };
    }
    const angle = ((i - 1) / Math.max(1, nodes.length - 1)) * (Math.PI * 2);
    return {
      ...node,
      x: centerX + Math.cos(angle) * radius,
      y: centerY + Math.sin(angle) * radius,
      vx: (Math.random() - 0.5) * 0.2,
      vy: (Math.random() - 0.5) * 0.2,
      r: 16,
      color: "#a855f7"
    };
  });

  function draw() {
    ctx.clearRect(0, 0, width, height);

    // Draw Edges
    edges.forEach(edge => {
      const src = positionedNodes.find(n => n.id === edge.source) || positionedNodes[0];
      const tgt = positionedNodes.find(n => n.id === edge.target) || positionedNodes[1];
      if (src && tgt) {
        ctx.strokeStyle = "rgba(56, 189, 248, 0.25)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Draw Nodes
    positionedNodes.forEach((node, i) => {
      // Gentle floating animation
      if (i > 0) {
        node.x += node.vx;
        node.y += node.vy;
        if (Math.abs(node.x - centerX) > radius + 20) node.vx *= -1;
        if (Math.abs(node.y - centerY) > radius + 20) node.vy *= -1;
      }

      // Outer glow
      ctx.shadowBlur = 12;
      ctx.shadowColor = node.color;

      ctx.fillStyle = node.color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
      ctx.fill();

      ctx.shadowBlur = 0; // reset

      // Label text
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(node.label, node.x, node.y + node.r + 14);
    });

    kgAnimationId = requestAnimationFrame(draw);
  }

  draw();
}

// ---------------- Export Markdown ----------------
async function exportReportMarkdown() {
  if (!currentReport) return;
  try {
    let blob;
    if (appConfig.mode === "demo") {
      const lines = [`# ${currentReport.query}`, "", currentReport.executive_summary, ""];
      for (const finding of currentReport.key_findings || []) {
        lines.push(`## ${finding.heading}`, finding.content, "");
        for (const evidence of finding.evidence || []) lines.push(`> Support [${evidence.citation_id}] (${evidence.verification}): ${evidence.quote}`, "");
      }
      if (currentReport.citations?.length) {
        lines.push("## Sources", "");
        for (const cite of currentReport.citations) {
          lines.push(`- [${cite.id}] ${cite.title}${cite.url_or_page ? ` — ${cite.url_or_page}` : ""}`);
          lines.push(`  ${evidenceDates(cite)}`);
        }
      }
      blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    } else {
      const res = await fetch("/api/export/markdown", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentReport)
      });
      if (!res.ok) throw new Error(`Export failed (${res.status})`);
      blob = await res.blob();
    }
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `InsightPilot_Report_${currentReport.id}.md`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Export error:", err);
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ---------------- Research History (SQLite) ----------------
async function loadHistory() {
  if (!appConfig.history_enabled) return;
  try {
    const res = await fetch("/api/reports");
    if (!res.ok) return;
    const reports = await res.json();
    if (historyCountPill) historyCountPill.textContent = reports.length;

    if (!historyListContainer) return;
    if (reports.length === 0) {
      historyListContainer.innerHTML = '<div style="font-size: 0.85rem; color: var(--text-muted); text-align: center; padding: 2rem 0;">No saved research reports yet. Launch a research query to auto-save reports to SQLite.</div>';
      return;
    }

    historyListContainer.innerHTML = reports.map(r => {
      const dateStr = new Date(r.created_at * 1000).toLocaleString(undefined, {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
      });
      return `
        <div class="history-item">
          <div class="history-main">
            <div class="history-title" title="${escapeHtml(r.query)}">${escapeHtml(r.query)}</div>
            <div class="history-meta">
              <span>${dateStr}</span>
              <span>•</span>
              <span class="status-badge" style="padding: 0.15rem 0.5rem; font-size: 0.7rem;">${escapeHtml(r.model || 'Model not recorded')}</span>
              <span>•</span>
              <span>${r.citations_count} Citations</span>
              <span>•</span>
              <span style="text-transform: capitalize;">${r.depth}</span>
            </div>
          </div>
          <div class="history-actions">
            <button class="btn btn-primary" style="padding: 0.35rem 0.75rem; font-size: 0.78rem;" onclick="loadReportById('${r.id}')">
              View Report
            </button>
            <button class="delete-file-btn" style="padding: 0.4rem; font-size: 0.9rem;" onclick="deleteReportById('${r.id}')" title="Delete from SQLite">
              🗑️
            </button>
          </div>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Error loading history:", err);
  }
}

window.loadReportById = async function(reportId) {
  if (isRunning) return;
  try {
    const res = await fetch(`/api/reports/${reportId}`);
    if (!res.ok) {
      alert("Could not load report from SQLite");
      return;
    }
    const report = await res.json();
    reportTrail.length = 0;
    currentReport = report;
    renderReport(report);
    historyModal.classList.remove("open");

    // Scroll to report view
    if (reportPanel) {
      reportPanel.scrollIntoView({ behavior: 'smooth' });
    }
  } catch (err) {
    alert(`Error loading report: ${err.message}`);
  }
};

window.deleteReportById = async function(reportId) {
  if (!confirm("Are you sure you want to delete this saved report from SQLite?")) return;
  try {
    const res = await fetch(`/api/reports/${reportId}`, { method: "DELETE" });
    if (res.ok) {
      loadHistory();
    }
  } catch (err) {
    console.error("Delete report error:", err);
  }
};

