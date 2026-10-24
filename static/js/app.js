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

  // Quick topics chips
  document.querySelectorAll(".chip-tag").forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      queryInput.value = q;
      launchAgentResearch();
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
async function loadDocuments() {
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) return;
    const docs = await res.json();
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
async function launchAgentResearch() {
  const query = queryInput.value.trim();
  if (!query) {
    alert("Please enter a research topic or query first.");
    queryInput.focus();
    return;
  }

  if (isRunning) return;
  isRunning = true;
  btnRunAgent.disabled = true;
  btnRunAgent.innerHTML = `<span>⏳ Researching...</span>`;
  agentStatusText.textContent = "Agent Executing ReAct Loop...";
  agentStatusBadge.style.color = "var(--accent-cyan)";
  agentStatusBadge.style.borderColor = "var(--accent-cyan)";

  // Reset UI
  currentCitations = [];
  traceLogsContainer.innerHTML = "";
  startTime = Date.now();
  startTimer();

  const apiKey = apiKeyInput.value.trim();
  const depth = depthSelect.value;
  const useDocs = checkUseDocs.checked;
  const useWeb = checkUseWeb.checked;

  try {
    const response = await fetch("/api/research/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: query,
        depth: depth,
        api_key: appConfig.visitor_api_keys_enabled ? apiKey : null,
        use_documents: useDocs,
        search_web: useWeb
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // keep last incomplete chunk

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const jsonStr = line.slice(6);
          try {
            const eventPayload = JSON.parse(jsonStr);
            handleStreamEvent(eventPayload);
          } catch (e) {
            console.error("Error parsing SSE JSON:", e, jsonStr);
          }
        }
      }
    }
  } catch (err) {
    addTraceLog("error", "Execution Error", err.message);
  } finally {
    stopTimer();
    isRunning = false;
    btnRunAgent.disabled = false;
    btnRunAgent.innerHTML = `<span>⚡ Launch Agentic Research</span>`;
    agentStatusText.textContent = "ReAct Agent Ready";
    agentStatusBadge.style.color = "var(--accent-emerald)";
    agentStatusBadge.style.borderColor = "rgba(16, 185, 129, 0.25)";
  }
}

function handleStreamEvent(eventObj) {
  const event = eventObj.event;
  const data = eventObj.data;

  switch (event) {
    case "plan":
      renderPlanSteps(data);
      break;

    case "step_start":
      markStepActive(data.step_id);
      break;

    case "thought":
      addTraceLog("thought", "Cognitive Reasoning (Thought)", data.thought);
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

  reportHeaderTitle.textContent = report.query;
  reportTimestamp.textContent = new Date((report.created_at || Date.now() / 1000) * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  reportDuration.textContent = report.metrics?.total_duration_seconds != null ? `${report.metrics.total_duration_seconds}s` : "—";
  reportCitationsCount.textContent = `${report.citations?.length || 0} Citations`;
  const reportModeLabels = { extractive_fallback: "Source excerpts", no_evidence: "No evidence", verified_quotes: "Verified quotes" };
  reportModelBadge.textContent = report.metrics?.model || reportModeLabels[report.metrics?.mode] || "Evidence report";

  // Executive summary
  execSummaryContent.innerHTML = formatMarkdownWithCitations(report.executive_summary);

  // Key Findings
  findingsContainer.innerHTML = report.key_findings.map(f => `
    <div class="finding-item">
      <h4>${escapeHtml(f.heading)}</h4>
      <p>${formatMarkdownWithCitations(f.content)}</p>
      ${f.key_takeaway ? `<div class="key-takeaway">💡 Takeaway: ${escapeHtml(f.key_takeaway)}</div>` : ""}
    </div>
  `).join("");

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
      <p>"${escapeHtml(c.snippet)}"</p>
      <div style="font-size: 0.72rem; color: var(--accent-cyan); margin-top: 0.35rem;">
        Type: ${c.source_type.toUpperCase()}${c.source_type === "document" ? ` • Query term match: ${(c.relevance_score * 100).toFixed(0)}%` : ""}
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

  modalTitle.textContent = `Citation [${cite.id}]: ${cite.title}`;
  modalBody.innerHTML = `
    <div style="margin-bottom: 1rem;">
      <span class="status-badge" style="display: inline-flex;">${cite.source_type.toUpperCase()} SOURCE</span>
      ${cite.source_type === "document" ? `<span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.5rem;">Query term match: ${(cite.relevance_score * 100).toFixed(0)}%</span>` : ""}
    </div>
    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: var(--radius-sm); border-left: 3px solid var(--accent-cyan); font-style: italic; color: #e2e8f0; font-size: 0.92rem; line-height: 1.6; margin-bottom: 1rem;">
      "${escapeHtml(cite.snippet)}"
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted);">
      <strong>Source Location:</strong> ${escapeHtml(cite.url_or_page || "Document Context")}
    </div>
    ${cite.doc_id && cite.chunk_id ? `<button class="btn btn-primary" id="btnViewSource" style="margin-top: 1rem;">Open source</button>` : ""}
  `;

  modalBody.querySelector("#btnViewSource")?.addEventListener("click", () => {
    citationModal.classList.remove("open");
    openSourceViewer(cite);
  });

  citationModal.classList.add("open");
}

async function openSourceViewer(cite) {
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
    const selectedQuote = finding?.content?.replace(/\s*\[\d+\]\s*$/, "").replace(/\.\.\.$/, "").trim();
    const sourceSnippet = cite.snippet.replace(/\.\.\.$/, "").trim();
    const excerpt = selectedQuote && source.content.includes(selectedQuote) ? selectedQuote : sourceSnippet;
    const offset = source.content.indexOf(excerpt);
    textBox.innerHTML = offset >= 0
      ? `${escapeHtml(source.content.slice(0, offset))}<mark>${escapeHtml(excerpt)}</mark>${escapeHtml(source.content.slice(offset + excerpt.length))}`
      : escapeHtml(source.content);
    status.textContent = `Page ${source.page_number || 1} · Highlighted indexed passage`;
    if (source.pdf_available) {
      pdfPanel.style.display = "block";
      image.src = `/api/documents/${encodeURIComponent(cite.doc_id)}/pages/${source.page_number || 1}/preview?quote=${encodeURIComponent(excerpt)}`;
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
    const cards = [
      ["Hit @ 1", `${(metrics.hit_at_1 * 100).toFixed(0)}%`],
      ["Hit @ 3", `${(metrics.hit_at_3 * 100).toFixed(0)}%`],
      ["Source @ 1", `${(metrics.source_at_1 * 100).toFixed(0)}%`],
      ["Abstention", `${(metrics.abstention_rate * 100).toFixed(0)}%`],
      ["MRR @ 3", metrics.mrr_at_3.toFixed(3)],
      ["Mean search time", `${metrics.mean_latency_ms.toFixed(1)} ms`]
    ];
    container.innerHTML = `
      <p class="evaluation-description">${escapeHtml(data.dataset)} · ${data.questions} questions · ${data.documents} documents · ${data.chunks} chunks</p>
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
    `;
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
      }
      if (currentReport.citations?.length) {
        lines.push("## Sources", "");
        for (const cite of currentReport.citations) {
          lines.push(`- [${cite.id}] ${cite.title}${cite.url_or_page ? ` — ${cite.url_or_page}` : ""}`);
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
              <span class="status-badge" style="padding: 0.15rem 0.5rem; font-size: 0.7rem;">${r.model || 'Gemini 3.8 Flash'}</span>
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
  try {
    const res = await fetch(`/api/reports/${reportId}`);
    if (!res.ok) {
      alert("Could not load report from SQLite");
      return;
    }
    const report = await res.json();
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

