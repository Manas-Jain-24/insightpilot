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
const btnOpenDocs = document.getElementById("btnOpenDocs");
const btnCloseDocs = document.getElementById("btnCloseDocs");
const btnOpenSettings = document.getElementById("btnOpenSettings");
const btnCloseSettings = document.getElementById("btnCloseSettings");
const btnCloseCitation = document.getElementById("btnCloseCitation");
const docsCountPill = document.getElementById("docsCountPill");
const docsListContainer = document.getElementById("docsListContainer");
const fileDropzone = document.getElementById("fileDropzone");
const fileInput = document.getElementById("fileInput");
const apiKeyInput = document.getElementById("apiKeyInput");
const btnSaveSettings = document.getElementById("btnSaveSettings");
const btnExportMarkdown = document.getElementById("btnExportMarkdown");
const btnPrintReport = document.getElementById("btnPrintReport");

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  loadDocuments();
  setupEventListeners();
  loadSavedApiKey();
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

  btnOpenSettings.addEventListener("click", () => settingsModal.classList.add("open"));
  btnCloseSettings.addEventListener("click", () => settingsModal.classList.remove("open"));
  btnCloseCitation.addEventListener("click", () => citationModal.classList.remove("open"));

  // Close modals on backdrop click
  [docsModal, settingsModal, citationModal].forEach(modal => {
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
    const key = apiKeyInput.value.trim();
    if (key) {
      localStorage.setItem("insightpilot_gemini_key", key);
    } else {
      localStorage.removeItem("insightpilot_gemini_key");
    }
    settingsModal.classList.remove("open");
  });

  // Exports
  btnExportMarkdown.addEventListener("click", exportReportMarkdown);
  btnPrintReport.addEventListener("click", () => window.print());
}

function loadSavedApiKey() {
  const savedKey = localStorage.getItem("insightpilot_gemini_key");
  if (savedKey) {
    apiKeyInput.value = savedKey;
  }
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
              ${doc.file_type} • ${Math.round(doc.size_bytes / 1024)} KB • ${doc.chunks_count} vector chunks
            </div>
          </div>
        </div>
        <button class="delete-file-btn" onclick="deleteDoc('${doc.id}')" title="Delete document">✕</button>
      </div>
    `).join("");
  } catch (err) {
    console.error("Error loading documents:", err);
  }
}

async function handleFileUpload(file) {
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

  const apiKey = localStorage.getItem("insightpilot_gemini_key") || "";
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
        api_key: apiKey,
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
  reportTimestamp.textContent = new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  reportDuration.textContent = `${report.metrics?.total_duration_seconds || '2.4'}s`;
  reportCitationsCount.textContent = `${report.citations?.length || 0} Citations`;
  reportModelBadge.textContent = report.metrics?.model || "Gemini 3.8 Flash";

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
        Type: ${c.source_type.toUpperCase()} • Relevance Score: ${(c.relevance_score * 100).toFixed(0)}%
        ${c.url_or_page ? ` • Reference: ${escapeHtml(c.url_or_page)}` : ""}
      </div>
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

  // Render Concept Knowledge Graph on Canvas
  if (report.knowledge_graph) {
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
      <span style="font-size: 0.8rem; color: var(--text-muted); margin-left: 0.5rem;">
        Relevance Score: ${(cite.relevance_score * 100).toFixed(1)}%
      </span>
    </div>
    <div style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: var(--radius-sm); border-left: 3px solid var(--accent-cyan); font-style: italic; color: #e2e8f0; font-size: 0.92rem; line-height: 1.6; margin-bottom: 1rem;">
      "${escapeHtml(cite.snippet)}"
    </div>
    <div style="font-size: 0.82rem; color: var(--text-muted);">
      <strong>Source Location:</strong> ${escapeHtml(cite.url_or_page || "Document Context")}
    </div>
  `;

  citationModal.classList.add("open");
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
    const res = await fetch("/api/export/markdown", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentReport)
    });
    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `InsightPilot_Report_${currentReport.id}.md`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    }
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
