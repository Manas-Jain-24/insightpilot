<div align="center">

# 🧠 InsightPilot
### Autonomous Multimodal Research & Knowledge Agent

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%203.8%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Pydantic](https://img.shields.io/badge/Pydantic-V2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

*A research assistant with document search, web retrieval, tool calling, and live streaming reports.*

[Features](#-key-features) • [Architecture](#-architecture) • [Quickstart](#-quickstart) • [API Specs](#-api-endpoints) • [Resume Highlights](#-cv--resume-highlights)

</div>

---

## 📌 Executive Overview

**InsightPilot** searches uploaded documents and web sources, then presents cited excerpts for a research question.

Unlike conventional single-turn LLM chatbots that hallucinate citations and suffer from static context limits, InsightPilot deploys a **ReAct (Reasoning + Acting) execution loop**:
1. **Plans & Deconstructs** complex research queries into atomic sub-goals.
2. **Retrieves & Ranks** document chunks via SQLite full-text search (RAG).
3. **Calls External Tools** to ingest real-time web knowledge and state-of-the-art benchmarks.
4. **Checks Quotes** selected by Gemini against the cited excerpts. Unsupported selections are dropped.
5. **Streams Progress** and the resulting source excerpts via Server-Sent Events (SSE).

---

## 🏗️ Architecture

```
                                  [ User Query / Uploaded Documents ]
                                                   │
                                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Async Backend                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│   ┌────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         Research Orchestrator (ReAct Loop)                     │   │
│   │                                                                                │   │
│   │   [Plan Generator] ──► [Thought] ──► [Tool Execution] ──► [Observation] ───────┼───┼──► SSE Event Stream
│   │          ▲                                   │                                 │   │    (HTTP/text-event-stream)
│   │          │                                   ▼                                 │   │
│   │          │                       ┌───────────────────────┐                     │   │
│   │          │                       │       Tool Suite      │                     │   │
│   │          │                       ├───────────────────────┤                     │   │
│   │          │                       │ 1. Document Search    │                     │   │
│   │          │                       │ 2. Web Search API     │                     │   │
│   │          │                       │ 3. Quote Selector     │                     │   │
│   │          │                       │ 4. Quote Verifier     │                     │   │
│   │          │                       └───────────────────────┘                     │   │
│   │          └───────────────────────────────────┘                                 │   │
│   └────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                            │
│                                           ▼                                            │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────────────────┐
│   │  SQLite Document Search     │   │      Gemini (optional)       │   │      Persistent SQLite DB   │
│   │ (pypdf + SQLite FTS5/BM25) │   │ (google-genai / Extractive) │   │   (WAL Mode / Auto-Save)    │
│   └─────────────────────────────┘   └─────────────────────────────┘   └─────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Frontend Workstation                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  • Live Agent Cognitive Trace (Thought ➔ Action ➔ Observation)                         │
│  • Source Excerpts with Interactive Clickable Citation Modals                           │
│  • Report History and Export                                                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Autonomous ReAct Agent Loop**: Explicit cognitive decomposition where the agent reasons, invokes tools, observes results, and evaluates completion.
- **Indexed Retrieval-Augmented Generation (RAG)**:
  - Multi-format ingestion (`.pdf`, `.md`, `.txt`, `.json`, `.csv`) with recursive sliding-window chunking and sentence boundary preservation.
  - Persistent SQLite FTS5 search with BM25 ranking and page-level source tracking.
- **Quote Verification**: Gemini selects excerpts, and the server accepts only substantial quotes copied from the matching cited source. Unsupported selections are dropped; if all fail, the report falls back to source excerpts.
- **Source Viewer**: Document citations open the indexed passage with a highlighted quote. Uploaded PDFs are retained locally and open at the cited page beside the passage. Older PDFs uploaded before this feature need to be uploaded again for preview.
- **Evaluation Dashboard**: A 36-question local benchmark reports hit@1, hit@3, source match, abstention, reciprocal rank, and retrieval latency. The dataset and every case result are visible in the app.
- **Persistent SQLite Storage & Session History**: Auto-saves research briefs to local SQLite with a responsive History Drawer for 1-click reload, search, and deletion across restarts.
- **Web Retrieval**: Queries DuckDuckGo Instant Answer and Wikipedia, checks the topic against source titles, and selects answer-relevant sentences. These sources are best suited to established topics; they do not provide comprehensive current news or a general web search index. If the evidence is insufficient, the report says so.
- **Real-Time SSE Streaming**: Async generator architecture streaming incremental thoughts, tool latency, and final report JSON to the client.
- **Optional Gemini Selection**:
  - *Gemini Mode*: Uses the configured model to select evidence; the server checks each selected quote against its citation.
  - *No-key Mode*: Selects relevant sentences from available sources, or returns an explicit no-evidence result. Broad comparisons need directly supporting passages; no-key mode does not invent a comparison from unrelated excerpts.
- **Export Capabilities**: 1-click export to Markdown (`.md`) or printable PDF format.

---

## 🚀 Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Manas-Jain-24/insightpilot.git
cd insightpilot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env and paste your GEMINI_API_KEY (Optional)
```
> *Note: An API key is optional. A key in `.env` stays on the server and is used in local or demo mode. Without one, InsightPilot displays source excerpts. Uploaded documents persist in `data/documents.db` across restarts in local mode. Uploads are limited to 10 MB.*

Open **Evaluation** in the top bar to run the bundled benchmark. The benchmark uses [`evaluation/questions.json`](evaluation/questions.json) and the two sample documents. It measures retrieval behavior; it does not establish whether statements in those sample documents are true.

### 3. Launch the Server

```bash
python main.py
```

Open your browser to: **`http://127.0.0.1:8080`**  
Interactive API Docs (Swagger): **`http://127.0.0.1:8080/docs`**

---

## 🌐 Public Portfolio Deployment

The included [`render.yaml`](render.yaml) deploys a single public demo service on Render. Connect this repository as a **Blueprint** in the Render dashboard and approve creation of the `insightpilot-demo` web service. The Blueprint installs dependencies, starts `uvicorn backend.app:app` on Render's port, checks `/api/health`, and waits for CI checks before automatic deploys. Python 3.13 is selected in [`.python-version`](.python-version).

The deployed service sets `APP_MODE=demo`. This mode indexes only the bundled sample documents in memory, including a small PDF for the **Try PDF Source Viewer** quick pick. Visitors can run research, inspect sources, run the retrieval evaluation, print reports, and download Markdown. Uploads, document and report deletion, saved report history, and visitor supplied Gemini keys are disabled. A server-side `GEMINI_API_KEY` is used when configured; the included Render Blueprint does not set one. Model calls can incur usage charges. The service stores no visitor research after a restart. Research and evaluation requests have single-process rate limits. Do not set `APP_MODE=local` on a public service: that mode has shared uploads and reports without accounts.

For local editing, leave `APP_MODE=local` in `.env` (or unset it) and run `python main.py`. Local mode stores documents and reports in two SQLite files under `data/`. Set `DATA_DIR` if you want both files on a different persistent directory. A hosted service that allows private user uploads needs authentication, per-user document access, abuse controls, and durable storage before enabling local mode features for visitors.

CI runs backend tests and JavaScript syntax checks on pushes and pull requests. Direct Python dependencies are pinned in `requirements.txt`. Check the **Actions** tab before deploying a change. After Render deploys, visit `/api/health` and run a sample research query to confirm the live service.

---

## 💡 Why InsightPilot vs. ChatGPT or Public Chatbots?

| Dimension | Prompting ChatGPT / Gemini Web | InsightPilot Autonomous Agent Architecture |
| :--- | :--- | :--- |
| **Citations & Verification** | Generated citations need source checking. | The server accepts selected quotes only when they occur in the matching retrieved excerpt. This verifies attribution, not the truth of an external source. |
| **Data Privacy & Governance** | Review data handling before uploading sensitive material. | Documents and history are stored locally. Using Gemini or web search sends request data to external services. |
| **Execution Paradigm** | **Single-turn black-box completion**: Gives a one-off linear best guess without showing intermediate reasoning. | **Autonomous ReAct Agent Loop**: Explicit cognitive decomposition (`Thought ➔ Action ➔ Observation`) with dynamic tool selection and live SSE streaming. |
| **Output Structure** | Free-form prose can be difficult to audit. | **Typed Pydantic Report**: Source excerpts, citation records, and verification status. |
| **Enterprise Integration** | Cannot be integrated programmatically into microservices, CI/CD, or automated ETL workflows. | **Production FastAPI REST & SSE API**: Programmatic endpoints for document ingestion, streaming research, and SQLite persistent report querying. |

---

## 🧪 Running Automated Tests

```bash
python -m unittest tests/test_agent.py
```

The automated tests verify:
- Semantic chunking & PDF text extraction.
- Persistent SQLite full-text indexing and retrieval.
- Multi-strategy web search tool execution.
- Knowledge graph entity building.
- ReAct orchestrator event streaming.
- SQLite database persistent storage & CRUD operations.
- PDF source retention, citation source lookup, and benchmark repeatability.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Healthcheck, model configuration & document index stats |
| `POST` | `/api/documents/upload` | Multipart file upload (PDF/TXT/MD/JSON/CSV) & search indexing |
| `GET` | `/api/documents` | List all indexed document metadata |
| `GET` | `/api/documents/{id}/file` | Open a retained PDF source |
| `GET` | `/api/documents/{id}/chunks/{chunk_id}` | Retrieve an indexed source passage |
| `DELETE` | `/api/documents/{id}` | Delete document from search index |
| `GET` | `/api/evaluation` | Run the bundled retrieval benchmark |
| `POST` | `/api/research/stream` | Server-Sent Events (SSE) streaming ReAct research loop |
| `GET` | `/api/reports` | List all saved research reports from persistent SQLite |
| `GET` | `/api/reports/{id}` | Retrieve full structured research report by ID |
| `DELETE` | `/api/reports/{id}` | Delete report from persistent SQLite database |
| `GET` | `/api/research/quick-topics` | Pre-bundled sample research topics |
| `POST` | `/api/export/markdown` | Generate downloadable Markdown report |

---

## 💼 CV / Resume Highlights

Here is how you can describe this project on your resume:

> **InsightPilot | Research & Source Verification Assistant** *(Python, FastAPI, Gemini API, SQLite FTS5, Pydantic)*
> - Built a FastAPI research workflow that plans queries, searches documents and web sources, and streams progress to the browser.
> - Developed a Retrieval-Augmented Generation (RAG) pipeline with sentence chunking and persistent SQLite full-text search over uploaded documents.
> - Added Server-Sent Events (SSE) streaming for tool actions and retrieved citations.
> - Added exact-quote verification for Gemini-selected evidence, an extractive fallback, and explicit no-evidence results.
> - Used Pydantic V2 models and automated tests for retrieval, uploads, persistence, and report validation.

---

## 📄 License

MIT License. Built for portfolio demonstration and production AI engineering showcases.
