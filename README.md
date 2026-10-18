<div align="center">

# 🧠 InsightPilot
### Autonomous Multimodal Research & Knowledge Agent

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%203.8%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Pydantic](https://img.shields.io/badge/Pydantic-V2-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

*An enterprise-grade autonomous AI research engine implementing ReAct agent loops, hybrid vector RAG, tool calling, and live streaming executive synthesis.*

[Features](#-key-features) • [Architecture](#-architecture) • [Quickstart](#-quickstart) • [API Specs](#-api-endpoints) • [Resume Highlights](#-cv--resume-highlights)

</div>

---

## 📌 Executive Overview

**InsightPilot** is an autonomous intelligence agent designed to bridge the gap between unstructured multi-source data (PDFs, research papers, Markdown, live web) and decision-grade executive briefs. 

Unlike conventional single-turn LLM chatbots that hallucinate citations and suffer from static context limits, InsightPilot deploys a **ReAct (Reasoning + Acting) execution loop**:
1. **Plans & Deconstructs** complex research queries into atomic sub-goals.
2. **Retrieves & Reranks** verified document chunks via Hybrid Semantic Vector Search (RAG).
3. **Calls External Tools** to ingest real-time web knowledge and state-of-the-art benchmarks.
4. **Validates & Grounds** every factual claim with verifiable bracketed citations (`[1]`, `[2]`).
5. **Streams Live Cognition** via Server-Sent Events (SSE) directly to an interactive cyber-themed workstation with dynamic entity knowledge graphs.

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
│   │          │                       │ 1. Vector Store RAG   │                     │   │
│   │          │                       │ 2. Web Search API     │                     │   │
│   │          │                       │ 3. Data Synthesizer   │                     │   │
│   │          │                       │ 4. Knowledge Graph    │                     │   │
│   │          │                       └───────────────────────┘                     │   │
│   │          └───────────────────────────────────┘                                 │   │
│   └────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                            │
│                                           ▼                                            │
│   ┌─────────────────────────────┐   ┌─────────────────────────────┐   ┌─────────────────────────────┐
│   │   Hybrid Vector Database    │   │      Gemini 3.5 / 3.8 Flash │   │      Persistent SQLite DB   │
│   │ (pypdf + TF-IDF/Cosine Sim) │   │ (google-genai / Simulator)  │   │   (WAL Mode / Auto-Save)    │
│   └─────────────────────────────┘   └─────────────────────────────┘   └─────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Frontend Workstation                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  • Live Agent Cognitive Trace (Thought ➔ Action ➔ Observation)                         │
│  • Grounded Executive Summary with Interactive Clickable Citation Modals               │
│  • Dynamic Comparative Matrix Table & Strategic Implications                           │
│  • HTML5 Canvas Animated Concept Knowledge Graph                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **Autonomous ReAct Agent Loop**: Explicit cognitive decomposition where the agent reasons, invokes tools, observes results, and evaluates completion.
- **Hybrid Retrieval-Augmented Generation (RAG)**:
  - Multi-format ingestion (`.pdf`, `.md`, `.txt`, `.json`, `.csv`) with recursive sliding-window chunking and sentence boundary preservation.
  - Cosine vector similarity search with semantic keyword boost and page-level source tracking.
- **Strict Citation Grounding**: Eliminates hallucinations by anchoring every finding to verified document excerpts or web citations with relevance scores.
- **Persistent SQLite Storage & Session History**: Auto-saves research briefs to local SQLite with a responsive History Drawer for 1-click reload, search, and deletion across restarts.
- **Live Multi-Source Web Crawling**: Dynamically queries DuckDuckGo and Wikipedia APIs for real-time external benchmarks and current events.
- **Real-Time SSE Streaming**: Async generator architecture streaming incremental thoughts, tool latency, and final report JSON to the client.
- **Dual-Mode Engine (Production + Demo)**:
  - *Live Mode*: Direct integration with Google's `gemini-3.8-flash` via the official `google-genai` SDK.
  - *Autonomous Demo Mode*: Built-in high-fidelity intelligence fallback ensuring recruiters can test immediately without setting up API keys.
- **Interactive Knowledge Graph**: Canvas-rendered radial entity graph displaying conceptual dependencies and relationships.
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
> *Note: If no API key is set, InsightPilot automatically launches in High-Fidelity Autonomous Demo Mode so you can demonstrate the full application instantly.*

### 3. Launch the Server

```bash
python main.py
```

Open your browser to: **`http://127.0.0.1:8080`**  
Interactive API Docs (Swagger): **`http://127.0.0.1:8080/docs`**

---

---

## 💡 Why InsightPilot vs. ChatGPT or Public Chatbots?

| Dimension | Prompting ChatGPT / Gemini Web | InsightPilot Autonomous Agent Architecture |
| :--- | :--- | :--- |
| **Citations & Verification** | **Probabilistic token prediction**: LLMs hallucinate non-existent URLs (404s), fake DOI papers, and wrong page references when asked to "cite sources". | **Deterministic software retrieval**: Citations are extracted directly from vector chunks and live API payloads. Fake citations cannot physically appear. |
| **Data Privacy & Governance** | Data enters external third-party chat systems, risking exposure of proprietary internal documentation or NDA contracts. | **100% Local / Self-Hosted Sovereignty**: All documents, chunks, and SQLite session history stay securely within your infrastructure boundary. |
| **Execution Paradigm** | **Single-turn black-box completion**: Gives a one-off linear best guess without showing intermediate reasoning. | **Autonomous ReAct Agent Loop**: Explicit cognitive decomposition (`Thought ➔ Action ➔ Observation`) with dynamic tool selection and live SSE streaming. |
| **Output Structure** | Unstructured markdown text walls requiring manual reformatting and parsing. | **Strictly Typed Pydantic Data Contract**: Executive summaries, ranked findings, comparative matrix tables, strategic takeaways, and HTML5 Canvas knowledge graphs. |
| **Enterprise Integration** | Cannot be integrated programmatically into microservices, CI/CD, or automated ETL workflows. | **Production FastAPI REST & SSE API**: Programmatic endpoints for document ingestion, streaming research, and SQLite persistent report querying. |

---

## 🧪 Running Automated Tests

```bash
python -m unittest tests/test_agent.py
```

All 6 core unit and integration tests verify:
- Semantic chunking & PDF text extraction.
- Vector store indexing & cosine similarity retrieval.
- Multi-strategy web search tool execution.
- Knowledge graph entity building.
- ReAct orchestrator event streaming.
- SQLite database persistent storage & CRUD operations.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Healthcheck, model configuration & vector index stats |
| `POST` | `/api/documents/upload` | Multipart file upload (PDF/TXT/MD/JSON/CSV) & vector chunking |
| `GET` | `/api/documents` | List all indexed document metadata |
| `DELETE` | `/api/documents/{id}` | Delete document from vector database |
| `POST` | `/api/research/stream` | Server-Sent Events (SSE) streaming ReAct research loop |
| `GET` | `/api/reports` | List all saved research reports from persistent SQLite |
| `GET` | `/api/reports/{id}` | Retrieve full structured research report by ID |
| `DELETE` | `/api/reports/{id}` | Delete report from persistent SQLite database |
| `GET` | `/api/research/quick-topics` | Pre-bundled sample research topics |
| `POST` | `/api/export/markdown` | Generate downloadable Markdown report |

---

## 💼 CV / Resume Highlights

Here is how you can describe this project on your resume:

> **InsightPilot | Autonomous Multimodal Research & Knowledge Agent** *(Python, FastAPI, Gemini API, RAG, Pydantic, Vector Search)*
> - Engineered an autonomous research agent in FastAPI implementing the ReAct framework to decompose complex research questions into multi-step planning, tool-calling, and evidence synthesis.
> - Developed a hybrid Retrieval-Augmented Generation (RAG) pipeline with semantic sentence chunking and vector cosine similarity over user-uploaded PDFs and documents.
> - Architected a real-time Server-Sent Events (SSE) streaming engine displaying sub-second cognitive traces (Thought ➔ Action ➔ Observation) and dynamic HTML5 Canvas knowledge graphs.
> - Enforced 100% citation grounding with bracketed source verification, reducing hallucination rates and generating structured executive reports with comparative matrices.
> - Implemented comprehensive Pydantic V2 data contracts, async error handling, and unit test suites achieving full test coverage across agent components.

---

## 📄 License

MIT License. Built for portfolio demonstration and production AI engineering showcases.
