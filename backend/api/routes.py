import os
import json
import shutil
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from typing import List, Dict, Any, Optional
from backend.models import (
    ResearchRequest, DocumentMetadata, ResearchReport
)
from backend.rag.document_loader import DocumentLoader
from backend.rag.vector_store import VectorStore
from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.db.database import Database


def create_router(
    vector_store: VectorStore,
    gemini_client: GeminiClient,
    db: Optional[Database] = None
) -> APIRouter:
    router = APIRouter()
    doc_loader = DocumentLoader()
    orchestrator = ResearchOrchestrator(vector_store, gemini_client, db=db)

    @router.get("/health")
    async def health_check():
        reports_count = len(db.list_reports()) if db else 0
        return {
            "status": "healthy",
            "service": "InsightPilot Intelligence API",
            "gemini_api_configured": gemini_client.has_api_key(),
            "indexed_documents": len(vector_store.list_documents()),
            "total_chunks": len(vector_store.chunks),
            "database_connected": bool(db),
            "saved_reports_count": reports_count
        }

    # ---------------- Document Ingestion ----------------
    @router.post("/documents/upload", response_model=DocumentMetadata)
    async def upload_document(file: UploadFile = File(...)):
        filename = file.filename or "uploaded_doc"
        ext = os.path.splitext(filename)[1].lower()

        allowed_exts = [".pdf", ".txt", ".md", ".json", ".csv"]
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(allowed_exts)}"
            )

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)

        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            metadata, chunks = doc_loader.load_file(temp_path, filename)
            vector_store.add_document(metadata, chunks)
            return metadata
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    @router.get("/documents", response_model=List[DocumentMetadata])
    async def list_documents():
        return vector_store.list_documents()

    @router.delete("/documents/{doc_id}")
    async def delete_document(doc_id: str):
        if not vector_store.get_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        vector_store.remove_document(doc_id)
        return {"status": "success", "message": f"Document '{doc_id}' deleted from vector store"}

    # ---------------- Research Streaming & Execution ----------------
    @router.post("/research/stream")
    async def stream_research(request: ResearchRequest):
        """Streams real-time ReAct agent steps and final synthesis via Server-Sent Events (SSE)."""

        async def event_generator():
            try:
                async for event in orchestrator.execute_stream(request):
                    data_str = json.dumps(event)
                    yield f"data: {data_str}\n\n"
            except Exception as e:
                err_data = json.dumps({"event": "error", "data": {"message": str(e)}})
                yield f"data: {err_data}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # ---------------- Report History & Database Persistence ----------------
    @router.get("/reports")
    async def list_reports():
        """Lists past saved research reports from SQLite."""
        if not db:
            return []
        return db.list_reports()

    @router.get("/reports/{report_id}", response_model=ResearchReport)
    async def get_report(report_id: str):
        """Retrieves full research report by ID."""
        if not db:
            raise HTTPException(status_code=503, detail="Database not configured")
        report = db.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return report

    @router.delete("/reports/{report_id}")
    async def delete_report(report_id: str):
        """Deletes a report from SQLite."""
        if not db:
            raise HTTPException(status_code=503, detail="Database not configured")
        deleted = db.delete_report(report_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Report not found")
        return {"status": "success", "message": f"Report '{report_id}' deleted"}

    @router.get("/research/quick-topics")
    async def get_quick_topics():
        return [
            {
                "id": "agents",
                "title": "Autonomous Multi-Agent Systems in Production",
                "query": "What are the latest architectures and evaluation benchmarks for autonomous multi-agent systems and ReAct loops in 2025/2026?",
                "tag": "Agentic AI",
                "badge": "Popular"
            },
            {
                "id": "rag",
                "title": "Graph & Hybrid RAG vs Traditional Vector Search",
                "query": "Compare Hybrid RAG (Dense + Sparse retrieval) with Graph-based RAG in terms of latency, token cost, and factual grounding.",
                "tag": "RAG & Search",
                "badge": "Benchmark"
            },
            {
                "id": "quantum",
                "title": "Quantum Computing & Post-Quantum Cryptography",
                "query": "Analyze recent breakthroughs in quantum error correction and industry migration timelines for NIST Post-Quantum Cryptography.",
                "tag": "Emerging Tech",
                "badge": "Deep Dive"
            }
        ]

    @router.post("/export/markdown")
    async def export_markdown(report: ResearchReport):
        md_lines = [
            f"# Research Intelligence Report: {report.query}",
            f"*Generated by InsightPilot Autonomous Agent | Model: {report.metrics.get('model', 'Gemini 3.8 Flash')}*",
            "",
            "## Executive Summary",
            report.executive_summary,
            "",
            "## Key Analytical Findings",
        ]

        for finding in report.key_findings:
            md_lines.append(f"### {finding.heading}")
            md_lines.append(finding.content)
            if finding.key_takeaway:
                md_lines.append(f"> **Takeaway**: {finding.key_takeaway}")
            md_lines.append("")

        if report.comparative_matrix:
            md_lines.append("## Comparative Matrix")
            headers = list(report.comparative_matrix.keys())
            if headers:
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                rows = zip(*[report.comparative_matrix[h] for h in headers])
                for row in rows:
                    md_lines.append("| " + " | ".join(map(str, row)) + " |")
            md_lines.append("")

        if report.strategic_implications:
            md_lines.append("## Strategic Implications")
            for imp in report.strategic_implications:
                md_lines.append(f"- {imp}")
            md_lines.append("")

        if report.citations:
            md_lines.append("## Verified Citations & Grounded Sources")
            for cite in report.citations:
                source_tag = f"[{cite.source_type.upper()}]"
                url_str = f" - {cite.url_or_page}" if cite.url_or_page else ""
                md_lines.append(f"- **[{cite.id}]** {source_tag} *{cite.title}*{url_str}")
                md_lines.append(f"  > \"{cite.snippet}\"")

        content = "\n".join(md_lines)
        return PlainTextResponse(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=insightpilot_report_{report.id}.md"}
        )

    return router
