import os
import json
import shutil
import tempfile
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, PlainTextResponse, Response
from typing import List, Dict, Any, Optional
from backend.models import (
    ResearchRequest, DocumentMetadata, ResearchReport
)
from backend.rag.document_loader import DocumentLoader
from backend.rag.vector_store import VectorStore
from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.db.database import Database
from backend.eval.benchmark import run_benchmark
from backend.rag.pdf_preview import render_pdf_page
from backend.api.rate_limit import DemoRateLimiter


def create_router(
    vector_store: VectorStore,
    gemini_client: GeminiClient,
    db: Optional[Database] = None,
    mode: str = "local"
) -> APIRouter:
    router = APIRouter()
    doc_loader = DocumentLoader()
    orchestrator = ResearchOrchestrator(vector_store, gemini_client, db=db)
    limiter = DemoRateLimiter()

    def require_local():
        if mode == "demo":
            raise HTTPException(status_code=403, detail="Unavailable in the public demo")

    def demo_limit(request: Request, action: str, per_client: int, global_limit: int):
        if mode != "demo":
            return
        client = request.client.host if request.client else "unknown"
        if not limiter.allow_many([
            (f"{action}:global", global_limit),
            (f"{action}:{client}", per_client),
        ]):
            raise HTTPException(status_code=429, detail="Demo request limit reached. Try again in a minute.",
                                headers={"Retry-After": "60"})

    @router.get("/health")
    def health_check():
        reports_count = len(db.list_reports()) if db else 0
        return {
            "status": "healthy",
            "service": "InsightPilot Intelligence API",
            "gemini_api_configured": gemini_client.has_api_key(),
            "indexed_documents": len(vector_store.list_documents()),
            "total_chunks": vector_store.count_chunks(),
            "database_connected": bool(db),
            "saved_reports_count": reports_count,
            "mode": mode
        }

    @router.get("/config")
    def public_config():
        return {"mode": mode, "uploads_enabled": mode != "demo",
                "history_enabled": bool(db), "visitor_api_keys_enabled": mode != "demo"}

    @router.get("/evaluation")
    def evaluation(request: Request):
        demo_limit(request, "evaluation", 12, 60)
        return run_benchmark()

    # ---------------- Document Ingestion ----------------
    @router.post("/documents/upload", response_model=DocumentMetadata)
    async def upload_document(file: UploadFile = File(...)):
        require_local()
        filename = os.path.basename((file.filename or "uploaded_doc").replace("\\", "/"))
        ext = os.path.splitext(filename)[1].lower()

        allowed_exts = [".pdf", ".txt", ".md", ".json", ".csv"]
        if ext not in allowed_exts:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(allowed_exts)}"
            )

        max_upload_bytes = 10 * 1024 * 1024
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{ext}")

        try:
            total = 0
            with open(temp_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="File exceeds 10 MB upload limit")
                    buffer.write(chunk)

            metadata, chunks = doc_loader.load_file(temp_path, filename)
            if not chunks:
                raise HTTPException(status_code=422, detail="Document contains no extractable text")
            source_bytes = None
            if ext == ".pdf":
                with open(temp_path, "rb") as source_file:
                    source_bytes = source_file.read()
            vector_store.add_document(metadata, chunks, source_bytes=source_bytes)
            return metadata
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to process document")
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    @router.get("/documents", response_model=List[DocumentMetadata])
    def list_documents():
        return vector_store.list_documents()

    @router.get("/documents/{doc_id}/file")
    def get_document_file(doc_id: str):
        pdf = vector_store.get_pdf_bytes(doc_id)
        if pdf is None:
            raise HTTPException(status_code=404, detail="PDF source unavailable; re-upload this document")
        return Response(pdf, media_type="application/pdf", headers={
            "Content-Disposition": "inline; filename=source.pdf",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff"
        })

    @router.get("/documents/{doc_id}/pages/{page_number}/preview")
    def preview_pdf_page(doc_id: str, page_number: int, quote: str = Query(default="", max_length=240)):
        pdf = vector_store.get_pdf_bytes(doc_id)
        if pdf is None:
            raise HTTPException(status_code=404, detail="PDF source unavailable")
        try:
            png = render_pdf_page(pdf, page_number, quote)
        except IndexError:
            raise HTTPException(status_code=404, detail="Page not found")
        except Exception:
            raise HTTPException(status_code=422, detail="Could not render PDF page")
        return Response(png, media_type="image/png", headers={"Cache-Control": "private, no-store"})

    @router.get("/documents/{doc_id}/chunks/{chunk_id}")
    def get_document_chunk(doc_id: str, chunk_id: str):
        chunk = vector_store.get_chunk(doc_id, chunk_id)
        if chunk is None:
            raise HTTPException(status_code=404, detail="Source excerpt unavailable")
        return {"doc_id": doc_id, "chunk_id": chunk.id,
                "page_number": chunk.page_number, "content": chunk.content,
                "pdf_available": vector_store.has_pdf(doc_id)}

    @router.delete("/documents/{doc_id}")
    def delete_document(doc_id: str):
        require_local()
        if not vector_store.get_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        vector_store.remove_document(doc_id)
        return {"status": "success", "message": f"Document '{doc_id}' deleted from vector store"}

    # ---------------- Research Streaming & Execution ----------------
    @router.post("/research/stream")
    async def stream_research(request: ResearchRequest, http_request: Request):
        """Streams real-time ReAct agent steps and final synthesis via Server-Sent Events (SSE)."""
        if mode == "demo" and request.api_key:
            raise HTTPException(status_code=403, detail="Visitor API keys are disabled in the public demo")
        demo_limit(http_request, "research", 6, 30)

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
    def list_reports():
        """Lists past saved research reports from SQLite."""
        if not db:
            return []
        return db.list_reports()

    @router.get("/reports/{report_id}", response_model=ResearchReport)
    def get_report(report_id: str):
        """Retrieves full research report by ID."""
        if not db:
            raise HTTPException(status_code=503, detail="Database not configured")
        report = db.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        return report

    @router.delete("/reports/{report_id}")
    def delete_report(report_id: str):
        """Deletes a report from SQLite."""
        require_local()
        if not db:
            raise HTTPException(status_code=503, detail="Database not configured")
        deleted = db.delete_report(report_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Report not found")
        return {"status": "success", "message": f"Report '{report_id}' deleted"}

    @router.get("/research/quick-topics")
    def get_quick_topics():
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
        require_local()
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
