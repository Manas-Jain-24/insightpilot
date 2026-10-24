import unittest
import asyncio
import os
import httpx
import tempfile
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch
from backend.rag.document_loader import DocumentLoader
from backend.rag.vector_store import VectorStore
from backend.agent.tools import DocumentRetrievalTool, WebSearchTool, KnowledgeGraphBuilderTool
from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.models import ResearchRequest


class TestInsightPilot(unittest.TestCase):

    def setUp(self):
        self.loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
        self.vector_store = VectorStore(db_path=":memory:")
        self.gemini_client = GeminiClient()
        self.orchestrator = ResearchOrchestrator(self.vector_store, self.gemini_client)

    def tearDown(self):
        self.vector_store.close()

    def test_document_loader_and_chunking(self):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "quantum_computing_2025.txt")
        self.assertTrue(os.path.exists(sample_path), f"Sample file missing at {sample_path}")

        meta, chunks = self.loader.load_file(sample_path, "quantum_computing_2025.txt")
        self.assertGreater(len(chunks), 0)
        self.assertEqual(meta.file_type, "TXT")
        self.assertIn("Quantum", chunks[0].content)

    def test_vector_store_similarity_search(self):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "quantum_computing_2025.txt")
        meta, chunks = self.loader.load_file(sample_path, "quantum_computing_2025.txt")
        self.vector_store.add_document(meta, chunks)

        results = self.vector_store.similarity_search("NIST post-quantum cryptography algorithms", top_k=2)
        self.assertGreater(len(results), 0)
        best_chunk, score = results[0]
        self.assertIn("NIST", best_chunk.content)
        self.assertGreater(score, 0.0)

    def test_web_search_tool_async(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
            "Heading": "Quantum computing", "AbstractText": "Quantum computers use qubits.",
            "AbstractURL": "https://example.org/quantum", "RelatedTopics": []
        }))
        original_client = httpx.AsyncClient
        async def run_test():
            tool = WebSearchTool()
            with patch("backend.agent.tools.httpx.AsyncClient",
                       side_effect=lambda *a, **kw: original_client(transport=transport, **kw)):
                res = await tool.execute_async("Quantum computing", num_results=1)
            self.assertIn("results", res)
            self.assertEqual(res["results"][0]["url"], "https://example.org/quantum")

        asyncio.run(run_test())

    def test_web_failure_does_not_invent_sources(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(503))
        original_client = httpx.AsyncClient
        async def run_test():
            with patch("backend.agent.tools.httpx.AsyncClient",
                       side_effect=lambda *a, **kw: original_client(transport=transport, **kw)):
                result = await WebSearchTool().execute_async("unknown topic")
            self.assertEqual(result["results"], [])
        asyncio.run(run_test())

    def test_no_evidence_report_has_no_findings_or_citations(self):
        report = self.gemini_client.synthesize_final_report("unknown topic", [], [])
        self.assertEqual(report.key_findings, [])
        self.assertEqual(report.citations, [])
        self.assertEqual(report.metrics["mode"], "no_evidence")

    def test_gemini_accepts_only_quotes_from_matching_citation(self):
        from backend.models import Citation

        source_one = Citation(id=1, source_type="document", title="Document A",
                              snippet="Quantum computers use qubits to represent information and perform calculations.")
        source_two = Citation(id=2, source_type="web", title="Source B",
                              snippet="Classical computers use bits to represent information in binary form.")
        response = {
            "executive_summary": "Invented result: 99 percent faster.",
            "comparative_matrix": {"Speed": ["99 percent faster"]},
            "selected_evidence": [
                {"citation_id": 1, "quote": "Quantum computers use qubits to represent information"},
                {"citation_id": 2, "quote": "Quantum computers use qubits to represent information"},
                {"citation_id": 1, "quote": "Quantum computers are 99 percent faster"},
            ]
        }
        client = GeminiClient()
        client.api_key = "test-key"
        client._client = SimpleNamespace(interactions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(response))
        ))
        report = client.synthesize_final_report("quantum computing", [], [source_one, source_two])
        self.assertEqual(report.metrics["mode"], "verified_quotes")
        self.assertEqual(report.metrics["rejected_quotes"], 2)
        self.assertEqual(len(report.key_findings), 1)
        self.assertEqual(report.key_findings[0].citations, [1])
        self.assertIn("Quantum computers use qubits", report.key_findings[0].content)
        self.assertNotIn("99 percent", report.model_dump_json())
        self.assertIsNone(report.comparative_matrix)

    def test_gemini_without_valid_quotes_uses_extractive_fallback(self):
        from backend.models import Citation

        citation = Citation(id=1, source_type="document", title="Document A",
                            snippet="A source excerpt describing a measured research result.")
        client = GeminiClient()
        client.api_key = "test-key"
        client._client = SimpleNamespace(interactions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text=json.dumps({
                "selected_evidence": [{"citation_id": 1, "quote": "A fabricated conclusion with no supporting text."}]
            }))
        ))
        report = client.synthesize_final_report("research result", [], [citation])
        self.assertEqual(report.metrics["mode"], "extractive_fallback")
        self.assertEqual(report.key_findings[0].citations, [1])

    def test_research_controls_disable_both_tools(self):
        async def run_test():
            request = ResearchRequest(query="unknown topic", use_documents=False, search_web=False)
            with patch.object(self.orchestrator.doc_tool, "execute") as doc_search, \
                 patch.object(self.orchestrator.web_tool, "execute_async") as web_search:
                events = [item async for item in self.orchestrator.execute_stream(request)]
            doc_search.assert_not_called()
            web_search.assert_not_called()
            self.assertEqual(next(e["data"] for e in events if e["event"] == "report")["metrics"]["mode"], "no_evidence")
        asyncio.run(run_test())

    def test_request_key_does_not_change_shared_client(self):
        async def run_test():
            request = ResearchRequest(query="unknown topic", api_key="request-key",
                                      use_documents=False, search_web=False)
            request_client = GeminiClient()
            with patch("backend.agent.orchestrator.GeminiClient", return_value=request_client) as factory:
                events = [item async for item in self.orchestrator.execute_stream(request)]
            factory.assert_called_once_with(api_key="request-key")
            self.assertEqual(self.gemini_client.api_key, os.environ.get("GEMINI_API_KEY", "").strip())
            self.assertTrue(any(item["event"] == "report" for item in events))
        asyncio.run(run_test())

    def test_disabled_web_is_not_called(self):
        async def run_test():
            request = ResearchRequest(query="ReAct hallucinations", search_web=False)
            with patch.object(self.orchestrator.web_tool, "execute_async") as search:
                events = [item async for item in self.orchestrator.execute_stream(request)]
            search.assert_not_called()
            self.assertTrue(any(item["event"] == "report" for item in events))
        asyncio.run(run_test())

    def test_document_index_persists(self):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "quantum_computing_2025.txt")
        meta, chunks = self.loader.load_file(sample_path, "quantum_computing_2025.txt")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "documents.db")
            first = VectorStore(db_path=path)
            first.add_document(meta, chunks)
            first.close()
            second = VectorStore(db_path=path)
            self.assertIsNotNone(second.get_document(meta.id))
            self.assertTrue(second.similarity_search("NIST cryptography"))
            second.remove_document(meta.id)
            self.assertEqual(second.count_chunks(), 0)
            second.close()

    def test_pdf_bytes_persist_with_document(self):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "quantum_computing_2025.txt")
        meta, chunks = self.loader.load_file(sample_path, "quantum_computing_2025.txt")
        pdf_meta = meta.model_copy(update={"file_type": "PDF", "filename": "source.pdf"})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "documents.db")
            first = VectorStore(db_path=path)
            first.add_document(pdf_meta, chunks, source_bytes=b"%PDF-persisted-test")
            first.close()
            second = VectorStore(db_path=path)
            self.assertEqual(second.get_pdf_bytes(pdf_meta.id), b"%PDF-persisted-test")
            second.remove_document(pdf_meta.id)
            self.assertIsNone(second.get_pdf_bytes(pdf_meta.id))
            second.close()

    def test_upload_rejects_large_and_invalid_pdf(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes import create_router

        app = FastAPI()
        app.include_router(create_router(self.vector_store, self.gemini_client), prefix="/api")
        client = TestClient(app)
        large = client.post("/api/documents/upload", files={"file": ("big.txt", b"a" * (10 * 1024 * 1024 + 1))})
        self.assertEqual(large.status_code, 413)
        invalid = client.post("/api/documents/upload", files={"file": ("bad.pdf", b"not a pdf")})
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(self.vector_store.count_chunks(), 0)

    def test_upload_normalizes_filename(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes import create_router

        app = FastAPI()
        app.include_router(create_router(self.vector_store, self.gemini_client), prefix="/api")
        response = TestClient(app).post("/api/documents/upload", files={"file": ("../outside.txt", b"Useful evidence")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["filename"], "outside.txt")

    def test_pdf_source_and_chunk_are_available_until_deleted(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes import create_router
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

        writer = PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        font_ref = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})
        })
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 50 700 Td (The cited PDF contains a verifiable passage.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(stream)
        pdf_buffer = BytesIO()
        writer.write(pdf_buffer)
        pdf = pdf_buffer.getvalue()

        app = FastAPI()
        app.include_router(create_router(self.vector_store, self.gemini_client), prefix="/api")
        client = TestClient(app)
        upload = client.post("/api/documents/upload", files={"file": ("proof.pdf", pdf, "application/pdf")})
        self.assertEqual(upload.status_code, 200)
        doc_id = upload.json()["id"]
        source = client.get(f"/api/documents/{doc_id}/file")
        self.assertEqual(source.status_code, 200)
        self.assertEqual(source.content, pdf)
        self.assertEqual(source.headers["content-type"], "application/pdf")
        plain_page = client.get(f"/api/documents/{doc_id}/pages/1/preview")
        highlighted_page = client.get(
            f"/api/documents/{doc_id}/pages/1/preview",
            params={"quote": "verifiable passage"}
        )
        self.assertEqual(highlighted_page.status_code, 200)
        self.assertEqual(highlighted_page.headers["content-type"], "image/png")
        self.assertTrue(highlighted_page.content.startswith(b"\x89PNG"))
        self.assertNotEqual(hash(plain_page.content), hash(highlighted_page.content))
        chunk = self.vector_store.similarity_search("verifiable passage")[0][0]
        excerpt = client.get(f"/api/documents/{doc_id}/chunks/{chunk.id}")
        self.assertEqual(excerpt.status_code, 200)
        self.assertTrue(excerpt.json()["pdf_available"])
        self.assertIn("verifiable passage", excerpt.json()["content"])
        self.assertEqual(excerpt.json()["page_number"], 1)
        client.delete(f"/api/documents/{doc_id}")
        self.assertEqual(client.get(f"/api/documents/{doc_id}/file").status_code, 404)
        self.assertEqual(client.get(f"/api/documents/{doc_id}/pages/1/preview").status_code, 404)
        self.assertEqual(client.get(f"/api/documents/{doc_id}/chunks/{chunk.id}").status_code, 404)

    def test_evaluation_dashboard_has_repeatable_cases(self):
        from backend.eval.benchmark import run_benchmark

        first = run_benchmark()
        second = run_benchmark()
        self.assertGreaterEqual(first["questions"], 30)
        self.assertEqual(first["questions"], len(first["cases"]))
        self.assertEqual(first["metrics"]["hit_at_3"], second["metrics"]["hit_at_3"])
        self.assertEqual(first["metrics"]["abstention_rate"], second["metrics"]["abstention_rate"])
        self.assertTrue(any(case["type"] == "abstention" for case in first["cases"]))

    def test_knowledge_graph_builder(self):
        tool = KnowledgeGraphBuilderTool()
        kg = tool.execute("Agentic AI", ["ReAct Loop", "Vector RAG", "Evaluation"])
        self.assertEqual(len(kg.nodes), 4)
        self.assertEqual(len(kg.edges), 3)

    def test_orchestrator_execution_stream(self):
        async def run_stream():
            # Ingest sample
            sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data", "ai_agent_architectures.md")
            meta, chunks = self.loader.load_file(sample_path, "ai_agent_architectures.md")
            self.vector_store.add_document(meta, chunks)

            req = ResearchRequest(
                query="How does ReAct reduce hallucinations?",
                depth="quick"
            )

            events = []
            reports = []
            async for event in self.orchestrator.execute_stream(req):
                events.append(event["event"])
                if event["event"] == "report":
                    reports.append(event["data"])

            self.assertIn("plan", events)
            self.assertIn("step_start", events)
            self.assertIn("thought", events)
            self.assertIn("action", events)
            self.assertIn("observation", events)
            self.assertIn("report", events)
            self.assertIn("done", events)
            document_citations = [c for c in reports[0]["citations"] if c["source_type"] == "document"]
            self.assertTrue(document_citations)
            self.assertEqual(document_citations[0]["doc_id"], meta.id)
            self.assertTrue(document_citations[0]["chunk_id"])

        asyncio.run(run_stream())


    def test_database_persistence(self):
        import tempfile
        from backend.db.database import Database
        from backend.models import ResearchReport, Finding, Citation

        temp_db_file = os.path.join(tempfile.gettempdir(), "test_insightpilot.db")
        if os.path.exists(temp_db_file):
            os.remove(temp_db_file)

        test_db = Database(db_path=temp_db_file)
        sample_report = ResearchReport(
            id="test_rep_1",
            query="Test Quantum Query",
            depth="quick",
            executive_summary="This is a test summary for SQLite persistence.",
            key_findings=[Finding(heading="Test Heading", content="Test Content", citations=[1])],
            citations=[Citation(id=1, source_type="document", title="Sample Doc", snippet="Sample snippet")]
        )

        # Save
        test_db.save_report(sample_report)

        # List
        reports = test_db.list_reports()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["id"], "test_rep_1")
        self.assertEqual(reports[0]["query"], "Test Quantum Query")

        # Get
        retrieved = test_db.get_report("test_rep_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.query, "Test Quantum Query")
        self.assertEqual(len(retrieved.key_findings), 1)

        # Delete
        deleted = test_db.delete_report("test_rep_1")
        self.assertTrue(deleted)
        self.assertEqual(len(test_db.list_reports()), 0)

        # Clean up
        if os.path.exists(temp_db_file):
            os.remove(temp_db_file)


if __name__ == "__main__":
    unittest.main()
