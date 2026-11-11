import asyncio
import tempfile
import unittest
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock, patch

from backend.agent.cache import TTLCache
from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.agent.page_reader import _Paragraphs
from backend.db.database import Database
from backend.models import Citation, ClaimEvidence, DocumentChunk, DocumentMetadata, Finding, ResearchReport, ResearchRequest
from backend.rag.vector_store import VectorStore


class FakeEmbeddings:
    model_name = "fixture-v1"
    def __init__(self):
        self.batches = 0
    def prepare(self):
        return True
    def passages(self, texts):
        self.batches += 1
        return [[1., 0.] if "reimbursement" in text else [0., 1.] for text in texts]
    def query(self, text):
        return [1., 0.]


def seed(store):
    for name, text in [("billing", "Request reimbursement from the finance department."),
                       ("library", "Readers may extend a book loan online.")]:
        meta = DocumentMetadata(id=name, filename=name + ".txt", file_type="TXT", size_bytes=60, chunks_count=1)
        store.add_document(meta, [DocumentChunk(id=name + "-chunk", doc_id=name,
            doc_name=meta.filename, chunk_index=0, content=text)])


class UpgradeTests(unittest.TestCase):
    def test_request_trims_query_and_rejects_blank_or_oversized_scope(self):
        from pydantic import ValidationError
        self.assertEqual(ResearchRequest(query="  research  ").query, "research")
        for payload in ({"query": "   "}, {"query": " a "},
                        {"query": "research", "document_ids": ["doc"] * 101}):
            with self.assertRaises(ValidationError):
                ResearchRequest(**payload)

    def test_api_scope_rejects_missing_documents_and_excludes_other_sources(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes import create_router
        store = VectorStore(db_path=":memory:")
        self.addCleanup(store.close)
        seed(store)
        app = FastAPI()
        app.include_router(create_router(store, GeminiClient(api_key="")), prefix="/api")
        with TestClient(app) as client:
            missing = client.post("/api/research/stream", json={"query": "Request reimbursement",
                "search_web": False, "document_ids": ["billing", "deleted"]})
            self.assertEqual(missing.status_code, 422)
            self.assertIn("no longer available", missing.json()["detail"])
            response = client.post("/api/research/stream", json={"query": "Request reimbursement",
                "search_web": False, "document_ids": ["billing"]})
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            report = next(e["data"] for e in events if e["event"] == "report")
            self.assertTrue(report["citations"])
            self.assertEqual({c["doc_id"] for c in report["citations"]}, {"billing"})
            self.assertEqual(report["metrics"]["source_scope"], {
                "documents": ["billing.txt"], "web_enabled": False, "selection": "selected"})
            other = client.post("/api/research/stream", json={"query": "Request reimbursement",
                "search_web": False, "document_ids": ["library"]})
            other_events = [json.loads(line[6:]) for line in other.text.splitlines() if line.startswith("data: ")]
            other_report = next(e["data"] for e in other_events if e["event"] == "report")
            self.assertEqual(other_report["citations"], [])

    def test_grounding_cooldown_skips_model_and_reuses_fallback_cache(self):
        store = VectorStore(db_path=":memory:")
        self.addCleanup(store.close)
        client = GeminiClient(api_key="")
        client.api_key = "fixture"
        client._client = SimpleNamespace()
        client._grounding_unavailable_until = time.monotonic() + 300
        client.stream_interaction = Mock(side_effect=AssertionError("cooldown bypassed"))
        agent = ResearchOrchestrator(store, client)
        agent.web_tool.execute_async = AsyncMock(return_value={"source": "live_web_engine", "results": [
            {"title": "Photosynthesis", "snippet": "Photosynthesis converts sunlight into chemical energy.", "url": "https://example.org/plants"}]})
        async def check():
            reports = []
            for _ in range(2):
                events = [event async for event in agent.execute_stream(ResearchRequest(query="What is photosynthesis?", use_documents=False))]
                reports.append(next(e["data"] for e in events if e["event"] == "report"))
            self.assertTrue(reports[0]["metrics"]["model_rate_limited"])
            self.assertTrue(reports[1]["metrics"]["web_cache_hit"])
            self.assertEqual(agent.web_tool.execute_async.call_count, 1)
        asyncio.run(check())
        client.stream_interaction.assert_not_called()

    def test_demo_followup_resolves_server_report_and_rejects_expired_context(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes import create_router
        store = VectorStore(db_path=":memory:")
        self.addCleanup(store.close)
        app = FastAPI()
        app.include_router(create_router(store, GeminiClient(api_key=""), mode="demo"), prefix="/api")
        with TestClient(app) as client:
            missing = client.post("/api/research/stream", json={"query": "Explain point one", "parent_report_id": "rep_missing"})
            self.assertEqual(missing.status_code, 404)
            response = client.post("/api/research/stream", json={"query": "Initial question", "search_web": False, "use_documents": False})
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            parent = next(e["data"] for e in events if e["event"] == "report")
            followup = client.post("/api/research/stream", json={"query": "Explain point one", "parent_report_id": parent["id"]})
            self.assertEqual(followup.status_code, 200)
            self.assertIn(parent["id"], followup.text)
            self.assertEqual(client.get("/api/reports").json(), [])

    def test_hybrid_recovers_paraphrase_and_respects_document_scope(self):
        model = FakeEmbeddings()
        store = VectorStore(db_path=":memory:", embedder=model)
        self.addCleanup(store.close)
        seed(store)
        results, mode = store.ranked_search("Can I get my money back?")
        self.assertEqual(mode, "hybrid")
        self.assertEqual(results[0][0].doc_id, "billing")
        store.ranked_search("Can I get my money back?")
        self.assertEqual(model.batches, 1)
        self.assertEqual(store.ranked_search("money back", filter_doc_ids=["library"])[0], [])
        self.assertEqual(store.ranked_search("money back", filter_doc_ids=[])[0], [])
        store.remove_document("billing")
        self.assertEqual(store.ranked_search("money back")[0], [])

    def test_embedding_vectors_survive_restart_and_model_change_rebuilds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "docs.db")
            first = VectorStore(db_path=path, embedder=FakeEmbeddings())
            seed(first)
            first.ranked_search("money back")
            first.close()
            model = FakeEmbeddings()
            second = VectorStore(db_path=path, embedder=model)
            try:
                self.assertTrue(second.ranked_search("money back")[0])
                self.assertEqual(model.batches, 0)
                model.model_name = "fixture-v2"
                second.ranked_search("money back")
                self.assertEqual(model.batches, 1)
            finally:
                second.close()

    def test_embedding_failure_preserves_keyword_search(self):
        model = FakeEmbeddings()
        model.passages = Mock(side_effect=RuntimeError("model unavailable"))
        store = VectorStore(db_path=":memory:", embedder=model)
        self.addCleanup(store.close)
        seed(store)
        results, mode = store.ranked_search("finance department")
        self.assertEqual(mode, "keyword_fallback")
        self.assertEqual(results[0][0].doc_id, "billing")

    def test_cache_expiry_eviction_and_copy_isolation(self):
        now = [0]
        cache = TTLCache(max_entries=2, ttl=10, clock=lambda: now[0])
        cache.put("a", {"results": [1]})
        cached = cache.get("a")
        cached["results"].append(2)
        self.assertEqual(cache.get("a")["results"], [1])
        cache.put("b", {})
        cache.put("c", {})
        self.assertIsNone(cache.get("a"))
        now[0] = 11
        self.assertIsNone(cache.get("c"))

    def test_web_cache_reuses_sources_and_preserves_retrieval_date(self):
        store = VectorStore(db_path=":memory:")
        self.addCleanup(store.close)
        client = GeminiClient(api_key="")
        agent = ResearchOrchestrator(store, client)
        agent.web_tool.execute_async = AsyncMock(return_value={"source": "live_web_engine", "results": [
            {"title": "Photosynthesis", "snippet": "Plants convert sunlight into chemical energy.", "url": "https://example.org/plants"}]})
        async def check():
            first = await agent._search_web(client, "photosynthesis")
            second = await agent._search_web(client, "photosynthesis")
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["retrieved_at"], second["retrieved_at"])
            self.assertEqual(agent.web_tool.execute_async.call_count, 1)
        asyncio.run(check())
        other = GeminiClient(api_key="")
        other.api_key = "different-credential"
        self.assertNotEqual(agent._web_key(client, "topic"), agent._web_key(other, "topic"))

    def test_transient_retry_is_bounded_and_rate_limit_is_not_retried(self):
        class ProviderError(Exception):
            status_code = 503
        client = GeminiClient(api_key="")
        create = Mock(side_effect=[ProviderError(), "ok"])
        client._client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        with patch("backend.agent.gemini_client.time.sleep"):
            self.assertEqual(client._create_with_retry(input="question"), "ok")
        self.assertEqual(create.call_count, 2)
        ProviderError.status_code = 429
        create.reset_mock(side_effect=True)
        create.side_effect = ProviderError()
        with self.assertRaises(ProviderError):
            client._create_with_retry(input="question")
        self.assertEqual(create.call_count, 1)

    def test_stream_failure_after_output_is_not_replayed(self):
        class ProviderError(Exception):
            status_code = 503
        def partial():
            yield "first event"
            raise ProviderError()
        client = GeminiClient(api_key="")
        client.api_key = "fixture"
        create = Mock(return_value=partial())
        client._client = SimpleNamespace(interactions=SimpleNamespace(create=create))
        stream = client.stream_interaction("query", [])
        self.assertEqual(next(stream), "first event")
        with self.assertRaises(ProviderError):
            next(stream)
        self.assertEqual(create.call_count, 1)
        self.assertFalse(client.grounding_available())

    def test_followup_uses_only_referenced_report_evidence(self):
        store = VectorStore(db_path=":memory:")
        self.addCleanup(store.close)
        agent = ResearchOrchestrator(store, GeminiClient(api_key=""))
        parent = ResearchReport(query="Research context", executive_summary="Two findings.",
            key_findings=[Finding(heading="First", content="First fact [1]", citations=[1]),
                          Finding(heading="Second", content="Second fact [2]", citations=[2])],
            citations=[Citation(id=1, title="First source", source_type="document", snippet="The first source contains an unrelated statement."),
                       Citation(id=2, title="Second source", source_type="document", snippet="The second source contains the supporting evidence.")])
        agent.report_cache.put(parent.id, parent)
        agent.doc_tool.execute = Mock(side_effect=AssertionError("new retrieval"))
        agent.web_tool.execute_async = AsyncMock(side_effect=AssertionError("new web call"))
        async def check():
            events = [event async for event in agent.execute_stream(ResearchRequest(
                query="Explain finding two", parent_report_id=parent.id))]
            report = next(e["data"] for e in events if e["event"] == "report")
            self.assertEqual(report["metrics"]["parent_report_id"], parent.id)
            self.assertEqual([c["id"] for c in report["citations"]], [2])
            self.assertEqual(report["key_findings"][0]["evidence"][0]["quote"], parent.citations[1].snippet)
        asyncio.run(check())
        agent.doc_tool.execute.assert_not_called()
        agent.web_tool.execute_async.assert_not_called()

    def test_claim_evidence_survives_database_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Database(str(Path(directory) / "reports.db"))
            report = ResearchReport(query="test", executive_summary="A claim [1]", key_findings=[
                Finding(heading="Claim", content="A claim [1]", citations=[1], evidence=[
                    ClaimEvidence(citation_id=1, quote="Supporting passage", verification="exact_quote")])],
                citations=[Citation(id=1, title="Source", source_type="web", snippet="Supporting passage", published_at="2026-01-02", retrieved_at=123.)])
            db.save_report(report)
            restored = db.get_report(report.id)
            self.assertEqual(restored.key_findings[0].evidence, report.key_findings[0].evidence)
            self.assertEqual(restored.citations[0].published_at, "2026-01-02")
            self.assertEqual(db.list_reports()[0]["model"], "Model not recorded")
            report.metrics["mode"] = "extractive_fallback"
            db.save_report(report)
            self.assertEqual(db.list_reports()[0]["model"], "Source excerpts")
            report.metrics["model"] = "recorded-model"
            db.save_report(report)
            self.assertEqual(db.list_reports()[0]["model"], "recorded-model")

    def test_publication_date_is_read_only_from_valid_metadata(self):
        parser = _Paragraphs()
        parser.feed('<meta property="article:published_time" content="2026-03-04T10:00:00Z">')
        self.assertEqual(parser.published_at, "2026-03-04")
        unknown = _Paragraphs()
        unknown.feed('<meta name="date" content="not a date">')
        self.assertIsNone(unknown.published_at)


if __name__ == "__main__":
    unittest.main()
