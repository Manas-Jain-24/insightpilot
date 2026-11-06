"""Offline tests for answer grounding and multi-subject retrieval."""

import asyncio
import json
import unittest
import httpx
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.agent.query_planner import research_queries
from backend.agent.page_reader import fetch_page_excerpt, _public_https_url
from backend.eval.benchmark import run_answer_benchmark
from backend.models import Citation, ResearchRequest
from backend.rag.vector_store import VectorStore
from scripts.run_live_eval import score_report


class ResearchQualityTests(unittest.TestCase):
    def test_streamed_grounded_answer_uses_combined_tools(self):
        body = "A heckler's veto occurs when a hostile audience prevents a speaker from being heard."
        annotation = SimpleNamespace(type="url_citation", url="https://example.org/speech",
                                     title="Speech source", start_index=0, end_index=len(body))
        block = SimpleNamespace(type="text", text=body, annotations=[annotation])
        interaction = SimpleNamespace(status="completed", steps=[SimpleNamespace(type="model_output", content=[block])])
        calls = []
        client = GeminiClient(api_key="")
        client.api_key = "test"
        client._client = SimpleNamespace()
        def stream_interaction(**kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(event_type="interaction.completed", interaction=interaction)])
        client.stream_interaction = stream_interaction
        store = VectorStore(db_path=":memory:")
        try:
            agent = ResearchOrchestrator(store, client)
            async def collect():
                return [event async for event in agent.execute_stream(
                    ResearchRequest(query="What is a heckler's veto?")
                )]
            events = asyncio.run(collect())
            self.assertEqual(len(calls), 1)
            self.assertEqual({tool["type"] for tool in calls[0]["tools"]}, {"google_search", "function"})
            report = next(event["data"] for event in events if event["event"] == "report")
            self.assertEqual(report["metrics"]["mode"], "search_grounded_stream")
            self.assertIn("hostile audience", report["executive_summary"])
            self.assertEqual(report["citations"][0]["url_or_page"], "https://example.org/speech")
        finally:
            store.close()

    def test_claim_validation_rejects_added_number_and_negation(self):
        source = Citation(id=1, source_type="web", title="Source",
                          snippet="The intervention improved reading scores for students in the trial.")
        claims = [
            {"citation_id": 1, "claim": "Reading scores improved by 99 percent.",
             "support": source.snippet},
            {"citation_id": 1, "claim": "The intervention did not improve reading scores.",
             "support": source.snippet},
            {"citation_id": 1, "claim": "The intervention improved reading scores in the trial.",
             "support": source.snippet},
        ]
        findings, rejected = GeminiClient._verify_grounded_claims(claims, [source])
        self.assertEqual(rejected, 2)
        self.assertEqual(len(findings), 1)

    def test_live_eval_scoring_checks_citation_wiring(self):
        from backend.models import Finding, ResearchReport
        report = ResearchReport(query="test", executive_summary="A hostile audience stops a speaker.",
                                key_findings=[Finding(heading="Source", content="Claim [2]", citations=[2])],
                                citations=[Citation(id=1, source_type="web", title="Source", snippet="evidence")])
        result = score_report({"expected_terms": ["hostile audience", "speaker"]}, report)
        self.assertTrue(result["answer_match"])
        self.assertFalse(result["citations_valid"])

    def test_comparison_queries_are_decomposed(self):
        self.assertEqual(
            research_queries("Compare Python and JavaScript for web development"),
            ["Python web development", "JavaScript web development"],
        )
        self.assertEqual(research_queries("Python vs JavaScript"), ["Python", "JavaScript"])
        self.assertEqual(research_queries("What is photosynthesis?"), ["What is photosynthesis?"])

    def test_google_grounding_keeps_only_annotated_https_claims(self):
        body = "Python runs on the server. This sentence has no source."
        annotation = SimpleNamespace(type="url_citation", url="https://docs.python.org/3/",
                                     title="Python docs", start_index=0, end_index=26)
        unsafe = SimpleNamespace(type="url_citation", url="http://127.0.0.1/private",
                                 title="Private", start_index=0, end_index=26)
        block = SimpleNamespace(type="text", text=body, annotations=[annotation, unsafe])
        response = SimpleNamespace(steps=[SimpleNamespace(type="model_output", content=[block])])
        client = GeminiClient(api_key="")
        client.api_key = "test"
        client._client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kwargs: response))
        result = client.search_web_grounded("Python web development")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["evidence_type"], "grounded_claim")
        self.assertEqual(result["results"][0]["url"], "https://docs.python.org/3/")

    def test_page_reader_blocks_private_urls_and_extracts_relevant_passage(self):
        self.assertFalse(_public_https_url("https://127.0.0.1/private"))
        self.assertFalse(_public_https_url("http://example.org/page"))
        markup = ("<html><script>Ignore this text.</script><p>Unrelated navigation text that "
                  "describes nothing about the requested subject.</p>"
                  "<p>Photosynthesis converts sunlight into chemical energy in green plants.</p></html>")
        transport = httpx.MockTransport(lambda request: httpx.Response(
            200, text=markup, headers={"content-type": "text/html; charset=utf-8"}
        ))
        original_client = httpx.AsyncClient
        async def run_test():
            with patch("backend.agent.page_reader._public_https_url", return_value=True), \
                 patch("backend.agent.page_reader.httpx.AsyncClient",
                       side_effect=lambda *a, **kw: original_client(transport=transport, **kw)):
                return await fetch_page_excerpt("https://example.org/photosynthesis", "photosynthesis")
        passage = asyncio.run(run_test())
        self.assertIn("converts sunlight", passage)
        self.assertNotIn("Unrelated navigation", passage)

    def test_grounding_rate_limit_opens_short_circuit(self):
        class RateLimitError(Exception):
            pass
        calls = []
        def fail(**kwargs):
            calls.append(kwargs)
            raise RateLimitError()
        client = GeminiClient(api_key="")
        client.api_key = "test"
        client._client = SimpleNamespace(interactions=SimpleNamespace(create=fail))
        with self.assertRaises(RateLimitError):
            client.search_web_grounded("photosynthesis")
        self.assertEqual(client.search_web_grounded("photosynthesis")["results"], [])
        self.assertEqual(len(calls), 1)

    def test_comparison_searches_both_subjects(self):
        store = VectorStore(db_path=":memory:")
        try:
            orchestrator = ResearchOrchestrator(store, GeminiClient(api_key=""))
            async def result(query, num_results=3):
                return {"source": "test", "results": [{"title": query,
                        "snippet": f"{query} supports an independently retrieved fact.",
                        "url": f"https://example.org/{query.split()[0].lower()}"}]}
            orchestrator.web_tool.execute_async = AsyncMock(side_effect=result)
            found = asyncio.run(orchestrator._search_web(
                GeminiClient(api_key=""), "Compare Python and JavaScript for web development"
            ))
            self.assertEqual(orchestrator.web_tool.execute_async.call_count, 2)
            self.assertEqual(found["comparison_groups"], [[0], [1]])
            self.assertEqual(len(found["results"]), 2)
        finally:
            store.close()

    def test_grounded_search_failure_uses_existing_web_sources(self):
        store = VectorStore(db_path=":memory:")
        try:
            client = GeminiClient(api_key="")
            client.api_key = "test"
            client._client = SimpleNamespace()
            def fail(query):
                raise RuntimeError("grounding unavailable")
            client.search_web_grounded = fail
            orchestrator = ResearchOrchestrator(store, client)
            orchestrator.web_tool.execute_async = AsyncMock(return_value={
                "source": "live_web_engine", "results": [{
                    "title": "Photosynthesis",
                    "snippet": "Photosynthesis converts sunlight into chemical energy in plants.",
                    "url": "https://en.wikipedia.org/wiki/Photosynthesis",
                }]
            })
            found = asyncio.run(orchestrator._search_web(client, "photosynthesis"))
            self.assertEqual(found["source"], "live_web_engine")
            self.assertEqual(found["results"][0]["title"], "Photosynthesis")
        finally:
            store.close()

    def test_generated_answer_requires_real_support_and_both_comparison_sides(self):
        sources = [
            Citation(id=1, source_type="web", title="Python",
                     snippet="Python is commonly used for server-side web development."),
            Citation(id=2, source_type="web", title="JavaScript",
                     snippet="JavaScript is commonly used for browser-side web development."),
        ]
        valid = {"claims": [
            {"claim": "Python is used on the server side.", "citation_id": 1,
             "support": "Python is commonly used for server-side web development."},
            {"claim": "JavaScript is used in the browser.", "citation_id": 2,
             "support": "JavaScript is commonly used for browser-side web development."},
            {"claim": "It is 100 times faster.", "citation_id": 1,
             "support": "This evidence was invented."},
        ]}
        client = GeminiClient(api_key="")
        client.api_key = "test"
        client._client = SimpleNamespace(interactions=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(valid))
        ))
        observations = [{"comparison_groups": [[1], [2]]}]
        report = client.synthesize_final_report("Compare Python and JavaScript", observations, sources)
        self.assertEqual(report.metrics["mode"], "grounded_synthesis")
        self.assertEqual(report.metrics["rejected_claims"], 1)
        self.assertIn("[1]", report.executive_summary)
        self.assertIn("[2]", report.executive_summary)
        self.assertNotIn("100 times faster", report.model_dump_json())

        valid["claims"] = valid["claims"][:1]
        incomplete = client.synthesize_final_report("Compare Python and JavaScript", observations, sources)
        self.assertEqual(incomplete.metrics["mode"], "no_evidence")

    def test_comparison_pipeline_returns_two_cited_subjects(self):
        store = VectorStore(db_path=":memory:")
        try:
            client = GeminiClient(api_key="")
            client.api_key = "test"
            client.create_interaction = lambda **kwargs: None
            client.plan_research = GeminiClient(api_key="").plan_research
            source_text = {
                "Python": "Python is commonly used for server-side web development.",
                "JavaScript": "JavaScript is commonly used for browser-side web development.",
            }
            def search(query):
                subject = query.split()[0]
                return {"source": "google_search_grounding", "results": [{
                    "title": subject, "snippet": source_text[subject],
                    "url": f"https://example.org/{subject.lower()}",
                    "evidence_type": "grounded_claim",
                }]}
            client.search_web_grounded = search
            response = {"claims": [
                {"claim": "Python is used for server-side web development.", "citation_id": 1,
                 "support": source_text["Python"]},
                {"claim": "JavaScript is used for browser-side web development.", "citation_id": 2,
                 "support": source_text["JavaScript"]},
            ]}
            client._client = SimpleNamespace(interactions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(output_text=json.dumps(response))
            ))
            orchestrator = ResearchOrchestrator(store, client)
            async def run_test():
                with patch("backend.agent.orchestrator.fetch_page_excerpt", new=AsyncMock(return_value="")):
                    events = [event async for event in orchestrator.execute_stream(ResearchRequest(
                        query="Compare Python and JavaScript for web development",
                        use_documents=False, search_web=True,
                    ))]
                return next(event["data"] for event in events if event["event"] == "report")
            report = asyncio.run(run_test())
            self.assertEqual(report["metrics"]["mode"], "grounded_synthesis")
            self.assertEqual([c["title"] for c in report["citations"]], ["Python", "JavaScript"])
            self.assertIn("[1]", report["executive_summary"])
            self.assertIn("[2]", report["executive_summary"])
        finally:
            store.close()

    def test_offline_answer_benchmark_checks_answer_citation_and_abstention(self):
        result = run_answer_benchmark()
        self.assertGreaterEqual(result["questions"], 6)
        self.assertEqual(result["metrics"]["answer_match_rate"], 1.0)
        self.assertEqual(result["metrics"]["citation_validity_rate"], 1.0)
        self.assertEqual(result["metrics"]["answer_abstention_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
