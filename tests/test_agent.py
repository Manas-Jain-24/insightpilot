import unittest
import asyncio
import os
from backend.rag.document_loader import DocumentLoader
from backend.rag.vector_store import VectorStore
from backend.agent.tools import DocumentRetrievalTool, WebSearchTool, KnowledgeGraphBuilderTool
from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.models import ResearchRequest


class TestInsightPilot(unittest.TestCase):

    def setUp(self):
        self.loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
        self.vector_store = VectorStore()
        self.gemini_client = GeminiClient()
        self.orchestrator = ResearchOrchestrator(self.vector_store, self.gemini_client)

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
        async def run_test():
            tool = WebSearchTool()
            res = await tool.execute_async("Quantum computing", num_results=2)
            self.assertIn("results", res)
            self.assertGreater(len(res["results"]), 0)

        asyncio.run(run_test())

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
            async for event in self.orchestrator.execute_stream(req):
                events.append(event["event"])

            self.assertIn("plan", events)
            self.assertIn("step_start", events)
            self.assertIn("thought", events)
            self.assertIn("action", events)
            self.assertIn("observation", events)
            self.assertIn("report", events)
            self.assertIn("done", events)

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
