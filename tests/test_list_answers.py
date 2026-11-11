"""List completeness and provenance regressions, without model or network calls."""

import asyncio
from pathlib import Path
import tempfile
import unittest

from backend.agent.gemini_client import GeminiClient
from backend.agent.list_evidence import extract_list_report
from backend.agent.orchestrator import ResearchOrchestrator
from backend.models import Citation, ResearchRequest
from backend.rag.document_loader import DocumentLoader
from backend.rag.vector_store import VectorStore


def source(text, index=0, doc="manual", cid=1):
    return Citation(id=cid, source_type="document", title=doc,
                    doc_id=doc, chunk_id=f"{doc}_{index}", chunk_index=index, snippet=text)


class ListAnswerTests(unittest.TestCase):
    def assert_support(self, report):
        by_id = {c.id: c for c in report.citations}
        for finding in report.key_findings:
            self.assertTrue(finding.evidence)
            self.assertEqual(set(finding.citations), {e.citation_id for e in finding.evidence})
            for evidence in finding.evidence:
                self.assertIn(evidence.quote, by_id[evidence.citation_id].snippet)

    def test_more_than_three_items_survive_without_repeating_query(self):
        text = "Equipment inspection steps:\n1. Isolate the power.\n2. Check the seals.\n3. Inspect cables.\n4. Record readings.\n5. Sign the log."
        report = extract_list_report("What are the equipment inspection steps?", [source(text)], "quick")
        self.assertEqual(len(report.key_findings), 5)
        self.assertIn("Sign the log", report.key_findings[-1].content)
        self.assert_support(report)

    def test_overlap_is_removed_and_item_crossing_chunk_boundary_has_both_sources(self):
        shared = "- Check the seal for damage before"
        a = source("Equipment inspection steps:\n- Isolate power.\n" + shared)
        b = source(shared + " restarting the pump.\n- Record readings.\n- Sign the log.", 1, cid=2)
        report = extract_list_report("List equipment inspection steps", [b, a], "balanced")
        self.assertEqual(len(report.key_findings), 4)
        self.assertIn("restarting the pump", report.key_findings[1].content)
        self.assert_support(report)
        # A continuation without overlap still needs support from both chunks.
        a.snippet = "Equipment inspection steps:\n- Isolate power.\n- Check the seal for"
        b.snippet = "damage before restarting the pump.\n- Record readings."
        report = extract_list_report("List equipment inspection steps", [a, b], "balanced")
        self.assertEqual(report.key_findings[1].citations, [1, 2])
        self.assert_support(report)

    def test_next_section_and_unrelated_list_are_excluded(self):
        text = ("# Equipment inspection\nInspection steps:\n- Isolate power.\n- Check cables.\n"
                "\n# Catering menu\n- Order sandwiches.\n- Buy coffee.")
        report = extract_list_report("List equipment inspection steps", [source(text)], "balanced")
        self.assertEqual(len(report.key_findings), 2)
        self.assertNotIn("coffee", str([f.model_dump() for f in report.key_findings]))

    def test_nonadjacent_chunks_and_other_documents_are_not_joined(self):
        a = source("Equipment inspection steps:\n- Isolate power.\n- Check cables.")
        for b in (source("- Buy coffee.\n- Order sandwiches.", 3, cid=2),
                  source("- Buy coffee.\n- Order sandwiches.", 1, doc="other", cid=2)):
            report = extract_list_report("List equipment inspection steps", [a, b], "balanced")
            self.assertEqual(len(report.key_findings), 2)
            self.assertEqual([c.id for c in report.citations], [1])

    def test_document_title_alone_does_not_authorize_unrelated_list(self):
        citation = source("Catering menu:\n- Order sandwiches.\n- Buy coffee.", doc="Equipment inspection")
        self.assertIsNone(extract_list_report("List equipment inspection steps", [citation], "balanced"))

    def test_plain_prose_boundary_does_not_reuse_previous_topic(self):
        citation = source("Equipment inspection steps:\n- Isolate power.\n- Check cables.\n\n"
                          "Catering menu:\n- Sandwiches.\n- Coffee.\n- Fruit.")
        report = extract_list_report("List equipment inspection steps", [citation], "balanced")
        self.assertEqual(len(report.key_findings), 2)
        self.assertNotIn("Coffee", str([f.content for f in report.key_findings]))

    def test_search_annotations_are_not_labeled_as_verbatim_source_lists(self):
        citation = source("Equipment inspection steps:\n- Isolate power.\n- Check cables.")
        citation.evidence_type = "grounded_claim"
        self.assertIsNone(extract_list_report("List equipment inspection steps", [citation], "balanced"))

    def test_truncated_item_is_not_presented_as_complete(self):
        citation = source("Equipment inspection steps:\n- Isolate power.\n- Check cables.\n- Record the...")
        report = extract_list_report("List equipment inspection steps", [citation], "balanced")
        self.assertEqual(len(report.key_findings), 2)
        self.assertIn("may not be the complete list", report.executive_summary)

    def test_repeated_items_are_deduplicated_and_long_lists_are_bounded(self):
        text = "Equipment inspection steps:\n" + "\n".join(f"- Inspect component {i}." for i in range(14))
        text += "\n- Inspect component 0."
        report = extract_list_report("List equipment inspection steps", [source(text)], "deep")
        self.assertEqual(len(report.key_findings), 12)
        self.assertTrue(report.metrics["list_limit_reached"])

    def test_definitions_and_comparisons_keep_existing_answer_path(self):
        citation = source("Equipment inspection steps:\n- Isolate power.\n- Check cables.")
        for query in ("What is equipment inspection?", "Compare equipment inspection steps and maintenance steps"):
            self.assertIsNone(extract_list_report(query, [citation], "balanced"))

    def test_bullet_fragments_without_periods_are_kept(self):
        citation = source("Required onboarding documents:\n• Passport\n• Signed agreement\n• Bank details")
        report = extract_list_report("List required onboarding documents", [citation], "balanced")
        self.assertEqual(len(report.key_findings), 3)
        self.assert_support(report)

    def test_real_chunking_retrieval_and_no_key_fallback(self):
        store = VectorStore(db_path=":memory:")
        self.addCleanup(store.close)
        loader = DocumentLoader(chunk_size=180)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.md"
            path.write_text("# Equipment inspection\nRequired steps:\n"
                "- Isolate: Disconnect electrical power before opening the enclosure.\n"
                "- Examine: Check every cable and seal for visible damage.\n"
                "- Measure: Record the pressure and temperature on the inspection form.\n"
                "- Restore: Close the enclosure and reconnect power after completing all checks.\n"
                "- Record: Sign the maintenance log and send a copy to the supervisor.\n\n"
                "# Catering menu\n- Sandwiches.\n- Coffee.", encoding="utf-8")
            meta, chunks = loader.load_file(str(path), path.name)
            store.add_document(meta, chunks)
        agent = ResearchOrchestrator(store, GeminiClient(api_key=""))
        async def run():
            return [e async for e in agent.execute_stream(ResearchRequest(
                query="What are the equipment inspection steps?", document_ids=[meta.id], search_web=False))]
        events = asyncio.run(run())
        report = next(e["data"] for e in events if e["event"] == "report")
        self.assertEqual(report["metrics"]["mode"], "extractive_list")
        self.assertEqual(len(report["key_findings"]), 5)
        self.assertNotIn("Coffee", str(report["key_findings"]))
        for finding in report["key_findings"]:
            for evidence in finding["evidence"]:
                citation = next(c for c in report["citations"] if c["id"] == evidence["citation_id"])
                original = store.get_chunk(citation["doc_id"], citation["chunk_id"])
                self.assertIn(evidence["quote"], original.content)

    def test_existing_sample_recovers_all_four_stages(self):
        from backend.agent.tools import DocumentRetrievalTool
        store = VectorStore(db_path=":memory:")
        self.addCleanup(store.close)
        path = Path(__file__).resolve().parents[1] / "sample_data" / "ai_agent_architectures.md"
        meta, chunks = DocumentLoader().load_file(str(path), path.name)
        store.add_document(meta, chunks)
        query = "What are the steps in a ReAct loop?"
        result = DocumentRetrievalTool(store).execute(query)
        citations = [source(chunk["content"], chunk["chunk_index"], chunk["doc_id"], i)
                     for i, chunk in enumerate(result["chunks"], 1)]
        report = GeminiClient(api_key="")._generate_fallback_report(query, [], citations, "balanced")
        self.assertEqual(report.metrics["mode"], "extractive_list")
        self.assertEqual(len(report.key_findings), 4)
        for finding, expected in zip(report.key_findings, ["Thought", "Action", "Observation", "Reflection"]):
            self.assertIn(expected, finding.content)
        self.assert_support(report)


if __name__ == "__main__":
    unittest.main()
