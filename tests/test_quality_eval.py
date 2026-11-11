import asyncio
from copy import deepcopy
import unittest

from backend.eval.quality import compare, evaluate, load_dataset, score_report, summarize
from backend.models import Citation, ClaimEvidence, DocumentChunk, Finding, ResearchReport


class QualityEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {"expected_behavior": "answer", "expected_sources": ["manual"],
                     "facets": [{"name": "duration", "alternatives": ["twenty minutes"]}]}
        self.chunk = DocumentChunk(id="chunk", doc_id="manual", doc_name="manual.txt", chunk_index=0,
                                   content="The verification link expires after twenty minutes.")
        self.chunks = {("manual", "chunk"): self.chunk}
        self.report = ResearchReport(query="When does the verification link expire?",
            executive_summary="The verification link expires after twenty minutes. [1]",
            key_findings=[Finding(heading="Expiry", content="The verification link expires after twenty minutes. [1]",
                citations=[1], evidence=[ClaimEvidence(citation_id=1, quote=self.chunk.content, verification="exact_quote")])],
            citations=[Citation(id=1, source_type="document", title="manual", snippet=self.chunk.content,
                                doc_id="manual", chunk_id="chunk")], metrics={"mode": "extractive_fallback"})

    def test_valid_answer_requires_facets_sources_and_exact_support(self):
        self.assertTrue(score_report(self.case, self.report, self.chunks)["passed"])
        self.report.key_findings[0].evidence[0].quote = "The link never expires."
        result = score_report(self.case, self.report, self.chunks)
        self.assertFalse(result["passed"])
        self.assertIn("invalid_evidence", result["failure_reasons"])

    def test_citation_snippet_cannot_supply_missing_answer_facets(self):
        self.report.executive_summary = "Related passages are available."
        self.report.key_findings[0].content = "Inspect the source for details. [1]"
        result = score_report(self.case, self.report, self.chunks)
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_facets"], ["duration"])

    def test_summary_only_and_empty_findings_do_not_pass_evidence_check(self):
        self.report.key_findings = []
        self.assertFalse(score_report(self.case, self.report, self.chunks)["evidence_valid"])

    def test_unknown_original_duplicate_ids_and_wrong_inline_link_fail(self):
        for mutation in (lambda r: setattr(r.citations[0], "chunk_id", "missing"),
                         lambda r: r.citations.append(r.citations[0]),
                         lambda r: setattr(r.key_findings[0], "content", "twenty minutes [2]")):
            report = self.report.model_copy(deep=True)
            mutation(report)
            self.assertFalse(score_report(self.case, report, self.chunks)["evidence_valid"])

    def test_abstention_must_be_clean_and_expected(self):
        case = {"expected_behavior": "abstain", "expected_sources": [], "facets": []}
        report = ResearchReport(query="Missing context", executive_summary="No evidence.", metrics={"mode": "no_evidence"})
        self.assertTrue(score_report(case, report, self.chunks)["passed"])
        self.assertFalse(score_report(self.case, report, self.chunks)["passed"])
        report.key_findings = self.report.key_findings
        report.citations = self.report.citations
        self.assertFalse(score_report(case, report, self.chunks)["passed"])

    def test_dataset_has_eight_categories_and_reserved_validation_cases(self):
        data = load_dataset()
        self.assertEqual(len(data["cases"]), 32)
        self.assertEqual(len({c["category"] for c in data["cases"]}), 8)
        self.assertEqual(sum(c["split"] == "validation" for c in data["cases"]), 8)

    def test_comparison_requires_same_dataset_and_reports_regressions(self):
        baseline = {"dataset_sha256": "fixture", "cases": [
            {"id": "one", "requested_retrieval": "keyword", "passed": True},
            {"id": "two", "requested_retrieval": "keyword", "passed": False}]}
        current = deepcopy(baseline)
        current["cases"][0]["passed"] = False
        current["cases"][1]["passed"] = True
        result = compare(current, baseline)
        self.assertEqual(result["regressed"], ["keyword:one"])
        self.assertEqual(result["improved"], ["keyword:two"])
        current["dataset_sha256"] = "changed"
        with self.assertRaises(ValueError):
            compare(current, baseline)

    def test_real_workflow_runs_without_a_model_and_records_actual_retrieval(self):
        data = load_dataset()
        data["cases"] = data["cases"][:1]
        result = asyncio.run(evaluate(data))
        self.assertEqual(result["metrics"]["keyword"]["cases"], 1)
        self.assertTrue(result["cases"][0]["passed"])
        self.assertEqual(result["cases"][0]["actual_retrieval"], "keyword")
        self.assertIn("idempotency", result["cases"][0]["retrieved_sources"])

    def test_empty_rates_are_unknown_instead_of_perfect_scores(self):
        result = summarize([])
        self.assertIsNone(result["safe_abstention_rate"])
        self.assertIsNone(result["answer_facet_coverage"])
        self.assertIsNone(result["case_pass_rate"])


if __name__ == "__main__":
    unittest.main()
