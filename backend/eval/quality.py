"""Reproducible, network-free checks of the production document research workflow."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.models import DocumentMetadata, ResearchReport, ResearchRequest
from backend.rag.document_loader import DocumentLoader
from backend.rag.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation/research_quality.json"
SNAPSHOT = ROOT / "evaluation/reports/quality-latest.json"


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_dataset(path=DATASET):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    document_ids = {d["id"] for d in data["documents"]}
    ids = [c["id"] for c in data["cases"]]
    if len(ids) != len(set(ids)) or len(document_ids) != len(data["documents"]):
        raise ValueError("Dataset IDs must be unique")
    for case in data["cases"]:
        if case["split"] not in {"development", "validation"}:
            raise ValueError("Unknown dataset split")
        if case["expected_behavior"] not in {"answer", "abstain"}:
            raise ValueError("Unknown expected behavior")
        if not set(case["expected_sources"]) <= document_ids:
            raise ValueError("Unknown expected source")
        if case["expected_behavior"] == "answer" and (
            not case["expected_sources"] or not case["facets"] or
            any(not facet["alternatives"] or any(not text.strip() for text in facet["alternatives"])
                for facet in case["facets"])
        ):
            raise ValueError("Answer cases need sources and nonempty answer facets")
    return data


def score_report(case, report, chunks):
    """Score answer text, never citation snippets; copied support is not entailment."""
    reasons = []
    abstained = bool(report and report.metrics.get("mode") in {"no_evidence", "clarification_needed"})
    answer = "\n".join([report.executive_summary] + [f.content for f in report.key_findings]) if report else ""
    text = normalized(answer)
    facets = [{"name": facet["name"], "matched": any(normalized(option) in text
               for option in facet["alternatives"])} for facet in case["facets"]]
    missing = [facet["name"] for facet in facets if not facet["matched"]]
    cited = {c.doc_id for c in report.citations} if report else set()
    expected = set(case["expected_sources"])
    evidence_valid = bool(report)
    if report:
        by_id = {c.id: c for c in report.citations}
        evidence_valid = len(by_id) == len(report.citations)
        for citation in report.citations:
            original = chunks.get((citation.doc_id, citation.chunk_id))
            snippet = citation.snippet.removesuffix("...").strip()
            evidence_valid &= bool(original and snippet and normalized(snippet) in normalized(original.content))
        for finding in report.key_findings:
            evidence_valid &= bool(finding.citations and finding.evidence)
            evidence_valid &= set(finding.citations) == {e.citation_id for e in finding.evidence}
            for cid in finding.citations:
                evidence_valid &= cid in by_id and f"[{cid}]" in finding.content
            for evidence in finding.evidence:
                source = by_id.get(evidence.citation_id)
                evidence_valid &= bool(source and evidence.quote.strip() and
                                       normalized(evidence.quote) in normalized(source.snippet))
        if abstained:
            evidence_valid &= not report.key_findings and not report.citations
        else:
            evidence_valid &= bool(report.key_findings and report.citations)
    if report is None:
        reasons.append("no_report")
    elif case["expected_behavior"] == "abstain":
        if not abstained:
            reasons.append("answered_without_sufficient_context")
    else:
        if abstained:
            reasons.append("unexpected_abstention")
        if missing:
            reasons.append("missing_answer_facets")
        if not expected <= cited:
            reasons.append("missing_expected_sources")
    if not evidence_valid:
        reasons.append("invalid_evidence")
    return {"passed": not reasons, "failure_reasons": reasons, "abstained": abstained,
            "evidence_valid": bool(evidence_valid), "facets": facets, "missing_facets": missing,
            "source_recall": len(expected & cited) / len(expected) if expected else None,
            "answer": answer, "cited_sources": sorted(cited)}


def summarize(rows):
    def rate(numerator, denominator):
        return round(numerator / denominator, 4) if denominator else None
    positive = [r for r in rows if r["expected_behavior"] == "answer"]
    negative = [r for r in rows if r["expected_behavior"] == "abstain"]
    return {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
            "case_pass_rate": rate(sum(r["passed"] for r in rows), len(rows)),
            "answer_facet_coverage": rate(sum(f["matched"] for r in positive for f in r["facets"]),
                                          sum(len(r["facets"]) for r in positive)),
            "safe_abstention_rate": rate(sum(r["abstained"] for r in negative), len(negative)),
            "unexpected_abstentions": sum(r["abstained"] for r in positive),
            "evidence_validity_rate": rate(sum(r["evidence_valid"] for r in rows), len(rows)),
            "pipeline_errors": sum(bool(r["error"]) for r in rows)}


def code_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "backend").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


async def evaluate(data, split="all", modes=("keyword",), embedder=None, progress=None):
    if split not in {"all", "development", "validation"}:
        raise ValueError("Unknown split")
    if not modes or not set(modes) <= {"keyword", "hybrid"}:
        raise ValueError("Unknown retrieval mode")
    if "hybrid" in modes and (not embedder or not embedder.prepare()):
        raise RuntimeError("Hybrid evaluation requires a prepared local embedding model")
    cases = [c for c in data["cases"] if split == "all" or c["split"] == split]
    rows = []
    for mode in modes:
        store = VectorStore(db_path=":memory:", embedder=embedder if mode == "hybrid" else None)
        try:
            for doc in data["documents"]:
                name = doc["title"] + ".txt"
                chunks = DocumentLoader()._chunk_pages(doc["id"], name, [(1, doc["text"])])
                store.add_document(DocumentMetadata(id=doc["id"], filename=name, file_type="TXT",
                    size_bytes=len(doc["text"].encode()), chunks_count=len(chunks)), chunks)
            originals = {(chunk.doc_id, chunk.id): chunk for chunk in store.chunks}
            agent = ResearchOrchestrator(store, GeminiClient(api_key=""))
            for case in cases:
                started = time.perf_counter()
                report, error = None, None
                retrieved = set()
                try:
                    async for event in agent.execute_stream(ResearchRequest(query=case["query"], search_web=False)):
                        if event["event"] == "citation_added" and event["data"].get("doc_id"):
                            retrieved.add(event["data"]["doc_id"])
                        elif event["event"] == "report":
                            report = ResearchReport.model_validate(event["data"])
                        elif event["event"] == "error":
                            error = event["data"]["message"]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                row = {"id": case["id"], "query": case["query"], "category": case["category"],
                       "split": case["split"], "expected_behavior": case["expected_behavior"],
                       "expected_sources": case["expected_sources"], "requested_retrieval": mode,
                       "actual_retrieval": report.metrics.get("retrieval_mode") if report else None,
                       "answer_mode": report.metrics.get("mode") if report else None,
                       "retrieved_sources": sorted(retrieved), "error": error,
                       "retrieval_source_recall": len(set(case["expected_sources"]) & retrieved) / len(case["expected_sources"]) if case["expected_sources"] else None,
                       "latency_seconds": round(time.perf_counter()-started, 3),
                       **score_report(case, report, originals)}
                if error:
                    row["passed"] = False
                    row["failure_reasons"].append("pipeline_error")
                rows.append(row)
                if progress:
                    progress(row)
        finally:
            store.close()
    return {"dataset": data["version"], "dataset_sha256": hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()).hexdigest(),
            "code_sha256": code_fingerprint(), "created_at": datetime.now(timezone.utc).isoformat(),
            "split": split, "embedding_model": embedder.model_name if embedder and "hybrid" in modes else None,
            "limitations": "Synthetic, self-authored cases. Concept matching and quote attribution do not prove factual correctness or logical entailment. No Gemini or web calls. Latency includes orchestration delays and cold local embedding work.",
            "metrics": {mode: summarize([r for r in rows if r["requested_retrieval"] == mode]) for mode in modes},
            "by_category": {mode: {category: summarize([r for r in rows if r["requested_retrieval"] == mode and r["category"] == category])
                for category in sorted({r["category"] for r in rows})} for mode in modes}, "cases": rows}


def compare(current, baseline):
    if current["dataset_sha256"] != baseline["dataset_sha256"]:
        raise ValueError("Cannot compare runs with different datasets")
    old = {(r["requested_retrieval"], r["id"]): r for r in baseline["cases"]}
    new = {(r["requested_retrieval"], r["id"]): r for r in current["cases"]}
    common = sorted(old.keys() & new.keys())
    if not common:
        raise ValueError("No matching cases to compare")
    return {"compared_cases": len(common), "new_cases": len(new.keys()-old.keys()),
            "improved": [f"{mode}:{cid}" for mode, cid in common if new[mode, cid]["passed"] and not old[mode, cid]["passed"]],
            "regressed": [f"{mode}:{cid}" for mode, cid in common if old[mode, cid]["passed"] and not new[mode, cid]["passed"]]}


def load_snapshot():
    if not SNAPSHOT.exists():
        return None
    result = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    result["recorded_run"] = True
    return result
