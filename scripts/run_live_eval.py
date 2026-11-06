"""Opt-in live Gemini answer evaluation; calls the real research workflow."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from backend.agent.gemini_client import GeminiClient
from backend.agent.orchestrator import ResearchOrchestrator
from backend.models import ResearchRequest
from backend.rag.vector_store import VectorStore


def score_report(case, report):
    if report is None:
        return {"answer_match": False, "citations_valid": False, "abstained": False}
    summary = report.executive_summary.casefold()
    abstained = report.metrics.get("mode") == "no_evidence"
    expected = case.get("expected_terms", [])
    match = abstained if case.get("abstain") else all(term.casefold() in summary for term in expected)
    ids = {citation.id for citation in report.citations}
    valid = bool(report.citations) and all(
        finding.citations and all(cid in ids and f"[{cid}]" in finding.content
                                  for cid in finding.citations)
        for finding in report.key_findings
    )
    if case.get("abstain") and abstained:
        valid = True
    return {"answer_match": match, "citations_valid": valid, "abstained": abstained}


async def evaluate(cases, client):
    store = VectorStore(db_path=":memory:")
    rows = []
    try:
        agent = ResearchOrchestrator(store, client)
        for case in cases:
            started = time.monotonic()
            report = None
            error = None
            try:
                async for event in agent.execute_stream(ResearchRequest(query=case["query"], use_documents=False)):
                    if event["event"] == "report":
                        from backend.models import ResearchReport
                        report = ResearchReport.model_validate(event["data"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            row = {"query": case["query"], "latency_seconds": round(time.monotonic() - started, 2),
                   "mode": report.metrics.get("mode") if report else None,
                   "model_rate_limited": bool(report and report.metrics.get("model_rate_limited")),
                   "citation_count": len(report.citations) if report else 0,
                   "error": error, **score_report(case, report)}
            rows.append(row)
            print(f"{len(rows)}/{len(cases)} {row['mode'] or row['error'] or 'no report'}: {case['query']}", flush=True)
    finally:
        store.close()
    total = len(rows)
    return {"model": client.model, "questions": total,
            "metrics": {"answer_match_rate": round(sum(r["answer_match"] for r in rows) / total, 3),
                        "citation_validity_rate": round(sum(r["citations_valid"] for r in rows) / total, 3),
                        "median_latency_seconds": sorted(r["latency_seconds"] for r in rows)[total // 2]},
            "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 6:
        parser.error("--limit must be between 1 and 6")
    if not os.environ.get("GEMINI_API_KEY"):
        parser.error("Set GEMINI_API_KEY to run live evaluation")
    cases = json.loads((ROOT / "evaluation" / "live_questions.json").read_text(encoding="utf-8"))[:args.limit]
    result = asyncio.run(evaluate(cases, GeminiClient()))
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
