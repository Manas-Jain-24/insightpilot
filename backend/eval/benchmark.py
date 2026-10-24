"""Repeatable, local retrieval benchmark over the bundled sample documents."""

import json
import time
from pathlib import Path

from backend.rag.document_loader import DocumentLoader
from backend.rag.vector_store import VectorStore


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluation" / "questions.json"
SAMPLES = ROOT / "sample_data"


def run_benchmark() -> dict:
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    store = VectorStore(db_path=":memory:")
    loader = DocumentLoader()
    try:
        for path in sorted(SAMPLES.iterdir()):
            if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
                continue
            metadata, chunks = loader.load_file(str(path), path.name)
            store.add_document(metadata, chunks)

        results = []
        for index, case in enumerate(cases, start=1):
            began = time.perf_counter()
            matches = store.similarity_search(case["query"], top_k=3)
            latency_ms = round((time.perf_counter() - began) * 1000, 2)
            if case.get("unanswerable"):
                success = not matches
                rank = None
                source_match = False
            else:
                phrase = case["gold_phrase"].casefold()
                rank = next((i for i, (chunk, _) in enumerate(matches, start=1)
                             if chunk.doc_name == case["gold_file"] and phrase in chunk.content.casefold()), None)
                success = rank is not None
                source_match = bool(matches and matches[0][0].doc_name == case["gold_file"])
            results.append({
                "id": index, "query": case["query"],
                "type": "abstention" if case.get("unanswerable") else "retrieval",
                "expected_source": case.get("gold_file"),
                "top_source": matches[0][0].doc_name if matches else None,
                "rank": rank, "success": success,
                "source_match_at_1": source_match,
                "latency_ms": latency_ms
            })

        positives = [row for row in results if row["type"] == "retrieval"]
        negatives = [row for row in results if row["type"] == "abstention"]
        rate = lambda numerator, denominator: round(numerator / denominator, 3) if denominator else 0.0
        return {
            "dataset": "Bundled sample corpus v1",
            "questions": len(results),
            "documents": len(store.list_documents()),
            "chunks": store.count_chunks(),
            "metrics": {
                "hit_at_1": rate(sum(row["rank"] == 1 for row in positives), len(positives)),
                "hit_at_3": rate(sum(row["success"] for row in positives), len(positives)),
                "mrr_at_3": round(sum(1 / row["rank"] for row in positives if row["rank"]) / len(positives), 3) if positives else 0.0,
                "source_at_1": rate(sum(row["source_match_at_1"] for row in positives), len(positives)),
                "abstention_rate": rate(sum(row["success"] for row in negatives), len(negatives)),
                "mean_latency_ms": round(sum(row["latency_ms"] for row in results) / len(results), 2) if results else 0.0
            },
            "cases": results
        }
    finally:
        store.close()
