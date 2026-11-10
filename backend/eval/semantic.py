"""Compare keyword and local hybrid retrieval on an independent paraphrase corpus."""
import json
from pathlib import Path
from time import perf_counter
from backend.models import DocumentMetadata, DocumentChunk
from backend.rag.vector_store import VectorStore
from backend.rag.embeddings import LocalEmbeddings


def evaluate(embedder=None):
    data = json.loads((Path(__file__).resolve().parents[2] / "evaluation/semantic_cases.json").read_text())
    embedder = embedder or LocalEmbeddings()
    if not embedder.prepare():
        raise RuntimeError("Prepare the local embedding model with scripts/prepare_embeddings.py first")
    store = VectorStore(db_path=":memory:")
    try:
        for doc in data["documents"]:
            meta = DocumentMetadata(id=doc["id"], filename=doc["id"] + ".txt", file_type="TXT", size_bytes=len(doc["text"]), chunks_count=1)
            store.add_document(meta, [DocumentChunk(id=doc["id"], doc_id=doc["id"], doc_name=meta.filename, chunk_index=0, content=doc["text"])])
        rows = []
        for case in data["cases"]:
            row = {**case}
            for mode in ("keyword", "hybrid"):
                store.embedder = embedder if mode == "hybrid" else None
                started = perf_counter()
                results, actual = store.ranked_search(case["query"], top_k=3)
                ids = [chunk.doc_id for chunk, _ in results]
                row[mode] = {"mode": actual, "top": ids[0] if ids else None,
                    "hit_at_3": case["expected"] in ids if case["expected"] else not ids,
                    "latency_ms": round((perf_counter() - started) * 1000, 2)}
            rows.append(row)
        summary = {}
        for mode in ("keyword", "hybrid"):
            positives = [r for r in rows if r["expected"]]
            negatives = [r for r in rows if not r["expected"]]
            summary[mode] = {"hit_at_1": sum(r[mode]["top"] == r["expected"] for r in positives) / len(positives),
                "hit_at_3": sum(r[mode]["hit_at_3"] for r in positives) / len(positives),
                "abstention": sum(r[mode]["top"] is None for r in negatives) / len(negatives)}
        return {"model": embedder.model_name, "metrics": summary, "cases": rows}
    finally:
        store.close()
