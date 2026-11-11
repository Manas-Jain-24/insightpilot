"""Run the production document workflow against the broader offline quality suite."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.eval.quality import compare, evaluate, load_dataset
from backend.rag.embeddings import LocalEmbeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["development", "validation", "all"], default="development")
    parser.add_argument("--retrieval", choices=["keyword", "hybrid", "both"], default="keyword")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    if args.fail_on_regression and not args.compare:
        parser.error("--fail-on-regression requires --compare")
    modes = ("keyword", "hybrid") if args.retrieval == "both" else (args.retrieval,)
    embedder = LocalEmbeddings() if "hybrid" in modes else None
    def progress(row):
        print(f"{row['requested_retrieval']} {row['id']}: {'PASS' if row['passed'] else ', '.join(row['failure_reasons'])}", flush=True)
    result = asyncio.run(evaluate(load_dataset(), args.split, modes, embedder, progress))
    if args.compare:
        result["comparison"] = compare(result, json.loads(args.compare.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"metrics": result["metrics"], "comparison": result.get("comparison")}, indent=2))
    if args.fail_on_regression and (result["comparison"]["regressed"] or any(row["error"] for row in result["cases"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
