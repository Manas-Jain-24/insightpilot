"""Compare keyword and local hybrid retrieval on an independent paraphrase corpus."""
import json
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.eval.semantic import evaluate

if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2))
