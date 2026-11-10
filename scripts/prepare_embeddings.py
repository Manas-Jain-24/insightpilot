"""Download the local embedding model once; runtime search never downloads it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.rag.embeddings import LocalEmbeddings

if __name__ == "__main__":
    model = LocalEmbeddings()
    if not model.prepare(download=True):
        raise SystemExit(f"Could not prepare embeddings: {model.error}")
    print(f"Ready: {model.model_name} ({len(model.query('test'))} dimensions)")
