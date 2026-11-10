"""Local embeddings. Model downloads happen only during explicit preparation."""
import os
import threading
from pathlib import Path


class LocalEmbeddings:
    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self, cache_dir=None):
        root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(cache_dir or os.environ.get("EMBEDDING_CACHE_DIR", root / "data" / "models"))
        self._model = None
        self._lock = threading.RLock()
        self.error = None

    def prepare(self, download=False):
        with self._lock:
            if self._model is not None:
                return True
            if not download and not (self.cache_dir / "ready").exists():
                self.error = "Local model not prepared"
                return False
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(model_name=self.model_name,
                    cache_dir=str(self.cache_dir), threads=2, local_files_only=not download)
                if download:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    (self.cache_dir / "ready").write_text(self.model_name, encoding="utf-8")
                self.error = None
                return True
            except Exception as exc:
                self.error = type(exc).__name__
                return False

    def passages(self, texts):
        with self._lock:
            return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def query(self, text):
        with self._lock:
            return next(iter(self._model.query_embed(text))).tolist()
