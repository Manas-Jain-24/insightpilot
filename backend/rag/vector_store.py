import math
import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from backend.models import DocumentChunk, DocumentMetadata


class VectorStore:
    """Hybrid Vector Store with semantic similarity search and fallback embedding vectorizer."""

    def __init__(self, gemini_client=None):
        self.gemini_client = gemini_client
        self.documents: Dict[str, DocumentMetadata] = {}
        self.chunks: List[DocumentChunk] = []
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._tfidf_matrix: Optional[np.ndarray] = None

    def add_document(self, metadata: DocumentMetadata, chunks: List[DocumentChunk]) -> None:
        self.documents[metadata.id] = metadata
        self.chunks.extend(chunks)
        self._rebuild_index()

    def remove_document(self, doc_id: str) -> None:
        if doc_id in self.documents:
            del self.documents[doc_id]
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._rebuild_index()

    def list_documents(self) -> List[DocumentMetadata]:
        return list(self.documents.values())

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        return self.documents.get(doc_id)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text)
        return tokens

    def _rebuild_index(self) -> None:
        if not self.chunks:
            self._tfidf_matrix = None
            self._vocab = {}
            self._idf = {}
            return

        doc_count = len(self.chunks)
        vocab_counter: Dict[str, int] = {}
        doc_tokens_list: List[List[str]] = []

        for chunk in self.chunks:
            tokens = set(self._tokenize(chunk.content))
            doc_tokens_list.append(self._tokenize(chunk.content))
            for tok in tokens:
                vocab_counter[tok] = vocab_counter.get(tok, 0) + 1

        # Keep top informative tokens
        sorted_tokens = sorted(vocab_counter.items(), key=lambda x: x[1], reverse=True)[:5000]
        self._vocab = {tok: idx for idx, (tok, _) in enumerate(sorted_tokens)}

        self._idf = {}
        for tok, df in vocab_counter.items():
            if tok in self._vocab:
                self._idf[tok] = math.log((1 + doc_count) / (1 + df)) + 1.0

        # Build tf-idf matrix
        matrix = np.zeros((doc_count, len(self._vocab)), dtype=np.float32)
        for i, tokens in enumerate(doc_tokens_list):
            tf: Dict[str, int] = {}
            for t in tokens:
                if t in self._vocab:
                    tf[t] = tf.get(t, 0) + 1

            for t, count in tf.items():
                col = self._vocab[t]
                tf_norm = 1.0 + math.log(count) if count > 0 else 0
                matrix[i, col] = tf_norm * self._idf.get(t, 1.0)

            # L2 normalize
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] /= norm

        self._tfidf_matrix = matrix

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        filter_doc_ids: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Searches for chunks most relevant to query using cosine similarity."""
        if not self.chunks or self._tfidf_matrix is None:
            return []

        # Try Gemini API embedding if configured
        if self.gemini_client and self.gemini_client.has_api_key():
            try:
                results = self._gemini_vector_search(query, top_k, filter_doc_ids)
                if results:
                    return results
            except Exception:
                pass  # Fall back to local hybrid vector engine

        # Local semantic vector engine
        q_tokens = self._tokenize(query)
        q_vec = np.zeros(len(self._vocab), dtype=np.float32)

        tf: Dict[str, int] = {}
        for t in q_tokens:
            if t in self._vocab:
                tf[t] = tf.get(t, 0) + 1

        for t, count in tf.items():
            col = self._vocab[t]
            tf_norm = 1.0 + math.log(count) if count > 0 else 0
            q_vec[col] = tf_norm * self._idf.get(t, 1.0)

        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        # Compute cosine similarity
        scores = np.dot(self._tfidf_matrix, q_vec)

        # Apply term overlap boost
        q_set = set(q_tokens)
        candidate_scores = []
        for idx, chunk in enumerate(self.chunks):
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue

            score = float(scores[idx])
            # Boost score if exact keywords match
            chunk_tokens = set(self._tokenize(chunk.content))
            overlap = len(q_set.intersection(chunk_tokens))
            boosted_score = score * 0.7 + (min(overlap / max(1, len(q_set)), 1.0)) * 0.3

            if boosted_score > 0.05:
                candidate_scores.append((chunk, boosted_score))

        # Sort descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return candidate_scores[:top_k]

    def _gemini_vector_search(
        self,
        query: str,
        top_k: int,
        filter_doc_ids: Optional[List[str]]
    ) -> List[Tuple[DocumentChunk, float]]:
        # Compute embeddings for query and compare against stored embeddings
        q_emb = self.gemini_client.embed_text(query)
        if not q_emb:
            return []

        q_vec = np.array(q_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        results = []
        for chunk in self.chunks:
            if filter_doc_ids and chunk.doc_id not in filter_doc_ids:
                continue

            if chunk.embedding is None:
                # Lazy embedding
                chunk.embedding = self.gemini_client.embed_text(chunk.content[:1000])

            if chunk.embedding:
                c_vec = np.array(chunk.embedding, dtype=np.float32)
                c_norm = np.linalg.norm(c_vec)
                if c_norm > 0:
                    c_vec /= c_norm
                sim = float(np.dot(q_vec, c_vec))
                results.append((chunk, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
