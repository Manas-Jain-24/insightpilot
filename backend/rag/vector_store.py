"""Persistent document storage and indexed lexical retrieval."""

import os
import re
import sqlite3
import threading
import json
import math
from typing import List, Optional, Tuple

from backend.models import DocumentChunk, DocumentMetadata


class VectorStore:
    """SQLite FTS5 backed retrieval. The legacy name is kept for API compatibility."""

    def __init__(self, gemini_client=None, db_path: Optional[str] = None, embedder=None):
        self.gemini_client = gemini_client
        self.embedder = embedder
        data_dir = os.environ.get("DATA_DIR") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
        )
        self.db_path = db_path or os.path.join(data_dir, "documents.db")
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, payload TEXT NOT NULL)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS document_files (doc_id TEXT PRIMARY KEY, content BLOB NOT NULL)")
            self._connection.execute("CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id)")
            self._connection.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(chunk_id UNINDEXED, content)")
            self._connection.execute("CREATE TABLE IF NOT EXISTS chunk_vectors (chunk_id TEXT, model TEXT, vector TEXT NOT NULL, PRIMARY KEY(chunk_id, model))")

    @property
    def chunks(self) -> List[DocumentChunk]:
        with self._lock:
            rows = self._connection.execute("SELECT payload FROM chunks").fetchall()
        return [DocumentChunk.model_validate_json(row[0]) for row in rows]

    def count_chunks(self) -> int:
        with self._lock:
            return self._connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def add_document(self, metadata: DocumentMetadata, chunks: List[DocumentChunk],
                     source_bytes: Optional[bytes] = None) -> None:
        with self._lock, self._connection:
            self._delete_document(metadata.id)
            self._connection.execute("INSERT INTO documents VALUES (?, ?)", (metadata.id, metadata.model_dump_json()))
            if source_bytes is not None and metadata.file_type == "PDF":
                self._connection.execute("INSERT INTO document_files VALUES (?, ?)",
                                         (metadata.id, source_bytes))
            for chunk in chunks:
                self._connection.execute("INSERT INTO chunks VALUES (?, ?, ?)",
                                         (chunk.id, metadata.id, chunk.model_dump_json()))
                self._connection.execute("INSERT INTO chunk_search (chunk_id, content) VALUES (?, ?)",
                                         (chunk.id, chunk.content))

    def _delete_document(self, doc_id: str) -> None:
        self._connection.execute("DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)", (doc_id,))
        self._connection.execute("DELETE FROM document_files WHERE doc_id = ?", (doc_id,))
        self._connection.execute("DELETE FROM chunk_search WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)", (doc_id,))
        self._connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._connection.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def remove_document(self, doc_id: str) -> None:
        with self._lock, self._connection:
            self._delete_document(doc_id)

    def list_documents(self) -> List[DocumentMetadata]:
        with self._lock:
            rows = self._connection.execute("SELECT metadata FROM documents").fetchall()
        return [DocumentMetadata.model_validate_json(row[0]) for row in rows]

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        with self._lock:
            row = self._connection.execute("SELECT metadata FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return DocumentMetadata.model_validate_json(row[0]) if row else None

    def get_chunk(self, doc_id: str, chunk_id: str) -> Optional[DocumentChunk]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM chunks WHERE doc_id = ? AND id = ?", (doc_id, chunk_id)
            ).fetchone()
        return DocumentChunk.model_validate_json(row[0]) if row else None

    def following_chunks(self, doc_id: str, chunk_index: int, limit: int = 2) -> List[DocumentChunk]:
        """Return nearby source chunks when an answer spans a chunk boundary."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM chunks WHERE doc_id = ? "
                "AND CAST(json_extract(payload, '$.chunk_index') AS INTEGER) > ? "
                "AND CAST(json_extract(payload, '$.chunk_index') AS INTEGER) <= ? "
                "ORDER BY CAST(json_extract(payload, '$.chunk_index') AS INTEGER) LIMIT ?",
                (doc_id, chunk_index, chunk_index + limit, limit),
            ).fetchall()
        return [DocumentChunk.model_validate_json(row[0]) for row in rows]

    def get_pdf_bytes(self, doc_id: str) -> Optional[bytes]:
        with self._lock:
            row = self._connection.execute(
                "SELECT f.content FROM document_files f JOIN documents d ON d.id = f.doc_id "
                "WHERE f.doc_id = ?", (doc_id,)
            ).fetchone()
        return bytes(row[0]) if row else None

    def surrounding_chunks(self, doc_id: str, chunk_index: int, radius: int = 4) -> List[DocumentChunk]:
        """Bounded source context, including introductions before a retrieved list item."""
        radius = max(0, min(radius, 4))
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM chunks WHERE doc_id = ? "
                "AND CAST(json_extract(payload, '$.chunk_index') AS INTEGER) BETWEEN ? AND ? "
                "ORDER BY CAST(json_extract(payload, '$.chunk_index') AS INTEGER)",
                (doc_id, chunk_index-radius, chunk_index+radius)).fetchall()
        return [DocumentChunk.model_validate_json(row[0]) for row in rows]

    def has_pdf(self, doc_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM document_files WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row is not None

    def similarity_search(self, query: str, top_k: int = 4,
                          filter_doc_ids: Optional[List[str]] = None) -> List[Tuple[DocumentChunk, float]]:
        if top_k <= 0 or filter_doc_ids == []:
            return []
        tokens = re.findall(r"[\w]+", query.lower(), flags=re.UNICODE)
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens[:32])
        sql = ("SELECT c.payload, bm25(chunk_search) AS rank FROM chunk_search "
               "JOIN chunks c ON c.id = chunk_search.chunk_id "
               "WHERE chunk_search MATCH ?")
        params = [match]
        if filter_doc_ids is not None:
            if not filter_doc_ids:
                return []
            sql += f" AND c.doc_id IN ({','.join('?' for _ in filter_doc_ids)})"
            params.extend(filter_doc_ids)
        sql += " ORDER BY rank LIMIT ?"
        params.append(min(top_k, 100))
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        results = []
        query_terms = set(tokens)
        for row in rows:
            chunk = DocumentChunk.model_validate_json(row[0])
            chunk_terms = set(re.findall(r"[\w]+", chunk.content.lower(), flags=re.UNICODE))
            coverage = len(query_terms & chunk_terms) / len(query_terms)
            results.append((chunk, round(coverage, 3)))
        return results

    def ranked_search(self, query: str, top_k=3, filter_doc_ids=None):
        """Fuse BM25 and dense candidates, then rerank with semantic and lexical relevance."""
        from backend.agent.relevance import terms, coverage
        query_terms = terms(query)
        if not query_terms or top_k <= 0 or filter_doc_ids == []:
            return [], "keyword"
        lexical = self.similarity_search(" ".join(query_terms), max(top_k * 4, 12), filter_doc_ids) if query_terms else []
        lexical_scores = {chunk.id: coverage(query_terms, chunk.content) for chunk, _ in lexical}
        fallback = sorted(((chunk, lexical_scores[chunk.id]) for chunk, _ in lexical
                           if lexical_scores[chunk.id]), key=lambda item: item[1], reverse=True)[:top_k]
        if not self.embedder or not self.embedder.prepare():
            return fallback, "keyword"
        try:
            with self._lock:
                # Keep the document snapshot and its persisted vectors consistent with deletion.
                chunks = [chunk for chunk in self.chunks if filter_doc_ids is None or chunk.doc_id in filter_doc_ids]
                if not chunks:
                    return [], "hybrid"
                model = self.embedder.model_name
                vectors = {row[0]: json.loads(row[1]) for row in self._connection.execute(
                    "SELECT chunk_id, vector FROM chunk_vectors WHERE model = ?", (model,)).fetchall()}
                missing = [chunk for chunk in chunks if chunk.id not in vectors]
                if missing:
                    embedded = self.embedder.passages([chunk.content for chunk in missing])
                    if len(embedded) != len(missing):
                        raise ValueError("Incomplete embedding batch")
                    with self._connection:
                        for chunk, vector in zip(missing, embedded):
                            vectors[chunk.id] = vector
                            self._connection.execute("INSERT OR REPLACE INTO chunk_vectors VALUES (?, ?, ?)",
                                                     (chunk.id, model, json.dumps(vector)))
                query_vector = self.embedder.query(query)
                def cosine(vector):
                    if len(vector) != len(query_vector):
                        raise ValueError("Embedding dimensions changed")
                    denominator = math.sqrt(sum(v*v for v in vector) * sum(v*v for v in query_vector))
                    return sum(a*b for a,b in zip(vector, query_vector)) / denominator if denominator else 0.0
                dense = sorted(((chunk, cosine(vectors[chunk.id])) for chunk in chunks),
                               key=lambda item: item[1], reverse=True)
            dense_scores = {chunk.id: score for chunk, score in dense}
            fusion = {}
            for ranking in (lexical, dense[:max(top_k * 4, 12)]):
                for rank, (chunk, _) in enumerate(ranking, start=1):
                    fusion[chunk.id] = fusion.get(chunk.id, 0) + 1 / (60 + rank)
            candidates = []
            for chunk, semantic in dense:
                lexical_score = lexical_scores.get(chunk.id, 0)
                if chunk.id not in fusion or (not lexical_score and semantic < 0.60):
                    continue
                score = 0.65 * semantic + 0.20 * lexical_score + 0.15 * fusion[chunk.id] * 30
                candidates.append((chunk, round(score, 3)))
            return sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k], "hybrid"
        except Exception:
            return fallback, "keyword_fallback"
