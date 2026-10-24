"""Persistent document storage and indexed lexical retrieval."""

import os
import re
import sqlite3
import threading
from typing import List, Optional, Tuple

from backend.models import DocumentChunk, DocumentMetadata


class VectorStore:
    """SQLite FTS5 backed retrieval. The legacy name is kept for API compatibility."""

    def __init__(self, gemini_client=None, db_path: Optional[str] = None):
        self.gemini_client = gemini_client
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

    def get_pdf_bytes(self, doc_id: str) -> Optional[bytes]:
        with self._lock:
            row = self._connection.execute(
                "SELECT f.content FROM document_files f JOIN documents d ON d.id = f.doc_id "
                "WHERE f.doc_id = ?", (doc_id,)
            ).fetchone()
        return bytes(row[0]) if row else None

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
