import os
import sqlite3
import json
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from backend.models import (
    ResearchReport, Finding, Citation, KnowledgeGraph, KnowledgeNode, KnowledgeEdge
)


class Database:
    """Persistent SQLite Storage for InsightPilot Research Reports & Session History."""

    def __init__(self, db_path: Optional[str] = None):
        if not db_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "insightpilot.db")

        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=15000;")
        except Exception:
            pass
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    depth TEXT NOT NULL,
                    executive_summary TEXT NOT NULL,
                    key_findings TEXT NOT NULL,
                    comparative_matrix TEXT,
                    strategic_implications TEXT,
                    citations TEXT NOT NULL,
                    knowledge_graph TEXT,
                    metrics TEXT
                )
            """)
            conn.commit()

    def save_report(self, report: ResearchReport) -> None:
        """Saves or updates a research report in SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO reports (
                    id, query, created_at, depth, executive_summary,
                    key_findings, comparative_matrix, strategic_implications,
                    citations, knowledge_graph, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.id,
                report.query,
                report.created_at,
                report.depth,
                report.executive_summary,
                json.dumps([f.model_dump() for f in report.key_findings]),
                json.dumps(report.comparative_matrix) if report.comparative_matrix else None,
                json.dumps(report.strategic_implications),
                json.dumps([c.model_dump() for c in report.citations]),
                json.dumps(report.knowledge_graph.model_dump()) if report.knowledge_graph else None,
                json.dumps(report.metrics)
            ))
            conn.commit()

    def list_reports(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists recent reports with summary metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, query, created_at, depth, executive_summary, citations, metrics
                FROM reports
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

            results = []
            for r in rows:
                citations_list = []
                try:
                    citations_list = json.loads(r["citations"])
                except Exception:
                    pass

                metrics_dict = {}
                try:
                    if r["metrics"]:
                        metrics_dict = json.loads(r["metrics"])
                except Exception:
                    pass

                results.append({
                    "id": r["id"],
                    "query": r["query"],
                    "created_at": r["created_at"],
                    "depth": r["depth"],
                    "snippet": (r["executive_summary"][:140] + "...") if r["executive_summary"] else "",
                    "citations_count": len(citations_list),
                    "model": metrics_dict.get("model", "Gemini 3.8 Flash")
                })
            return results

    def get_report(self, report_id: str) -> Optional[ResearchReport]:
        """Retrieves and deserializes a full ResearchReport by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
            row = cursor.fetchone()
            if not row:
                return None

            findings_raw = json.loads(row["key_findings"]) if row["key_findings"] else []
            findings = [Finding(**f) for f in findings_raw]

            citations_raw = json.loads(row["citations"]) if row["citations"] else []
            citations = [Citation(**c) for c in citations_raw]

            kg = KnowledgeGraph()
            if row["knowledge_graph"]:
                try:
                    kg_raw = json.loads(row["knowledge_graph"])
                    nodes = [KnowledgeNode(**n) for n in kg_raw.get("nodes", [])]
                    edges = [KnowledgeEdge(**e) for e in kg_raw.get("edges", [])]
                    kg = KnowledgeGraph(nodes=nodes, edges=edges)
                except Exception:
                    pass

            matrix = None
            if row["comparative_matrix"]:
                try:
                    matrix = json.loads(row["comparative_matrix"])
                except Exception:
                    pass

            implications = []
            if row["strategic_implications"]:
                try:
                    implications = json.loads(row["strategic_implications"])
                except Exception:
                    pass

            metrics = {}
            if row["metrics"]:
                try:
                    metrics = json.loads(row["metrics"])
                except Exception:
                    pass

            return ResearchReport(
                id=row["id"],
                query=row["query"],
                created_at=row["created_at"],
                depth=row["depth"],
                executive_summary=row["executive_summary"],
                key_findings=findings,
                comparative_matrix=matrix,
                strategic_implications=implications,
                citations=citations,
                knowledge_graph=kg,
                metrics=metrics
            )

    def delete_report(self, report_id: str) -> bool:
        """Deletes a report by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))
            conn.commit()
            return cursor.rowcount > 0
