from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import time
import uuid


class PlanStep(BaseModel):
    id: int
    title: str
    description: str
    status: Literal["pending", "running", "completed", "skipped"] = "pending"


class AgentAction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_id: int
    thought: str
    tool_name: Literal["web_search", "document_retrieval", "data_synthesis", "knowledge_graph_builder"]
    tool_input: Dict[str, Any]
    observation: str
    latency_ms: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class Citation(BaseModel):
    id: int
    source_type: Literal["document", "web"]
    title: str
    snippet: str
    relevance_score: float = 0.0
    url_or_page: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None


class KnowledgeNode(BaseModel):
    id: str
    label: str
    category: str = "Concept"
    importance: float = 1.0


class KnowledgeEdge(BaseModel):
    source: str
    target: str
    relationship: str


class KnowledgeGraph(BaseModel):
    nodes: List[KnowledgeNode] = Field(default_factory=list)
    edges: List[KnowledgeEdge] = Field(default_factory=list)


class Finding(BaseModel):
    heading: str
    content: str
    key_takeaway: Optional[str] = None
    citations: List[int] = Field(default_factory=list)


class ResearchReport(BaseModel):
    id: str = Field(default_factory=lambda: f"rep_{uuid.uuid4().hex[:10]}")
    query: str
    created_at: float = Field(default_factory=time.time)
    depth: Literal["quick", "balanced", "deep"] = "balanced"
    executive_summary: str
    key_findings: List[Finding] = Field(default_factory=list)
    comparative_matrix: Optional[Dict[str, Any]] = None
    strategic_implications: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    knowledge_graph: KnowledgeGraph = Field(default_factory=KnowledgeGraph)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    depth: Literal["quick", "balanced", "deep"] = "balanced"
    document_ids: Optional[List[str]] = Field(default_factory=list)
    api_key: Optional[str] = None
    search_web: bool = True
    use_documents: bool = True


class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:8]}")
    doc_id: str
    doc_name: str
    chunk_index: int
    content: str
    page_number: Optional[int] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    id: str
    filename: str
    file_type: str
    size_bytes: int
    chunks_count: int
    upload_time: float = Field(default_factory=time.time)
    preview_snippet: str = ""
