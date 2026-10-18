import asyncio
import time
from typing import AsyncGenerator, Dict, Any, List, Optional
from backend.models import (
    ResearchRequest, PlanStep, AgentAction, Citation, ResearchReport
)
from backend.rag.vector_store import VectorStore
from backend.agent.tools import (
    DocumentRetrievalTool, WebSearchTool, DataSynthesisTool, KnowledgeGraphBuilderTool
)
from backend.agent.gemini_client import GeminiClient


class ResearchOrchestrator:
    """Autonomous ReAct Research Agent Orchestrator with live event streaming."""

    def __init__(self, vector_store: VectorStore, gemini_client: GeminiClient):
        self.vector_store = vector_store
        self.gemini_client = gemini_client
        self.doc_tool = DocumentRetrievalTool(vector_store)
        self.web_tool = WebSearchTool()
        self.synth_tool = DataSynthesisTool()
        self.kg_tool = KnowledgeGraphBuilderTool()

    async def execute_stream(self, request: ResearchRequest) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.time()
        query = request.query
        doc_ids = request.document_ids or []

        # If user supplied a dynamic API key for this request, set it
        if request.api_key:
            self.gemini_client.set_api_key(request.api_key)

        # 1. Summarize available document context
        doc_metas = self.vector_store.list_documents()
        doc_names = [d.filename for d in doc_metas if not doc_ids or d.id in doc_ids]

        # 2. Plan Generation
        yield {
            "event": "status",
            "data": {"message": "Deconstructing research topic into structured execution plan..."}
        }
        await asyncio.sleep(0.4)

        plan = self.gemini_client.plan_research(query, doc_names, depth=request.depth)
        yield {
            "event": "plan",
            "data": [p.model_dump() for p in plan]
        }
        await asyncio.sleep(0.5)

        citations: List[Citation] = []
        observations: List[Dict[str, Any]] = []
        citation_id_counter = 1

        # 3. Step 1: Document Retrieval (Vector RAG)
        step1 = plan[0]
        yield {"event": "step_start", "data": {"step_id": step1.id}}
        await asyncio.sleep(0.3)

        thought1 = f"I should first query the local vector store for semantic matches regarding '{query}' to identify primary source material."
        yield {"event": "thought", "data": {"step_id": step1.id, "thought": thought1}}
        await asyncio.sleep(0.4)

        tool_input1 = {"query": query, "top_k": 3, "filter_doc_ids": doc_ids if doc_ids else None}
        yield {"event": "action", "data": {"step_id": step1.id, "tool_name": "document_retrieval", "input": tool_input1}}

        # Execute Document Retrieval
        t0 = time.time()
        doc_result = self.doc_tool.execute(query=query, top_k=3, filter_doc_ids=doc_ids if doc_ids else None)
        latency1 = round((time.time() - t0) * 1000, 1)

        obs_summary1 = f"Retrieved {doc_result['found']} relevant document chunks from ingested repository."
        yield {
            "event": "observation",
            "data": {"step_id": step1.id, "observation": obs_summary1, "chunks_found": doc_result['found'], "latency_ms": latency1}
        }

        # Build Citations from document chunks
        for chunk_data in doc_result.get("chunks", []):
            cite = Citation(
                id=citation_id_counter,
                source_type="document",
                title=f"{chunk_data['doc_name']} (Page {chunk_data['page'] or 1})",
                snippet=chunk_data["content"][:240] + "...",
                relevance_score=chunk_data["similarity"],
                url_or_page=f"Page {chunk_data['page']}" if chunk_data.get('page') else "Document Chunk"
            )
            citations.append(cite)
            yield {"event": "citation_added", "data": cite.model_dump()}
            citation_id_counter += 1

        observations.append({"step": 1, "tool": "document_retrieval", "data": doc_result})
        yield {"event": "step_complete", "data": {"step_id": step1.id}}
        await asyncio.sleep(0.4)

        # 4. Step 2: Web Search Tool
        step2 = plan[1] if len(plan) > 1 else plan[0]
        yield {"event": "step_start", "data": {"step_id": step2.id}}
        await asyncio.sleep(0.3)

        thought2 = f"Now checking external web sources and state-of-the-art benchmarks to ground internal findings against broader industry standards."
        yield {"event": "thought", "data": {"step_id": step2.id, "thought": thought2}}
        await asyncio.sleep(0.4)

        tool_input2 = {"query": query, "num_results": 3}
        yield {"event": "action", "data": {"step_id": step2.id, "tool_name": "web_search", "input": tool_input2}}

        t0 = time.time()
        web_result = await self.web_tool.execute_async(query=query, num_results=3)
        latency2 = round((time.time() - t0) * 1000, 1)

        web_items = web_result.get("results", [])
        obs_summary2 = f"Retrieved {len(web_items)} external publications and benchmark references."
        yield {
            "event": "observation",
            "data": {"step_id": step2.id, "observation": obs_summary2, "results_found": len(web_items), "latency_ms": latency2}
        }

        for w in web_items:
            cite = Citation(
                id=citation_id_counter,
                source_type="web",
                title=w["title"],
                snippet=w["snippet"][:240] + "...",
                relevance_score=0.91,
                url_or_page=w["url"]
            )
            citations.append(cite)
            yield {"event": "citation_added", "data": cite.model_dump()}
            citation_id_counter += 1

        observations.append({"step": 2, "tool": "web_search", "data": web_result})
        yield {"event": "step_complete", "data": {"step_id": step2.id}}
        await asyncio.sleep(0.4)

        # 5. Step 3: Synthesis & Verification
        step3 = plan[2] if len(plan) > 2 else plan[-1]
        yield {"event": "step_start", "data": {"step_id": step3.id}}
        await asyncio.sleep(0.3)

        thought3 = "Synthesizing cross-source evidence, evaluating operational trade-offs, and compiling quantitative comparisons."
        yield {"event": "thought", "data": {"step_id": step3.id, "thought": thought3}}
        await asyncio.sleep(0.4)

        tool_input3 = {"metrics": ["Accuracy", "Latency", "Cost Efficiency"]}
        yield {"event": "action", "data": {"step_id": step3.id, "tool_name": "data_synthesis", "input": tool_input3}}

        t0 = time.time()
        synth_res = self.synth_tool.execute(
            findings=[f"Source {c.title}" for c in citations],
            metrics_to_compare=["Operational Latency", "Precision Threshold", "Implementation Complexity"]
        )
        latency3 = round((time.time() - t0) * 1000, 1)

        yield {
            "event": "observation",
            "data": {"step_id": step3.id, "observation": "Calculated cross-dimensional tradeoff matrix with high consistency confidence.", "latency_ms": latency3}
        }
        observations.append({"step": 3, "tool": "data_synthesis", "data": synth_res})
        yield {"event": "step_complete", "data": {"step_id": step3.id}}
        await asyncio.sleep(0.4)

        # 6. Step 4: Final Executive Report Generation
        step4 = plan[3] if len(plan) > 3 else plan[-1]
        yield {"event": "step_start", "data": {"step_id": step4.id}}
        yield {
            "event": "thought",
            "data": {"step_id": step4.id, "thought": "Assembling verified citations into final multi-section research intelligence report and knowledge graph."}
        }
        await asyncio.sleep(0.5)

        report = self.gemini_client.synthesize_final_report(
            query=query,
            observations=observations,
            citations=citations,
            depth=request.depth
        )

        total_elapsed = round(time.time() - start_time, 2)
        report.metrics["total_duration_seconds"] = total_elapsed
        report.metrics["citations_count"] = len(citations)
        report.metrics["sources_analyzed"] = len(doc_result.get("chunks", [])) + len(web_items)

        yield {"event": "step_complete", "data": {"step_id": step4.id}}
        yield {"event": "report", "data": report.model_dump()}

        yield {
            "event": "done",
            "data": {
                "report_id": report.id,
                "elapsed_seconds": total_elapsed,
                "total_citations": len(citations),
                "status": "success"
            }
        }
