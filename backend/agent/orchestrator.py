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


from backend.db.database import Database


class ResearchOrchestrator:
    """Autonomous ReAct Research Agent Orchestrator with live event streaming and Gemini function calling."""

    def __init__(self, vector_store: VectorStore, gemini_client: GeminiClient, db: Optional[Database] = None):
        self.vector_store = vector_store
        self.gemini_client = gemini_client
        self.db = db
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
        await asyncio.sleep(0.3)

        plan = self.gemini_client.plan_research(query, doc_names, depth=request.depth)
        yield {
            "event": "plan",
            "data": [p.model_dump() for p in plan]
        }
        await asyncio.sleep(0.4)

        citations: List[Citation] = []
        observations: List[Dict[str, Any]] = []
        citation_id_counter = 1

        # Check if we can run true autonomous Gemini function calling
        if self.gemini_client.has_api_key():
            try:
                step1 = plan[0]
                yield {"event": "step_start", "data": {"step_id": step1.id}}

                system_prompt = (
                    f"You are InsightPilot, an autonomous research agent investigating: '{query}'. "
                    f"Available tools: search_documents (for user uploaded files) and search_web (for live facts). "
                    f"Use these tools to gather facts before synthesizing a final answer."
                )

                interaction = self.gemini_client.create_interaction(
                    input_data=system_prompt,
                    tools=self.gemini_client.get_registered_tools()
                )

                turn = 0
                max_turns = 4
                current_step_id = 1

                while interaction and turn < max_turns:
                    turn += 1
                    steps = interaction.steps or []
                    has_tool_call = False

                    for step in steps:
                        # If model emitted internal reasoning
                        if step.type == "thought":
                            thought_text = getattr(step, "summary", "") or "Evaluating evidence and deciding next action."
                            yield {
                                "event": "thought",
                                "data": {"step_id": current_step_id, "thought": thought_text}
                            }
                            await asyncio.sleep(0.3)

                        elif step.type == "function_call":
                            has_tool_call = True
                            tool_name = step.name
                            args = step.arguments or {}

                            # Map plan step status
                            if current_step_id <= len(plan):
                                yield {"event": "step_start", "data": {"step_id": current_step_id}}

                            yield {
                                "event": "action",
                                "data": {
                                    "step_id": current_step_id,
                                    "tool_name": tool_name,
                                    "input": args
                                }
                            }

                            t0 = time.time()
                            tool_result_str = ""

                            if tool_name == "search_documents":
                                q = args.get("query", query)
                                top_k = args.get("top_k", 3)
                                doc_res = self.doc_tool.execute(query=q, top_k=top_k, filter_doc_ids=doc_ids if doc_ids else None)
                                latency = round((time.time() - t0) * 1000, 1)

                                for chunk_data in doc_res.get("chunks", []):
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

                                obs_summary = f"Retrieved {doc_res['found']} relevant document chunks from ingested repository."
                                yield {
                                    "event": "observation",
                                    "data": {"step_id": current_step_id, "observation": obs_summary, "latency_ms": latency}
                                }
                                observations.append({"tool": tool_name, "data": doc_res})
                                tool_result_str = str(doc_res)

                            elif tool_name == "search_web":
                                q = args.get("query", query)
                                num = args.get("num_results", 3)
                                web_res = await self.web_tool.execute_async(query=q, num_results=num)
                                latency = round((time.time() - t0) * 1000, 1)

                                for w in web_res.get("results", []):
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

                                obs_summary = f"Retrieved {len(web_res.get('results', []))} external publications and benchmark references."
                                yield {
                                    "event": "observation",
                                    "data": {"step_id": current_step_id, "observation": obs_summary, "latency_ms": latency}
                                }
                                observations.append({"tool": tool_name, "data": web_res})
                                tool_result_str = str(web_res)

                            yield {"event": "step_complete", "data": {"step_id": current_step_id}}
                            current_step_id = min(current_step_id + 1, len(plan))
                            await asyncio.sleep(0.3)

                            # Send tool execution result back to Gemini
                            interaction = self.gemini_client.create_interaction(
                                previous_interaction_id=interaction.id,
                                input_data=[{
                                    "type": "function_result",
                                    "name": step.name,
                                    "call_id": step.id,
                                    "result": tool_result_str
                                }]
                            )

                    # If model didn't ask for any more tools, it is ready to synthesize
                    if not has_tool_call:
                        break

            except Exception as e:
                print(f"[ResearchOrchestrator] Dynamic interaction error: {e}. Falling back to guided execution.")
                # We will fall through to ensure citations and final report are synthesized reliably

        # If citations were empty (fallback), ensure we perform baseline retrieval
        if not citations:
            step1 = plan[0]
            yield {"event": "step_start", "data": {"step_id": step1.id}}
            yield {
                "event": "thought",
                "data": {"step_id": step1.id, "thought": f"Querying vector database for primary source documentation regarding '{query}'."}
            }
            t0 = time.time()
            doc_result = self.doc_tool.execute(query=query, top_k=3, filter_doc_ids=doc_ids if doc_ids else None)
            latency1 = round((time.time() - t0) * 1000, 1)

            yield {
                "event": "action",
                "data": {"step_id": step1.id, "tool_name": "document_retrieval", "input": {"query": query, "top_k": 3}}
            }
            yield {
                "event": "observation",
                "data": {"step_id": step1.id, "observation": f"Retrieved {doc_result['found']} excerpts from vector store.", "latency_ms": latency1}
            }
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

            observations.append({"tool": "document_retrieval", "data": doc_result})
            yield {"event": "step_complete", "data": {"step_id": step1.id}}
            await asyncio.sleep(0.3)

            # Web search baseline
            step2 = plan[1] if len(plan) > 1 else plan[0]
            yield {"event": "step_start", "data": {"step_id": step2.id}}
            yield {
                "event": "thought",
                "data": {"step_id": step2.id, "thought": "Gathering live web intelligence and cross-referencing benchmarks."}
            }
            t0 = time.time()
            web_result = await self.web_tool.execute_async(query=query, num_results=3)
            latency2 = round((time.time() - t0) * 1000, 1)

            yield {
                "event": "action",
                "data": {"step_id": step2.id, "tool_name": "web_search", "input": {"query": query, "num_results": 3}}
            }
            yield {
                "event": "observation",
                "data": {"step_id": step2.id, "observation": f"Retrieved {len(web_result.get('results', []))} benchmark sources.", "latency_ms": latency2}
            }
            for w in web_result.get("results", []):
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

            observations.append({"tool": "web_search", "data": web_result})
            yield {"event": "step_complete", "data": {"step_id": step2.id}}
            await asyncio.sleep(0.3)

        # Final Synthesis Phase
        final_step = plan[-1]
        yield {"event": "step_start", "data": {"step_id": final_step.id}}
        yield {
            "event": "thought",
            "data": {"step_id": final_step.id, "thought": "Synthesizing cross-source evidence into grounded executive brief with comparative matrix and concept graph."}
        }
        await asyncio.sleep(0.4)

        report = self.gemini_client.synthesize_final_report(
            query=query,
            observations=observations,
            citations=citations,
            depth=request.depth
        )

        total_elapsed = round(time.time() - start_time, 2)
        report.metrics["total_duration_seconds"] = total_elapsed
        report.metrics["citations_count"] = len(citations)

        # Auto-save report to SQLite database
        if self.db:
            try:
                self.db.save_report(report)
            except Exception as e:
                print(f"[ResearchOrchestrator] Failed to auto-save report to DB: {e}")

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
