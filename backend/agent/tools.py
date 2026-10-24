import httpx
import json
import re
import urllib.parse
from typing import List, Dict, Any, Optional
from backend.rag.vector_store import VectorStore
from backend.models import KnowledgeNode, KnowledgeEdge, KnowledgeGraph


class DocumentRetrievalTool:
    """Tool for retrieving relevant text chunks from uploaded user documents."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.name = "document_retrieval"
        self.description = "Searches indexed user documents using SQLite full-text search."

    def execute(self, query: str, top_k: int = 3, filter_doc_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        results = self.vector_store.similarity_search(query=query, top_k=top_k, filter_doc_ids=filter_doc_ids)
        if not results:
            return {
                "found": 0,
                "message": "No matching document excerpts found for this query.",
                "chunks": []
            }

        extracted = []
        for chunk, score in results:
            extracted.append({
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page": chunk.page_number,
                "content": chunk.content,
                "similarity": round(score, 3)
            })

        return {
            "found": len(extracted),
            "query": query,
            "chunks": extracted
        }


class WebSearchTool:
    """Multi-strategy live web search engine supporting DuckDuckGo and Wikipedia."""

    def __init__(self):
        self.name = "web_search"
        self.description = "Performs real-time web knowledge retrieval across news, technical docs, and live encyclopedic sources."
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        self._wiki_headers = {
            "User-Agent": "InsightPilot/1.0 (https://github.com/Manas-Jain-24/insightpilot; research@insightpilot.ai)"
        }

    async def execute_async(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        results = []

        # Strategy 1: Live DuckDuckGo API (Abstract + Related Topics)
        try:
            clean_q = re.sub(r'site:\S+', '', query).strip()
            async with httpx.AsyncClient(timeout=5.0, headers=self._headers) as client:
                ddg_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q)}&format=json&no_html=1&skip_disambig=1"
                resp = await client.get(ddg_url)
                if resp.status_code == 200:
                    data = resp.json()
                    abstract = data.get("AbstractText", "")
                    abstract_url = data.get("AbstractURL", "")
                    heading = data.get("Heading", query)

                    if abstract and abstract_url:
                        results.append({
                            "title": heading,
                            "snippet": abstract,
                            "url": abstract_url
                        })

                    # Check RelatedTopics
                    for topic in data.get("RelatedTopics", []):
                        if len(results) >= num_results:
                            break
                        if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                            results.append({
                                "title": topic["Text"].split(" - ")[0] if " - " in topic["Text"] else query,
                                "snippet": topic["Text"],
                                "url": topic["FirstURL"]
                            })
        except Exception:
            pass

        # Strategy 2: Wikipedia Search & Extracts API (for authoritative academic & encyclopedic summaries)
        if len(results) < num_results:
            try:
                clean_term = re.sub(r'site:\S+', '', query).strip()
                clean_term = re.sub(r'[^\w\s]', ' ', clean_term).strip()
                words = [w for w in clean_term.split() if len(w) > 2]
                search_term = " ".join(words[:6]) if len(words) > 6 else clean_term

                async with httpx.AsyncClient(timeout=5.0, headers=self._wiki_headers) as client:
                    wiki_params = {
                        "action": "query",
                        "format": "json",
                        "generator": "search",
                        "gsrsearch": search_term,
                        "gsrlimit": num_results,
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "exsentences": 3
                    }
                    resp = await client.get("https://en.wikipedia.org/w/api.php", params=wiki_params)
                    if resp.status_code == 200:
                        data = resp.json()
                        pages = data.get("query", {}).get("pages", {})
                        for pid, page in pages.items():
                            if len(results) >= num_results:
                                break
                            title = page.get("title", "")
                            extract = page.get("extract", "").strip()
                            if title and extract:
                                results.append({
                                    "title": title,
                                    "snippet": extract,
                                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                                })
            except Exception:
                pass

        if results:
            return {
                "source": "live_web_engine",
                "query": query,
                "results": results[:num_results]
            }

        # A failed search is missing evidence, not a source to cite.
        return {
            "source": "unavailable",
            "query": query,
            "results": [],
            "message": "Live web search returned no usable sources."
        }


class DataSynthesisTool:
    """Extracts comparative metrics and highlights trade-offs."""

    def __init__(self):
        self.name = "data_synthesis"
        self.description = "Aggregates observations across document chunks and web findings into a comparative matrix."

    def execute(self, findings: List[str], metrics_to_compare: List[str]) -> Dict[str, Any]:
        return {
            "matrix": {metric: list(findings) for metric in metrics_to_compare},
            "source_observations": len(findings)
        }


class KnowledgeGraphBuilderTool:
    """Constructs concept entities and relationship edges for visualization."""

    def __init__(self):
        self.name = "knowledge_graph_builder"
        self.description = "Generates a structured entity-relationship graph connecting core topics and findings."

    def execute(self, topic: str, subtopics: List[str]) -> KnowledgeGraph:
        nodes = [KnowledgeNode(id="core", label=topic, category="Core Focus", importance=2.0)]
        edges = []

        for idx, sub in enumerate(subtopics):
            node_id = f"sub_{idx}"
            nodes.append(KnowledgeNode(
                id=node_id,
                label=sub,
                category="Insight Area",
                importance=1.2
            ))
            edges.append(KnowledgeEdge(
                source="core",
                target=node_id,
                relationship="analyzed_dimension"
            ))

        return KnowledgeGraph(nodes=nodes, edges=edges)
