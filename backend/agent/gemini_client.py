import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from backend.models import PlanStep, Citation, Finding, ResearchReport
from backend.agent.relevance import best_sentence, coverage, topic_terms


class GeminiClient:
    """Interfaces with Gemini (gemini-3.5-flash-lite / gemini-3.8-flash) via google-genai SDK with autonomous function calling and fallback intelligence."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self._client = None
        self._init_sdk()

    def _init_sdk(self):
        if self.api_key:
            try:
                from google import genai
                from google.genai import types
                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        timeout=15000,
                        retry_options=types.HttpRetryOptions(attempts=2)
                    )
                )
            except Exception as e:
                print(f"[GeminiClient] Could not initialize google-genai SDK: {e}")
                self._client = None

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()
        self._init_sdk()

    def has_api_key(self) -> bool:
        return bool(self.api_key and self._client)

    def get_registered_tools(self) -> List[Dict[str, Any]]:
        """Declarations of tools available to Gemini for autonomous function calling."""
        return [
            {
                "type": "function",
                "name": "search_documents",
                "description": "Searches indexed user documents using full-text retrieval.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Specific keywords or question to retrieve relevant chunks from ingested documents"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of document chunks to retrieve (default 3)"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "type": "function",
                "name": "search_web",
                "description": "Searches DuckDuckGo Instant Answer and Wikipedia for source excerpts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Web search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of search results to return (default 3)"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def create_interaction(
        self,
        input_data: Any,
        previous_interaction_id: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ):
        """Dispatches an interaction to Gemini 3.8 Flash."""
        if not self.has_api_key():
            return None

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_data
        }
        if previous_interaction_id:
            kwargs["previous_interaction_id"] = previous_interaction_id
        if tools:
            kwargs["tools"] = tools

        return self._client.interactions.create(**kwargs)

    def embed_text(self, text: str) -> Optional[List[float]]:
        if not self.has_api_key():
            return None
        try:
            resp = self._client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
            if hasattr(resp, "embedding") and hasattr(resp.embedding, "values"):
                return resp.embedding.values
            elif isinstance(resp, dict) and "embedding" in resp:
                return resp["embedding"].get("values")
        except Exception as e:
            print(f"[GeminiClient] embed_text error: {e}")
        return None

    def plan_research(self, query: str, doc_summaries: List[str], depth: str = "balanced") -> List[PlanStep]:
        """Plans the research decomposition into structured milestones."""
        if self.has_api_key():
            try:
                prompt = f"""You are an autonomous Research Planner Agent.
User Query: "{query}"
Available Ingested Documents: {doc_summaries}
Depth: {depth}

Create a numbered 4-step execution plan for an autonomous research workflow.
Return strictly a valid JSON array of objects with keys: "id" (int), "title" (string), "description" (string).
Example:
[
  {{"id": 1, "title": "Deconstruct Scope & Query Document Store", "description": "Formulate search queries for local document context"}},
  {{"id": 2, "title": "Retrieve Web Intel & State-of-the-Art Benchmarks", "description": "Gather live external verification and industry data"}},
  {{"id": 3, "title": "Review Retrieved Evidence", "description": "Identify relevant source excerpts"}},
  {{"id": 4, "title": "Verify Source Quotes", "description": "Check selected quotes against citation text"}}
]
"""
                resp = self._client.interactions.create(
                    model=self.model,
                    input=prompt
                )
                text = (resp.output_text or "").strip()
                start = text.find("[")
                end = text.rfind("]") + 1
                if start != -1 and end != 0:
                    data = json.loads(text[start:end])
                    return [PlanStep(**item) for item in data]
            except Exception as e:
                print(f"[GeminiClient] Fallback to deterministic planner due to: {e}")

        # Deterministic plan
        return [
            PlanStep(
                id=1,
                title="Deconstruct Query & Query Vector Store",
                description=f"Find relevant excerpts for '{query[:40]}...' in indexed documents."
            ),
            PlanStep(
                id=2,
                title="Search Web Sources",
                description="Look for sources that match the question."
            ),
            PlanStep(
                id=3,
                title="Review Retrieved Evidence",
                description="Identify source excerpts relevant to the question."
            ),
            PlanStep(
                id=4,
                title="Verify Source Quotes",
                description="Check selected quotes against their cited source excerpts."
            )
        ]

    def synthesize_final_report(
        self,
        query: str,
        observations: List[Dict[str, Any]],
        citations: List[Citation],
        depth: str = "balanced"
    ) -> ResearchReport:
        """Synthesizes observations and citations into a comprehensive executive report."""
        if not citations:
            return self._generate_fallback_report(query, observations, citations, depth)
        if self.has_api_key():
            try:
                cite_summary = "\n".join(
                    f"[{c.id}] {c.title}: {c.snippet.removesuffix('...').strip()}"
                    for c in citations[:20]
                )
                prompt = f"""Select up to five useful source excerpts for this research query: {query!r}.
Use only the numbered excerpts below. Copy each quote exactly from its own excerpt.
Do not paraphrase, infer a claim, add facts, or use a quote from a different citation.
Every quote must contain at least 24 characters.

{cite_summary}

Return only JSON: {{"selected_evidence": [{{"citation_id": 1, "quote": "exact source text"}}]}}
Return an empty list if none qualify.
"""
                resp = self._client.interactions.create(
                    model=self.model,
                    input=prompt
                )
                text = (resp.output_text or "").strip()
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end != 0:
                    data = json.loads(text[start:end])
                    findings, rejected_count = self._verify_selected_evidence(
                        data.get("selected_evidence", []), citations
                    )
                    if not findings:
                        raise ValueError("Model selected no valid source quotes")

                    return ResearchReport(
                        query=query,
                        depth=depth,
                        executive_summary=findings[0].content,
                        key_findings=findings,
                        citations=citations,
                        metrics={"model": self.model, "mode": "verified_quotes",
                                 "rejected_quotes": rejected_count}
                    )
            except Exception as e:
                print(f"[GeminiClient] Gemini synthesis error: {e}. Using fallback synthesizer.")

        return self._generate_fallback_report(query, observations, citations, depth)

    @staticmethod
    def _verify_selected_evidence(selected: Any, citations: List[Citation]) -> Tuple[List[Finding], int]:
        """Accept only substantial quotes copied from the matching citation excerpt."""
        if not isinstance(selected, list):
            return [], 1
        by_id = {c.id: c for c in citations[:20]}
        findings: List[Finding] = []
        rejected = 0
        seen = set()
        for item in selected[:20]:
            if not isinstance(item, dict):
                rejected += 1
                continue
            citation_id = item.get("citation_id")
            quote = item.get("quote")
            citation = by_id.get(citation_id) if type(citation_id) is int else None
            if not citation or not isinstance(quote, str):
                rejected += 1
                continue
            normalized_quote = re.sub(r"\s+", " ", quote).strip()
            source = re.sub(r"\s+", " ", citation.snippet.removesuffix("...")).strip()
            if (len(normalized_quote) < 24 or normalized_quote not in source or
                    (citation_id, normalized_quote) in seen):
                rejected += 1
                continue
            seen.add((citation_id, normalized_quote))
            findings.append(Finding(
                heading=citation.title,
                content=f"{normalized_quote} [{citation_id}]",
                citations=[citation_id]
            ))
            if len(findings) == 5:
                break
        return findings, rejected

    def _generate_fallback_report(
        self,
        query: str,
        observations: List[Dict[str, Any]],
        citations: List[Citation],
        depth: str
    ) -> ResearchReport:
        if not citations:
            return ResearchReport(
                query=query, depth=depth,
                executive_summary="No source relevant enough to answer this question was found. Try a narrower question or add a relevant document.",
                metrics={"mode": "no_evidence"}
            )

        candidates = []
        query_terms = topic_terms(query)
        needs_comparison = bool(re.search(r"\b(compare|comparison|versus|vs|difference)\b", query, re.I))
        for citation in citations:
            excerpt = citation.snippet.removesuffix("...").strip()
            sentence = best_sentence(query, excerpt)
            if not sentence or not coverage(query_terms, citation.title + " " + sentence):
                continue
            if needs_comparison and len(set(query_terms) & set(topic_terms(sentence))) < len(query_terms):
                continue
            score = coverage(query_terms, sentence) + 0.25 * coverage(query_terms, citation.title)
            candidates.append((score, citation, sentence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        findings = [Finding(heading=c.title, content=f"{sentence} [{c.id}]", citations=[c.id])
                    for _, c, sentence in candidates[:3]]
        if not findings:
            return ResearchReport(
                query=query, depth=depth,
                executive_summary="The retrieved sources do not contain enough relevant text to answer this question.",
                citations=[], metrics={"mode": "no_evidence"}
            )
        used_ids = {finding.citations[0] for finding in findings}
        return ResearchReport(
            query=query, depth=depth,
            executive_summary=findings[0].content,
            key_findings=findings,
            citations=[c for c in citations if c.id in used_ids],
            metrics={"mode": "extractive_fallback"}
        )
