"""Bounded list extraction with verbatim support from contiguous source chunks."""

import re
from collections import defaultdict

from backend.agent.relevance import coverage, topic_terms
from backend.models import ClaimEvidence, Finding, ResearchReport


LIST_INTENT = re.compile(r"\b(stages?|steps?|phases?|items?|list|components?|requirements?)\b", re.I)
MARKER = re.compile(r"(?<!\S)(?:[-*•]\s+|\d{1,2}[.)]\s+)")
HEADING = re.compile(r"(?<!\S)#{1,6}\s+[^\n]+")
FRAMING = {"stage", "stages", "step", "steps", "phase", "phases", "item", "items",
           "list", "listed", "component", "requirement", "required", "all", "each", "main", "key"}
MAX_ITEMS = 12


def is_list_question(query):
    return bool(LIST_INTENT.search(query)) and not re.search(
        r"\b(compare|comparison|versus|vs|difference)\b", query, re.I)


def _runs(citations):
    """Never stitch different documents, nonadjacent chunks, or truncated snippets."""
    groups = defaultdict(list)
    for citation in citations:
        if citation.evidence_type != "source_excerpt":
            continue
        if citation.doc_id and citation.chunk_index is not None:
            groups[citation.doc_id].append(citation)
        else:
            yield [citation]
    for group in groups.values():
        run = []
        seen = set()
        for citation in sorted(group, key=lambda c: c.chunk_index):
            if citation.chunk_index in seen:
                continue
            seen.add(citation.chunk_index)
            if run and (citation.chunk_index != run[-1].chunk_index + 1 or run[-1].snippet.endswith("...")):
                yield run
                run = []
            run.append(citation)
        if run:
            yield run


def _merge(run):
    text, spans = "", []
    for citation in run:
        snippet = citation.snippet
        overlap = 0
        # Chunking retains the last sentence. Keep one copy and its source offsets.
        for size in range(min(len(text), len(snippet)), 19, -1):
            if text.endswith(snippet[:size]):
                overlap = size
                break
        if text and not overlap:
            text += "\n"
        start = len(text) - overlap
        text += snippet[overlap:]
        spans.append((start, start + len(snippet), citation))
    return text, spans


def _blocks(text):
    markers = []
    for marker in MARKER.finditer(text):
        before = text[:marker.start()].rstrip(" \t")
        if re.search(r"#{1,6}\s*$", before):
            continue
        # A number at the end of a sentence is not a numbered list marker.
        if marker.group()[0].isdigit() and before and before[-1] not in "\n:.!?;":
            continue
        markers.append(marker)
    block = []
    previous_end = 0
    context_start = 0
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        tail = text[marker.end():end]
        # A new heading, paragraph, or prose sentence ends a list item.
        boundary = re.search(r"\n\s*\n|(?<!\S)#{1,6}\s|(?<=[.!?])\s+(?=[A-Z])", tail)
        if boundary:
            end = marker.end() + boundary.start()
        while end > marker.end() and text[end-1].isspace():
            end -= 1
        gap = text[previous_end:marker.start()].strip()
        if block and gap:
            yield block, context_start
            block = []
            context_start = previous_end
        item = text[marker.end():end]
        if item.strip() and not item.rstrip().endswith("..."):
            block.append((marker.start(), marker.end(), end))
        previous_end = end
    if block:
        yield block, context_start


def _support(start, end, text, spans):
    """Cover every non-whitespace character with a quote from its actual citation."""
    evidence = []
    cursor = start
    while cursor < end:
        if text[cursor].isspace():
            cursor += 1
            continue
        available = [span for span in spans if span[0] <= cursor < span[1]]
        if not available:
            return []
        left, right, citation = max(available, key=lambda span: span[1])
        stop = min(end, right)
        quote = text[cursor:stop].strip()
        if quote:
            evidence.append(ClaimEvidence(citation_id=citation.id, quote=quote, verification="exact_quote"))
        cursor = stop
    return evidence


def extract_list_report(query, citations, depth):
    if not is_list_question(query):
        return None
    subject = [term for term in topic_terms(query) if term not in FRAMING]
    if not subject:
        return None
    candidates = []
    for run in _runs(citations):
        text, spans = _merge(run)
        for block, context_start in _blocks(text):
            first = block[0][0]
            prefix = text[max(context_start, first-600):first]
            headings = list(HEADING.finditer(prefix))
            if headings:
                prefix = prefix[headings[-1].start():]
            # Anchor in the heading/introduction, never in a generic document title.
            score = coverage(subject, prefix)
            if not score or len(block) < 2:
                continue
            findings, seen = [], set()
            for _, start, end in block:
                item = text[start:end].strip()
                key = re.sub(r"\s+", " ", item).casefold()
                if key in seen:
                    continue
                support = _support(start, end, text, spans)
                if not support:
                    continue
                seen.add(key)
                ids = list(dict.fromkeys(e.citation_id for e in support))
                label = item.split(":", 1)[0].strip("* ")
                heading = f"{len(findings)+1}. {label}" if len(label) <= 90 else f"Item {len(findings)+1}"
                findings.append(Finding(heading=heading, content=item + " " + " ".join(f"[{i}]" for i in ids),
                                        citations=ids, evidence=support))
            if len(findings) >= 2:
                candidates.append((score, findings))
    if not candidates:
        return None
    # Keep a coherent list from one context; never splice different procedures.
    _, findings = max(candidates, key=lambda candidate: (candidate[0], len(candidate[1])))
    limited = len(findings) > MAX_ITEMS
    findings = findings[:MAX_ITEMS]
    used = {cid for finding in findings for cid in finding.citations}
    return ResearchReport(query=query, depth=depth,
        executive_summary=f"Found {len(findings)} listed items in the retrieved evidence, shown below in source order. "
                          "This may not be the complete list in the original source.",
        key_findings=findings, citations=[c for c in citations if c.id in used],
        metrics={"mode": "extractive_list", "list_items_returned": len(findings), "list_limit_reached": limited})
