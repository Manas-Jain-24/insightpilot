"""Shared lexical relevance checks for retrieval and extractive answers."""

import re
from typing import List


STOP_WORDS = {
    "a", "an", "and", "about", "are", "as", "at", "be", "by", "can", "could",
    "define", "describe", "do", "does", "explain", "find", "for", "from", "give",
    "how", "in", "is", "it", "me", "of", "on", "or", "please", "show", "tell",
    "the", "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "with", "would", "you",
}


def terms(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", re.sub(r"site:\S+", "", text.lower()))
    result = []
    for word in words:
        if len(word) <= 2 or word in STOP_WORDS:
            continue
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 5 and word.endswith("s") and not word.endswith(("ss", "is", "us")):
            word = word[:-1]
        result.append(word)
    return list(dict.fromkeys(result))


def topic_terms(query: str) -> List[str]:
    """Remove question framing while keeping the subject of the search."""
    framing = {"work", "works", "cause", "caused", "causes", "compare", "comparison",
               "difference", "different", "explain", "mean", "means", "happen", "happened"}
    return [term for term in terms(query) if term not in framing]


def coverage(query_terms: List[str], text: str) -> float:
    if not query_terms:
        return 0.0
    matches = set(query_terms) & set(terms(text))
    minimum = 1 if len(query_terms) == 1 else 2
    if len(matches) < minimum:
        return 0.0
    ratio = len(matches) / len(set(query_terms))
    return ratio if ratio >= 0.5 else 0.0


def best_sentence(query: str, passage: str) -> str:
    """Return a complete, on-topic sentence; do not fabricate a synthesis."""
    query_terms = topic_terms(query)
    sentence_terms = [term for term in query_terms if not term.isdigit()] or query_terms
    lowered = query.lower()
    is_causal = bool(re.search(r"\b(why|cause|caused|causes|reason)\b", lowered))
    is_process = bool(re.search(r"\bhow\s+(?:does|do|did)\b|\bwork(?:s)?\b", lowered))
    is_comparison = bool(re.search(r"\b(compare|comparison|versus|vs|difference)\b", lowered))
    candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", passage.strip())
    scored = []
    for position, sentence in enumerate(candidates):
        sentence = sentence.strip().removesuffix("...").strip()
        if not sentence or not re.search(r"[.!?]$", sentence):
            continue
        if is_comparison and not coverage(query_terms, sentence):
            continue
        overlap = len(set(sentence_terms) & set(terms(sentence)))
        if not overlap:
            continue
        score = overlap / len(sentence_terms)
        causal_cue = re.search(r"\b(caus\w*|trigger\w*|because|due|led|result\w*|collaps\w*|began|phase)\b", sentence, re.I)
        process_cue = re.search(r"\b(convert\w*|through|using|by|when|process|produc\w*|use\w*)\b", sentence, re.I)
        if is_causal and not causal_cue or is_process and not process_cue:
            continue
        if causal_cue and is_causal:
            score += 0.85
        if process_cue and is_process:
            score += 0.6
        scored.append((score, -position, sentence))
    return max(scored)[2] if scored else ""
