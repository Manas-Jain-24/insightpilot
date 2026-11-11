"""Conservative checks for common missing-fact and missing-subject questions.

These checks are deterministic heuristics, not an entailment classifier. They do
not establish source truth, and deliberately defer uncertain cases to abstention.
"""

import re

from backend.agent.relevance import terms, topic_terms


def answer_parts(query):
    """Split explicit question clauses; do not split subjects such as A and B."""
    parts = re.split(r"\s+and\s+(?=(?:what|how|when|where|why|which|who)\b)", query, flags=re.I)
    return [part.strip(" ?. ") for part in parts if part.strip()][:4]


def missing_fact(query, citations):
    text = " ".join(c.snippet for c in citations)
    # Topic similarity cannot supply a price, temperature or attribution absent
    # from the evidence. Apply each rule to the requested fact, not its topic.
    requirements = [
        (r"\b(price|cost|fee)\b|how much.*(?:pay|spend|charge)",
         r"\b(price|cost|fee|free|usd|eur|dollars?|rupees?)\b|[$€₹]"),
        (r"\btemperature\b", r"\b(temperature|celsius|fahrenheit|degrees?|kelvin)\b|°"),
    ]
    for question_pattern, source_pattern in requirements:
        if re.search(question_pattern, query, re.I) and not re.search(source_pattern, text, re.I):
            return True
    attribution = re.search(r"\bwho\b.*\b(found|invent|author|creat|establish)\w*", query, re.I)
    if attribution and not re.search(r"\b" + attribution.group(1) + r"\w*", text, re.I):
        return True
    return False


def needs_subject(query, citations):
    if re.search(r"\b(compare|versus|vs|difference)\b", query, re.I):
        return False
    wanted = set(topic_terms(query)) - {"long", "last", "need", "expire", "expiry", "limit"}
    if not wanted:
        return True
    # Restrict ambiguity checks to short requests for a scoped value. Broad
    # explanatory questions can appropriately draw on several documents.
    if len(wanted) > 3 or not re.search(r"\b(when|long|limit|code|expire|expiry)\b", query, re.I):
        return False
    titles = {}
    for citation in citations:
        if citation.doc_id and wanted & set(terms(citation.snippet)):
            name = re.sub(r"\s*\(Page \d+\)|\.(txt|md|pdf)\b", "", citation.title, flags=re.I)
            titles.setdefault(citation.doc_id, set(terms(name)))
    if len(titles) < 2:
        return False
    for doc_id, title in titles.items():
        others = set().union(*(value for key, value in titles.items() if key != doc_id))
        if wanted & (title - others):
            return False
    return True
