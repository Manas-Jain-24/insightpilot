"""Small, deterministic decomposition for questions requiring separate evidence."""

import re
from typing import List


def research_queries(query: str) -> List[str]:
    """Split a two-subject comparison; keep other questions intact."""
    value = query.strip().rstrip("?. ")
    match = re.match(
        r"(?i)^compare\s+(.+?)\s+(?:and|versus|vs\.?|with)\s+(.+?)(?:\s+(?:for|in|on)\s+(.+))?$",
        value,
    )
    if not match:
        match = re.match(r"(?i)^(.+?)\s+(?:versus|vs\.?)\s+(.+?)(?:\s+(?:for|in|on)\s+(.+))?$", value)
    if not match:
        return [query]
    left, right, context = (part.strip() if part else "" for part in match.groups())
    if not left or not right or len(left) > 100 or len(right) > 100:
        return [query]
    suffix = f" {context}" if context else ""
    return [f"{left}{suffix}", f"{right}{suffix}"]
