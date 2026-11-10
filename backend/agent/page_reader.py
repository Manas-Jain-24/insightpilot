"""Bounded, public-URL-only extraction for cited web pages."""

import asyncio
import ipaddress
import re
import socket
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from backend.agent.relevance import topic_terms


class _Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.depth = 0
        self.buffer = []
        self.paragraphs = []
        self.published_at = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "meta" and (values.get("property") or values.get("name")) in {
            "article:published_time", "datePublished", "date", "pubdate"
        }:
            try:
                self.published_at = datetime.fromisoformat(values.get("content", "").replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                pass
        if tag in {"script", "style", "nav", "footer", "header", "noscript"}:
            self.skip += 1
        if tag in {"p", "h1", "h2", "h3", "article"}:
            self.depth += 1
            if self.depth == 1:
                self.buffer = []

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "header", "noscript"}:
            self.skip = max(0, self.skip - 1)
        if tag in {"p", "h1", "h2", "h3", "article"} and self.depth:
            self.depth -= 1
            if self.depth == 0:
                paragraph = re.sub(r"\s+", " ", " ".join(self.buffer)).strip()
                if len(paragraph) >= 40:
                    self.paragraphs.append(paragraph)

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.buffer.append(data)


def _public_https_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        return False
    try:
        if parsed.port not in {None, 443}:
            return False
    except ValueError:
        return False
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".local", ".internal")):
        return False
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(item[4][0]).is_global for item in addresses)
    except (OSError, ValueError):
        return False


async def fetch_page_excerpt(url: str, query: str, metadata=None) -> str:
    """Return one on-topic passage, or empty text when a page cannot be read safely."""
    if not await asyncio.to_thread(_public_https_url, url):
        return ""
    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=False) as client:
            async with client.stream("GET", url, headers={"User-Agent": "InsightPilot/1.0"}) as response:
                if response.status_code != 200 or "text/html" not in response.headers.get("content-type", ""):
                    return ""
                chunks = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > 400_000:
                        break
                    chunks.append(chunk)
                html = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
    except (httpx.HTTPError, UnicodeError):
        return ""
    parser = _Paragraphs()
    try:
        parser.feed(html)
    except (ValueError, AssertionError):
        return ""
    if metadata is not None:
        metadata["published_at"] = parser.published_at
    wanted = set(topic_terms(query))
    ranked = []
    for paragraph in parser.paragraphs:
        overlap = len(wanted & set(topic_terms(paragraph)))
        if overlap:
            ranked.append((overlap / max(len(wanted), 1), min(len(paragraph), 1200), paragraph))
    if not ranked:
        return ""
    return max(ranked)[2][:1200]
