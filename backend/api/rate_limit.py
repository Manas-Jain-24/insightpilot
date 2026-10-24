"""Small single-process sliding-window limiter for the public demo."""

import threading
import time
from collections import defaultdict, deque


class DemoRateLimiter:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int) -> bool:
        return self.allow_many([(key, limit)])

    def allow_many(self, rules: list[tuple[str, int]]) -> bool:
        now = time.monotonic()
        with self._lock:
            for key, limit in rules:
                hits = self._hits[key]
                while hits and hits[0] <= now - self.window_seconds:
                    hits.popleft()
                if len(hits) >= limit:
                    return False
            for key, _ in rules:
                self._hits[key].append(now)
            if len(self._hits) > 1000:
                for key, hits in list(self._hits.items()):
                    if not hits or hits[-1] <= now - self.window_seconds:
                        del self._hits[key]
            return True
