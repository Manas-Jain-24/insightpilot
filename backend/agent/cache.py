"""Bounded, process-local TTL cache; values are isolated from caller mutation."""
from collections import OrderedDict
from copy import deepcopy
import threading
import time


class TTLCache:
    def __init__(self, max_entries=128, ttl=900, clock=time.monotonic):
        self.max_entries, self.ttl, self.clock = max_entries, ttl, clock
        self._values = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key):
        with self._lock:
            entry = self._values.get(key)
            if not entry:
                return None
            expires, value = entry
            if expires <= self.clock():
                del self._values[key]
                return None
            self._values.move_to_end(key)
            return deepcopy(value)

    def put(self, key, value):
        with self._lock:
            self._values[key] = (self.clock() + self.ttl, deepcopy(value))
            self._values.move_to_end(key)
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
