"""
Cache Agent
-----------
LRU in-memory cache for LLM predictions.

Why: The LLM is the slowest step (~500ms-2s per call). Many service desk
tickets are near-identical ("password reset", "VPN not connecting"). Caching
at the preprocessed-text level gives a large speed win for repeated patterns.

Features:
  - LRU eviction (configurable max size)
  - TTL expiry per entry (default: 1 hour)
  - Cache stats (hit rate, saved tokens)
  - Thread-safe

Usage:
    from agents.cache_agent import PredictionCache
    cache = PredictionCache(maxsize=512, ttl_seconds=3600)

    key   = cache.make_key(clean_text)
    result = cache.get(key)
    if result is None:
        result = run_full_pipeline(clean_text)
        cache.set(key, result)
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class PredictionCache:

    def __init__(self, maxsize: int = 512, ttl_seconds: float = 3600.0):
        self._maxsize      = maxsize
        self._ttl          = ttl_seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock         = threading.Lock()
        self._hits         = 0
        self._misses       = 0
        self._saved_tokens = 0

    def make_key(self, text: str) -> str:
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            value, expires_at = self._store[key]
            if time.time() > expires_at:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (LRU refresh)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Cache HIT  key=%s", key)
            return value

    def set(self, key: str, value: Any, token_count: int = 0):
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time() + self._ttl)
            self._saved_tokens += token_count
            if len(self._store) > self._maxsize:
                evicted = self._store.popitem(last=False)
                logger.debug("Cache evicted key=%s", evicted[0])

    def invalidate(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def clear(self):
        with self._lock:
            self._store.clear()
            self._hits = self._misses = self._saved_tokens = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "size":         self.size,
            "maxsize":      self._maxsize,
            "hits":         self._hits,
            "misses":       self._misses,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
            "saved_tokens": self._saved_tokens,
        }
