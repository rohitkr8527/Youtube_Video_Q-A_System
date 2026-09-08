from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

from app.config import get_settings


class Cache:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._memory: dict[str, tuple[float, str]] = {}
        self._redis = None
        if self.settings.redis_url:
            try:
                import redis

                client = redis.from_url(self.settings.redis_url, decode_responses=True, socket_timeout=1)
                client.ping()
                self._redis = client
            except Exception:
                self._redis = None

    def get_json(self, key: str) -> Any | None:
        if self._redis is not None:
            value = self._redis.get(key)
            return json.loads(value) if value else None
        item = self._memory.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at < time.time():
            self._memory.pop(key, None)
            return None
        return json.loads(value)

    def set_json(self, key: str, value: Any, ttl_seconds: int = 1800) -> None:
        encoded = json.dumps(value)
        if self._redis is not None:
            self._redis.setex(key, ttl_seconds, encoded)
            return
        self._memory[key] = (time.time() + ttl_seconds, encoded)


@lru_cache(maxsize=1)
def get_cache() -> Cache:
    return Cache()
