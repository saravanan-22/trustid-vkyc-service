"""
Session storage.

* InMemorySessionStore: single process only (default for development/tests).
* RedisSessionStore: used when REDIS_URL is set; required when running more
  than one worker or replica, since every worker must see the same sessions.

`claim()` atomically moves a session from INITIALIZED to IN_PROGRESS, so the
same session cannot be verified twice (including by concurrent requests).
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Protocol

from app.core.config import settings
from app.schemas.session import SessionRecord, SessionStatus

logger = logging.getLogger(__name__)

# Keep finished sessions readable for a while after they expire.
RETENTION_AFTER_EXPIRY = timedelta(hours=24)


class SessionStore(Protocol):
    def save(self, record: SessionRecord) -> None: ...

    def get(self, session_id: str) -> Optional[SessionRecord]: ...

    def claim(self, session_id: str) -> bool: ...

    def release(self, session_id: str) -> None: ...


def _ttl_seconds(record: SessionRecord) -> int:
    remaining = record.expires_at + RETENTION_AFTER_EXPIRY - datetime.now(timezone.utc)
    return max(1, int(remaining.total_seconds()))


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: Dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def _evict_old(self) -> None:
        cutoff = datetime.now(timezone.utc) - RETENTION_AFTER_EXPIRY
        for sid in [sid for sid, r in self._data.items() if r.expires_at < cutoff]:
            del self._data[sid]

    def save(self, record: SessionRecord) -> None:
        with self._lock:
            self._evict_old()
            self._data[record.session_id] = record.model_copy(deep=True)

    def get(self, session_id: str) -> Optional[SessionRecord]:
        with self._lock:
            record = self._data.get(session_id)
            return record.model_copy(deep=True) if record else None

    def claim(self, session_id: str) -> bool:
        with self._lock:
            record = self._data.get(session_id)
            if record is None or record.status != SessionStatus.INITIALIZED:
                return False
            record.status = SessionStatus.IN_PROGRESS
            return True

    def release(self, session_id: str) -> None:
        """Undo a claim (used when the pipeline itself fails) so the client can retry."""
        with self._lock:
            record = self._data.get(session_id)
            if record is not None and record.status == SessionStatus.IN_PROGRESS:
                record.status = SessionStatus.INITIALIZED

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisSessionStore:
    _PREFIX = "trustid:session:"
    _CLAIM_PREFIX = "trustid:session-claim:"

    def __init__(self, url: str) -> None:
        import redis  # optional dependency

        self._redis = redis.Redis.from_url(url, decode_responses=True)

    def save(self, record: SessionRecord) -> None:
        self._redis.set(self._PREFIX + record.session_id, record.model_dump_json(), ex=_ttl_seconds(record))

    def get(self, session_id: str) -> Optional[SessionRecord]:
        raw = self._redis.get(self._PREFIX + session_id)
        return SessionRecord.model_validate_json(raw) if raw else None

    def claim(self, session_id: str) -> bool:
        record = self.get(session_id)
        if record is None or record.status != SessionStatus.INITIALIZED:
            return False
        # SET NX is atomic: only one request can ever win the claim.
        won = self._redis.set(self._CLAIM_PREFIX + session_id, "1", nx=True, ex=_ttl_seconds(record))
        if not won:
            return False
        record.status = SessionStatus.IN_PROGRESS
        self.save(record)
        return True

    def release(self, session_id: str) -> None:
        record = self.get(session_id)
        if record is not None and record.status == SessionStatus.IN_PROGRESS:
            record.status = SessionStatus.INITIALIZED
            self.save(record)
        self._redis.delete(self._CLAIM_PREFIX + session_id)

    def ping(self) -> bool:
        return bool(self._redis.ping())


def build_session_store() -> SessionStore:
    if settings.REDIS_URL:
        logger.info("Using Redis session store")
        return RedisSessionStore(settings.REDIS_URL)
    return InMemorySessionStore()


session_store: SessionStore = build_session_store()
