from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.session import SessionRecord, SessionStatus
from app.services.session_store import InMemorySessionStore, RedisSessionStore


def record(sid="vkyc_test"):
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id=sid, user_id="u", status=SessionStatus.INITIALIZED, created_at=now,
        expires_at=now + timedelta(minutes=5), challenges=[], verification_code="1234",
        upload_url="/api/v1/verify/full", owner_key_id="owner",
    )


def redis_store():
    fakeredis = pytest.importorskip("fakeredis")
    store = RedisSessionStore.__new__(RedisSessionStore)
    store._redis = fakeredis.FakeRedis(decode_responses=True)
    return store


@pytest.fixture(params=["memory", "redis"])
def store(request):
    return InMemorySessionStore() if request.param == "memory" else redis_store()


def test_save_and_get_round_trip(store):
    store.save(record())
    loaded = store.get("vkyc_test")
    assert loaded is not None and loaded.owner_key_id == "owner"
    assert store.get("missing") is None


def test_claim_is_single_use(store):
    store.save(record())
    assert store.claim("vkyc_test") is True
    assert store.claim("vkyc_test") is False
    assert store.get("vkyc_test").status == SessionStatus.IN_PROGRESS


def test_release_allows_retry(store):
    store.save(record())
    assert store.claim("vkyc_test")
    store.release("vkyc_test")
    assert store.get("vkyc_test").status == SessionStatus.INITIALIZED
    assert store.claim("vkyc_test") is True


def test_claim_unknown_session_fails(store):
    assert store.claim("nope") is False


def test_returned_records_are_copies():
    store = InMemorySessionStore()
    store.save(record())
    loaded = store.get("vkyc_test")
    loaded.status = SessionStatus.FAILED
    assert store.get("vkyc_test").status == SessionStatus.INITIALIZED
