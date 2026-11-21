from datetime import datetime, timezone

from fastapi import HTTPException, status

from app.schemas.session import TERMINAL_SESSION_STATUSES, SessionRecord, SessionStatus
from app.services.session_store import session_store


def load_owned_session(session_id: str, owner_key_id: str) -> SessionRecord:
    """
    Returns the caller's session. Sessions belonging to another API key are
    reported as not found, so their existence is not revealed.
    """
    record = session_store.get(session_id)
    if record is None or record.owner_key_id != owner_key_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found.")
    if record.status not in TERMINAL_SESSION_STATUSES and record.expires_at <= datetime.now(timezone.utc):
        record.status = SessionStatus.EXPIRED
        session_store.save(record)
    return record


def require_active_session(record: SessionRecord) -> None:
    if record.status == SessionStatus.EXPIRED:
        raise HTTPException(status.HTTP_410_GONE, detail=f"Session {record.session_id} has expired.")
    if record.status != SessionStatus.INITIALIZED:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Session {record.session_id} is {record.status.value}; create a new session.",
        )
