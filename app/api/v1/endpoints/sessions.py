import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import load_owned_session
from app.core.config import settings
from app.core.security import verify_api_key
from app.core.url_safety import UnsafeURLError, validate_callback_url
from app.schemas.session import (
    ChallengeStep,
    ChallengeType,
    CreateSessionRequest,
    SessionRecord,
    SessionResponse,
    SessionStatus,
)
from app.services.session_store import session_store

router = APIRouter()

AVAILABLE_CHALLENGES = [
    (ChallengeType.TURN_HEAD_LEFT, "Look straight at the camera, then slowly turn your head to your left"),
    (ChallengeType.TURN_HEAD_RIGHT, "Look straight at the camera, then slowly turn your head to your right"),
    (ChallengeType.BLINK_EYES, "Look at the camera and blink naturally"),
    (ChallengeType.SMILE, "Start with a neutral face, then smile"),
]


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_vkyc_session(
    payload: CreateSessionRequest,
    key_id: str = Depends(verify_api_key),
):
    """
    Starts a VKYC session with randomized active liveness challenges. The
    session is single-use, expires after SESSION_EXPIRY_SECONDS, and is visible
    only to the API key that created it.
    """
    try:
        callback_url = validate_callback_url(payload.callback_url)
    except UnsafeURLError as exc:
        raise HTTPException(422, detail=str(exc))

    rng = secrets.SystemRandom()
    count = min(settings.REQUIRED_ACTIVE_STEPS, len(AVAILABLE_CHALLENGES))
    selected = rng.sample(AVAILABLE_CHALLENGES, k=count)
    challenges: List[ChallengeStep] = [
        ChallengeStep(step_id=i, challenge_type=ctype, instruction=text, time_limit_seconds=15)
        for i, (ctype, text) in enumerate(selected, start=1)
    ]

    now = datetime.now(timezone.utc)
    record = SessionRecord(
        session_id=f"vkyc_{uuid.uuid4().hex}",
        user_id=payload.user_id,
        status=SessionStatus.INITIALIZED,
        created_at=now,
        expires_at=now + timedelta(seconds=settings.SESSION_EXPIRY_SECONDS),
        challenges=challenges,
        verification_code=f"{secrets.randbelow(9000) + 1000}",
        upload_url=f"{settings.API_V1_STR}/verify/full",
        owner_key_id=key_id,
        callback_url=callback_url,
        document_type=payload.document_type,
        metadata=payload.metadata,
    )
    session_store.save(record)
    return record.to_response()


@router.get("/{session_id}", response_model=SessionResponse)
def get_vkyc_session(
    session_id: str,
    key_id: str = Depends(verify_api_key),
):
    """Returns the current state of one of your sessions."""
    return load_owned_session(session_id, key_id).to_response()
