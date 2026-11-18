import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.config import settings
from app.core.security import verify_api_key
from app.schemas.session import (
    ChallengeStep,
    ChallengeType,
    CreateSessionRequest,
    SessionResponse,
    SessionStatus,
)

router = APIRouter()

# In-memory session store (can be replaced with Redis in production)
active_sessions: Dict[str, SessionResponse] = {}


@router.post("/", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_vkyc_session(
    payload: CreateSessionRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Initializes a new Video KYC verification session with dynamic anti-spoofing challenges.
    """
    session_id = f"vkyc_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=settings.SESSION_EXPIRY_SECONDS)

    # Dynamic randomized active challenges
    available_challenges = [
        (ChallengeType.TURN_HEAD_LEFT, "Slowly turn your head to the left"),
        (ChallengeType.TURN_HEAD_RIGHT, "Slowly turn your head to the right"),
        (ChallengeType.BLINK_EYES, "Blink your eyes twice naturally"),
        (ChallengeType.SMILE, "Smile gently towards the camera"),
    ]
    selected = random.sample(available_challenges, k=min(settings.REQUIRED_ACTIVE_STEPS, len(available_challenges)))

    challenge_steps: List[ChallengeStep] = []
    for idx, (ctype, desc) in enumerate(selected, start=1):
        challenge_steps.append(
            ChallengeStep(
                step_id=idx,
                challenge_type=ctype,
                instruction=desc,
                time_limit_seconds=15,
                is_completed=False,
            )
        )

    # 4-digit verification code for audio-visual challenge
    verif_code = f"{random.randint(1000, 9999)}"

    session = SessionResponse(
        session_id=session_id,
        user_id=payload.user_id,
        status=SessionStatus.INITIALIZED,
        created_at=now,
        expires_at=expires_at,
        challenges=challenge_steps,
        verification_code=verif_code,
        upload_url=f"/api/v1/verify/full",
    )

    active_sessions[session_id] = session
    return session


@router.get("/{session_id}", response_model=SessionResponse)
def get_vkyc_session(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Retrieves the current state and challenge steps of a VKYC session.
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired.",
        )
    return session
