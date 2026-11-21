from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import load_owned_session
from app.core.images import read_frames, read_image
from app.core.security import verify_api_key
from app.schemas.session import SessionStatus
from app.schemas.verification import LivenessMetrics
from app.services.liveness_service import liveness_service

router = APIRouter()


@router.post("/passive", response_model=LivenessMetrics)
async def check_passive_liveness(
    frame: UploadFile = File(..., description="Single live camera frame"),
    key_id: str = Depends(verify_api_key),
):
    """
    Heuristic single-frame check (blur, glare, skin chroma) on the detected face.
    Not a certified presentation-attack-detection model.
    """
    img = await read_image(frame, "frame")
    return await run_in_threadpool(liveness_service.evaluate_live_frame, img)


def _evaluate_active(frames, challenges) -> LivenessMetrics:
    faces_per_frame = [liveness_service.engine.detect_faces(f) for f in frames]
    observations = [liveness_service.observe(f, faces) for f, faces in zip(frames, faces_per_frame)]
    passed, steps = liveness_service.evaluate_challenges(observations, challenges)
    middle = len(frames) // 2
    passive = liveness_service.evaluate_live_frame(frames[middle], faces_per_frame[middle])
    return passive.model_copy(update={"active_challenges_passed": passed, "completed_challenges": steps})


@router.post("/active", response_model=LivenessMetrics)
async def check_active_liveness(
    session_id: str = Form(..., description="Active KYC session ID"),
    frames: List[UploadFile] = File(..., description="Ordered keyframes during challenge execution"),
    key_id: str = Depends(verify_api_key),
):
    """
    Dry-run of the session's active challenges against ordered frames. Does not
    consume the session; the authoritative check runs in /verify/full.
    """
    session = load_owned_session(session_id, key_id)
    if session.status == SessionStatus.EXPIRED:
        raise HTTPException(status.HTTP_410_GONE, detail=f"Session {session_id} has expired.")
    decoded = await read_frames(frames, "frames")
    return await run_in_threadpool(_evaluate_active, decoded, session.challenges)
