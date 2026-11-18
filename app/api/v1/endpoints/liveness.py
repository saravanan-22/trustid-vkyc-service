import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.core.security import verify_api_key
from app.schemas.verification import LivenessMetrics
from app.services.liveness_service import liveness_service
from app.api.v1.endpoints.sessions import active_sessions

router = APIRouter()


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decodes binary image bytes into OpenCV BGR numpy array."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Could not decode frame.",
        )
    return img


@router.post("/passive", response_model=LivenessMetrics)
async def check_passive_liveness(
    frame: UploadFile = File(..., description="Single live camera frame"),
    api_key: str = Depends(verify_api_key),
):
    """
    Evaluates a single camera frame for passive liveness, Moiré patterns, screen replay, or paper mask spoofing.
    """
    content = await frame.read()
    img_bgr = decode_image_bytes(content)
    metrics = liveness_service.evaluate_live_frame(img_bgr)
    return metrics


@router.post("/active", response_model=LivenessMetrics)
async def check_active_liveness(
    session_id: str = Form(..., description="Active KYC session ID"),
    frames: List[UploadFile] = File(..., description="Ordered keyframes during challenge execution"),
    api_key: str = Depends(verify_api_key),
):
    """
    Evaluates ordered video keyframes against the assigned challenges (Head turns, smiles, blinks).
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found.",
        )

    decoded_frames: List[np.ndarray] = []
    for f in frames:
        content = await f.read()
        decoded_frames.append(decode_image_bytes(content))

    all_passed, updated_steps = liveness_service.verify_active_challenge_sequence(
        decoded_frames, session.challenges
    )

    # Passive score on middle frame
    middle_idx = len(decoded_frames) // 2
    passive_metrics = liveness_service.evaluate_live_frame(decoded_frames[middle_idx])

    return LivenessMetrics(
        passive_score=passive_metrics.passive_score,
        is_live_passive=passive_metrics.is_live_passive,
        active_challenges_passed=all_passed,
        completed_challenges=updated_steps,
        detected_spoof_type=passive_metrics.detected_spoof_type,
    )
