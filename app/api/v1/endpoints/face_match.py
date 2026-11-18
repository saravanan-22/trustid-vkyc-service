import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from app.core.security import verify_api_key
from app.schemas.verification import FaceMatchMetrics
from app.services.face_matcher import face_matcher

router = APIRouter()


def decode_image_bytes(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format.",
        )
    return img


@router.post("/match", response_model=FaceMatchMetrics)
async def match_faces(
    id_card_image: UploadFile = File(..., description="ID card containing applicant portrait"),
    live_selfie_image: UploadFile = File(..., description="Live captured selfie or video keyframe"),
    api_key: str = Depends(verify_api_key),
):
    """
    Performs 1:1 facial biometric matching between ID card portrait and live camera capture.
    """
    id_bytes = await id_card_image.read()
    live_bytes = await live_selfie_image.read()

    id_bgr = decode_image_bytes(id_bytes)
    live_bgr = decode_image_bytes(live_bytes)

    metrics = face_matcher.compare_faces(id_bgr, live_bgr)
    return metrics
