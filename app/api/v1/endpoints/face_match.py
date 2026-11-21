from fastapi import APIRouter, Depends, File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.images import read_image
from app.core.security import verify_api_key
from app.schemas.verification import FaceMatchMetrics
from app.services.face_matcher import face_matcher

router = APIRouter()


@router.post("/match", response_model=FaceMatchMetrics)
async def match_faces(
    id_card_image: UploadFile = File(..., description="ID card containing applicant portrait"),
    live_selfie_image: UploadFile = File(..., description="Live captured selfie or video keyframe"),
    key_id: str = Depends(verify_api_key),
):
    """1:1 face verification (SFace cosine similarity) between an ID portrait and a live capture."""
    id_bgr = await read_image(id_card_image, "id_card_image")
    live_bgr = await read_image(live_selfie_image, "live_selfie_image")
    return await run_in_threadpool(face_matcher.compare_faces, id_bgr, live_bgr)
