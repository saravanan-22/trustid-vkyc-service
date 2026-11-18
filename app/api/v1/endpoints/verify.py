import asyncio
import uuid
import cv2
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from app.core.config import settings
from app.core.security import verify_api_key
from app.schemas.session import SessionStatus
from app.schemas.verification import (
    FullVerificationResponse,
    VerificationVerdict,
)
from app.services.audit_service import audit_service
from app.services.deepfake_detector import deepfake_detector
from app.services.face_matcher import face_matcher
from app.services.liveness_service import liveness_service
from app.services.ocr_service import ocr_service
from app.services.webhook_service import webhook_service
from app.api.v1.endpoints.sessions import active_sessions

router = APIRouter()


def decode_image_bytes(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image frame.",
        )
    return img


@router.post("/full", response_model=FullVerificationResponse)
async def full_vkyc_verification(
    background_tasks: BackgroundTasks,
    session_id: str = Form(..., description="Active VKYC Session ID"),
    id_document: UploadFile = File(..., description="ID Document image (Aadhaar, Passport, etc.)"),
    live_frames: List[UploadFile] = File(
        ..., description="One or more live video frames captured during challenge execution"
    ),
    extracted_ocr_text: Optional[str] = Form(None, description="Pre-extracted text from OCR pipeline"),
    callback_url: Optional[str] = Form(None, description="Webhook URL for async result notification"),
    api_key: str = Depends(verify_api_key),
):
    """
    Executes the end-to-end Video KYC verification pipeline:
    1. Document OCR, Quality & Photo Extraction
    2. Deepfake & Virtual Camera Detection
    3. Active & Passive Liveness Verification
    4. 1:1 Facial Biometric Matching
    5. Tamper-Proof Cryptographic Audit Generation
    6. Signed Webhook Notification
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired.",
        )

    # 1. Decode Document
    doc_bytes = await id_document.read()
    doc_bgr = decode_image_bytes(doc_bytes)

    # 2. Decode Live Frames
    decoded_live_frames: List[np.ndarray] = []
    for f in live_frames:
        content = await f.read()
        decoded_live_frames.append(decode_image_bytes(content))

    if not decoded_live_frames:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one live frame is required.",
        )

    # Step A: Document OCR & Validation
    doc_result = ocr_service.verify_document(doc_bgr, provided_text=extracted_ocr_text)
    doc_photo_bgr = ocr_service.extract_document_photo(doc_bgr)

    # Step B: Deepfake & Virtual Camera Detection on Live Frames
    primary_live_frame = decoded_live_frames[len(decoded_live_frames) // 2]
    deepfake_metrics = deepfake_detector.evaluate_frame(primary_live_frame)

    # Step C: Liveness & Anti-Spoofing
    passive_liveness = liveness_service.evaluate_live_frame(primary_live_frame)
    active_passed, updated_challenges = liveness_service.verify_active_challenge_sequence(
        decoded_live_frames, session.challenges
    )

    passive_liveness.active_challenges_passed = active_passed
    passive_liveness.completed_challenges = updated_challenges

    # Step D: 1:1 Facial Biometric Match
    face_match_metrics = face_matcher.compare_faces(doc_photo_bgr, primary_live_frame)

    # Step E: Decision Policy Engine
    rejection_reasons = []

    if deepfake_metrics.is_deepfake_suspected:
        rejection_reasons.append("HIGH_RISK_DEEPFAKE_OR_SYNTHETIC_MEDIA_DETECTED")
    if deepfake_metrics.virtual_camera_detected:
        rejection_reasons.append("VIRTUAL_CAMERA_OR_INJECTION_DEVICE_DETECTED")

    if not passive_liveness.is_live_passive:
        rejection_reasons.append(f"PASSIVE_LIVENESS_FAILED: {passive_liveness.detected_spoof_type}")

    if not face_match_metrics.is_match:
        rejection_reasons.append(
            f"BIOMETRIC_FACE_MISMATCH: Similarity {face_match_metrics.match_score:.2f} below threshold {settings.FACE_MATCH_THRESHOLD}"
        )

    if not doc_result.is_valid:
        rejection_reasons.append(f"DOCUMENT_INTEGRITY_FAILED: {', '.join(doc_result.tamper_flags)}")

    # Calculate overall confidence
    scores = [
        face_match_metrics.match_score,
        passive_liveness.passive_score,
        (1.0 - deepfake_metrics.synthetic_risk_score),
        doc_result.confidence,
    ]
    overall_confidence = round(float(np.mean(scores)), 4)

    # Determine Verdict
    if len(rejection_reasons) == 0:
        verdict = VerificationVerdict.APPROVED
        summary = "Verification completed successfully. Customer identity and liveness confirmed."
        session.status = SessionStatus.VERIFIED
    elif len(rejection_reasons) == 1 and ("DOCUMENT_INTEGRITY_FAILED" in rejection_reasons[0] or not active_passed):
        verdict = VerificationVerdict.REVIEW_REQUIRED
        summary = "Automated verification flagged minor anomalies requiring human compliance review."
        session.status = SessionStatus.IN_PROGRESS
    else:
        verdict = VerificationVerdict.REJECTED
        summary = f"Verification rejected: {'; '.join(rejection_reasons)}"
        session.status = SessionStatus.FAILED

    # Step F: Cryptographic Tamper-Proof Audit Dossier
    verification_id = f"verif_{uuid.uuid4().hex[:12]}"
    dossier = audit_service.create_and_store_dossier(
        verification_id=verification_id,
        session_id=session_id,
        user_id=session.user_id,
        verdict=verdict,
        overall_confidence=overall_confidence,
        face_match=face_match_metrics,
        liveness=passive_liveness,
        deepfake=deepfake_metrics,
        document=doc_result,
        rejection_reasons=rejection_reasons,
    )

    # Step G: Webhook Dispatcher
    dest_url = callback_url or session_id
    if callback_url:
        background_tasks.add_task(
            webhook_service.send_verification_callback,
            callback_url=callback_url,
            payload=dossier.model_dump(),
        )

    return FullVerificationResponse(
        success=(verdict == VerificationVerdict.APPROVED),
        verification_id=verification_id,
        verdict=verdict,
        summary=summary,
        dossier=dossier,
    )
