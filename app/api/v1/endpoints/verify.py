import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.deps import load_owned_session, require_active_session
from app.core.images import read_frames, read_image
from app.core.security import verify_api_key
from app.core.url_safety import UnsafeURLError, validate_callback_url
from app.schemas.session import SessionStatus
from app.schemas.verification import FullVerificationResponse, VerificationVerdict
from app.services.audit_service import audit_service
from app.services.session_store import session_store
from app.services.verification_pipeline import POLICY_VERSION, decide, overall_confidence, run_pipeline
from app.services.webhook_service import webhook_service

router = APIRouter()

_SESSION_STATUS_FOR_VERDICT = {
    VerificationVerdict.APPROVED: SessionStatus.VERIFIED,
    VerificationVerdict.REVIEW_REQUIRED: SessionStatus.PENDING_REVIEW,
    VerificationVerdict.REJECTED: SessionStatus.FAILED,
}

_SUMMARY = {
    VerificationVerdict.APPROVED: "All checks passed.",
    VerificationVerdict.REVIEW_REQUIRED: "No hard check failed, but some signals need human review.",
    VerificationVerdict.REJECTED: "One or more mandatory checks failed.",
}


@router.post("/full", response_model=FullVerificationResponse)
async def full_vkyc_verification(
    background_tasks: BackgroundTasks,
    session_id: str = Form(..., description="Active VKYC session ID"),
    id_document: UploadFile = File(..., description="ID document image (Aadhaar, PAN, passport, ...)"),
    live_frames: List[UploadFile] = File(
        ...,
        description="Ordered live frames covering the assigned challenges (start with a few "
        "frontal, neutral frames). At least 3 frames with a visible face are required.",
    ),
    extracted_ocr_text: Optional[str] = Form(
        None, max_length=10_000, description="Document text from your own OCR (used when server OCR is unavailable)"
    ),
    callback_url: Optional[str] = Form(None, description="HTTPS webhook URL (overrides the session's)"),
    key_id: str = Depends(verify_api_key),
):
    """
    Runs the full pipeline: document checks, face detection, active and passive
    liveness, 1:1 face match, synthetic-media heuristics, then writes a signed,
    hash-chained audit dossier. Each session can be verified once.
    """
    session = load_owned_session(session_id, key_id)
    require_active_session(session)

    try:
        webhook_url = validate_callback_url(callback_url) or session.callback_url
    except UnsafeURLError as exc:
        raise HTTPException(422, detail=str(exc))

    document_bgr = await read_image(id_document, "id_document")
    frames_bgr = await read_frames(live_frames, "live_frames")

    if not session_store.claim(session_id):
        raise HTTPException(status.HTTP_409_CONFLICT, detail=f"Session {session_id} is already being verified.")

    try:
        result = await run_in_threadpool(
            run_pipeline,
            document_bgr,
            frames_bgr,
            session.challenges,
            extracted_ocr_text,
            session.document_type,
        )
    except Exception:
        # Let the client retry this session if the pipeline itself crashed.
        session_store.release(session_id)
        raise

    verdict, rejection_reasons, review_reasons = decide(
        result.face_match,
        result.liveness,
        result.deepfake,
        result.document,
        identity_consistent=result.identity_consistent,
        max_faces_in_frames=result.max_faces_in_frames,
    )

    verification_id = f"verif_{uuid.uuid4().hex[:24]}"
    dossier = await run_in_threadpool(
        lambda: audit_service.create_and_store_dossier(
            verification_id=verification_id,
            session_id=session_id,
            user_id=session.user_id,
            owner_key_id=key_id,
            verdict=verdict,
            policy_version=POLICY_VERSION,
            overall_confidence=overall_confidence(result),
            frames_analyzed=result.frames_analyzed,
            face_match=result.face_match,
            liveness=result.liveness,
            deepfake=result.deepfake,
            document=result.document,
            rejection_reasons=rejection_reasons,
            review_reasons=review_reasons,
        )
    )

    session.status = _SESSION_STATUS_FOR_VERDICT[verdict]
    session.verification_id = verification_id
    session.challenges = result.liveness.completed_challenges
    session_store.save(session)

    if webhook_url:
        background_tasks.add_task(
            webhook_service.send_verification_callback,
            callback_url=webhook_url,
            payload=dossier.model_dump(mode="json"),
        )

    reasons = rejection_reasons or review_reasons
    summary = _SUMMARY[verdict] + (f" Reasons: {'; '.join(reasons)}" if reasons else "")
    return FullVerificationResponse(
        success=verdict == VerificationVerdict.APPROVED,
        verification_id=verification_id,
        verdict=verdict,
        summary=summary,
        dossier=dossier,
    )
