import logging

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.core.security import verify_api_key
from app.schemas.verification import VerificationDossier
from app.services.audit_service import AuditIntegrityError, audit_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{verification_id}", response_model=VerificationDossier)
async def get_audit_dossier(
    verification_id: str,
    key_id: str = Depends(verify_api_key),
):
    """
    Returns one of your audit dossiers after recomputing its hash from the
    stored content and checking its HMAC signature. Any edit to the stored
    record returns 409.
    """
    try:
        dossier = await run_in_threadpool(audit_service.get_dossier, verification_id, key_id)
    except AuditIntegrityError as exc:
        logger.error("Audit integrity failure for %s: %s", verification_id, exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tamper detection alert: stored dossier failed integrity verification.",
        )
    if dossier is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Audit dossier {verification_id} not found.")
    return dossier
