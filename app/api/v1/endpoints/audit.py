from fastapi import APIRouter, Depends, HTTPException, status
from app.core.security import verify_api_key, verify_hmac_signature
from app.schemas.verification import VerificationDossier
from app.services.audit_service import audit_service

router = APIRouter()


@router.get("/{verification_id}", response_model=VerificationDossier)
def get_audit_dossier(
    verification_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Retrieves the tamper-evident cryptographic audit dossier for a past verification.
    Validates HMAC digital signature to ensure zero post-hoc modification.
    """
    dossier = audit_service.get_dossier(verification_id)
    if not dossier:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audit dossier {verification_id} not found.",
        )

    # Cryptographic integrity check
    is_valid_sig = verify_hmac_signature(
        dossier.tamper_proof_audit_hash, dossier.signature
    )
    if not is_valid_sig:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Tamper detection alert: Dossier cryptographic signature mismatch!",
        )

    return dossier
