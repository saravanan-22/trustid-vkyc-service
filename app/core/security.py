import hashlib
import hmac
from typing import Optional

from fastapi import Header, HTTPException, status

from app.core.config import settings


def api_key_fingerprint(api_key: str) -> str:
    """Stable, non-reversible identifier for an API key (used to scope resources)."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Validates the `X-API-KEY` header against the configured keys and returns the
    caller's key fingerprint. Sessions and audit dossiers are scoped to it, so one
    API client cannot read another client's data.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide 'X-API-KEY' header.",
        )

    presented = x_api_key.encode("utf-8")
    # Compare against every key so timing does not reveal which key matched.
    matched = False
    for key in settings.TRUSTID_API_KEYS:
        if hmac.compare_digest(presented, key.encode("utf-8")):
            matched = True
    if not matched:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key.",
        )
    return api_key_fingerprint(x_api_key)


def generate_hmac_signature(payload: str, secret: str) -> str:
    """HMAC-SHA256 hex digest of `payload`."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str, secret: str) -> bool:
    """Constant-time HMAC verification."""
    expected = generate_hmac_signature(payload, secret)
    return hmac.compare_digest(expected, signature)


def sign_audit_hash(audit_hash: str) -> str:
    return generate_hmac_signature(audit_hash, settings.AUDIT_SIGNING_SECRET)


def verify_audit_signature(audit_hash: str, signature: str) -> bool:
    return verify_hmac_signature(audit_hash, signature, settings.AUDIT_SIGNING_SECRET)


def sign_webhook(timestamp: str, body: str) -> str:
    """Webhook signature covers the timestamp too, so captured callbacks cannot be replayed later."""
    return generate_hmac_signature(f"{timestamp}.{body}", settings.WEBHOOK_SIGNING_SECRET)
