import hmac
import hashlib
from typing import Optional
from fastapi import Header, HTTPException, status
from app.core.config import settings


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Validates API key from request headers.
    Allows demo key or any configured production key.
    """
    if not x_api_key:
        # Allow requests with demo default key or raise unauthorized
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Provide 'X-API-KEY' header.",
        )
    
    if x_api_key != settings.DEFAULT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key.",
        )
    return x_api_key


def generate_hmac_signature(payload: str, secret: Optional[str] = None) -> str:
    """
    Generates a cryptographically secure HMAC-SHA256 signature for payloads.
    """
    key = (secret or settings.SECRET_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str, secret: Optional[str] = None) -> bool:
    """
    Constant-time comparison for HMAC verification to prevent timing attacks.
    """
    expected = generate_hmac_signature(payload, secret)
    return hmac.compare_digest(expected, signature)
