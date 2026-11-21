import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional, Union

import httpx

DEFAULT_WEBHOOK_TOLERANCE_SECONDS = 300


def verify_webhook_signature(
    body: Union[bytes, str],
    timestamp: str,
    signature: str,
    secret: str,
    tolerance_seconds: int = DEFAULT_WEBHOOK_TOLERANCE_SECONDS,
    now: Optional[float] = None,
) -> bool:
    """
    Verifies a TrustID webhook.

    Pass the *raw* request body and the `X-TrustID-Timestamp` and
    `X-TrustID-Signature` headers, plus your WEBHOOK_SIGNING_SECRET. Returns
    False for a bad signature or a timestamp outside the tolerance window
    (which blocks replay of captured webhooks).
    """
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else now
    if abs(current - ts) > tolerance_seconds:
        return False
    raw = body.decode("utf-8") if isinstance(body, bytes) else body
    expected = hmac.new(secret.encode("utf-8"), f"{ts}.{raw}".encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


class TrustIDClient:
    """
    Python client for the TrustID VKYC API.

    The API key is taken from `api_key` or the TRUSTID_API_KEY environment
    variable. Never embed it in browser or mobile code.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        api_key = api_key or os.getenv("TRUSTID_API_KEY")
        if not api_key:
            raise ValueError("TrustID API key required: pass api_key or set TRUSTID_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"X-API-KEY": self.api_key}
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Checks API health and model readiness."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()

    def create_session(
        self,
        user_id: str,
        document_type: str = "NATIONAL_ID",
        callback_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Initializes a new KYC session with dynamic anti-spoofing challenges."""
        payload = {
            "user_id": user_id,
            "document_type": document_type,
            "callback_url": callback_url,
            "metadata": metadata or {},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/v1/sessions/",
                json=payload,
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()

    def verify_document(
        self, document_image_path: str, extracted_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs quality and tamper checks on an ID card image."""
        with open(document_image_path, "rb") as f:
            files = {"document_image": (os.path.basename(document_image_path), f, "image/jpeg")}
            data = {"extracted_text": extracted_text} if extracted_text else {}
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/v1/document/verify",
                    files=files,
                    data=data,
                    headers=self.headers,
                )
                resp.raise_for_status()
                return resp.json()

    def match_faces(
        self, id_card_image_path: str, live_selfie_image_path: str
    ) -> Dict[str, Any]:
        """Performs 1:1 biometric facial comparison."""
        with open(id_card_image_path, "rb") as f_id, open(live_selfie_image_path, "rb") as f_live:
            files = {
                "id_card_image": (os.path.basename(id_card_image_path), f_id, "image/jpeg"),
                "live_selfie_image": (os.path.basename(live_selfie_image_path), f_live, "image/jpeg"),
            }
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/v1/face/match",
                    files=files,
                    headers=self.headers,
                )
                resp.raise_for_status()
                return resp.json()

    def verify_full(
        self,
        session_id: str,
        id_document_path: str,
        live_frame_paths: List[str],
        callback_url: Optional[str] = None,
        extracted_ocr_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full pipeline. `live_frame_paths` must be the ordered frames
        captured while the applicant performed the session's challenges.
        """
        files = []
        file_handles = []
        try:
            f_id = open(id_document_path, "rb")
            file_handles.append(f_id)
            files.append(("id_document", (os.path.basename(id_document_path), f_id, "image/jpeg")))

            for p in live_frame_paths:
                fh = open(p, "rb")
                file_handles.append(fh)
                files.append(("live_frames", (os.path.basename(p), fh, "image/jpeg")))

            data = {"session_id": session_id}
            if callback_url:
                data["callback_url"] = callback_url
            if extracted_ocr_text:
                data["extracted_ocr_text"] = extracted_ocr_text

            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/v1/verify/full",
                    data=data,
                    files=files,
                    headers=self.headers,
                )
                resp.raise_for_status()
                return resp.json()
        finally:
            for fh in file_handles:
                fh.close()

    def get_audit_dossier(self, verification_id: str) -> Dict[str, Any]:
        """Retrieves and verifies cryptographic audit record."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(
                f"{self.base_url}/api/v1/audit/{verification_id}",
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()
