import os
from typing import Any, Dict, List, Optional
import httpx


class TrustIDClient:
    """
    Official Python Client SDK for TrustID Video KYC Service.
    Enables one-line integration for FinTech and Banking backends.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: str = "trustid_live_demo_key_2026",
        timeout: float = 30.0,
    ):
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
    ) -> Dict[str, Any]:
        """Runs full end-to-end Video KYC pipeline."""
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
