import asyncio
import json
import logging
import time
from typing import Any, Dict

import httpx

from app.core.config import settings
from app.core.security import sign_webhook
from app.core.url_safety import UnsafeURLError, validate_callback_url

logger = logging.getLogger(__name__)


class WebhookService:
    """
    Delivers signed verification results to client callback URLs.

    Headers sent:
      X-TrustID-Timestamp: unix seconds
      X-TrustID-Signature: hex HMAC-SHA256 of "<timestamp>.<raw body>" with
                           WEBHOOK_SIGNING_SECRET
    Receivers should verify the signature and reject old timestamps
    (see `client_sdk.trustid.verify_webhook_signature`).
    """

    def __init__(self, transport: httpx.AsyncBaseTransport = None):
        self._transport = transport

    async def send_verification_callback(
        self,
        callback_url: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> bool:
        body = json.dumps(payload, default=str, separators=(",", ":"), sort_keys=True)

        async with httpx.AsyncClient(
            timeout=settings.WEBHOOK_TIMEOUT_SECONDS,
            follow_redirects=False,
            transport=self._transport,
        ) as client:
            for attempt in range(1, max_retries + 1):
                # Re-check right before sending: DNS may have changed since the request.
                try:
                    await asyncio.to_thread(validate_callback_url, callback_url)
                except UnsafeURLError as exc:
                    logger.warning("Webhook blocked: %s", exc)
                    return False

                timestamp = str(int(time.time()))
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "TrustID-VKYC-Webhook/1.1",
                    "X-TrustID-Timestamp": timestamp,
                    "X-TrustID-Signature": sign_webhook(timestamp, body),
                }
                try:
                    response = await client.post(callback_url, content=body, headers=headers)
                    if 200 <= response.status_code < 300:
                        logger.info("Webhook delivered (attempt %d)", attempt)
                        return True
                    logger.warning("Webhook returned HTTP %d (attempt %d)", response.status_code, attempt)
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        return False  # client error: retrying will not help
                except httpx.HTTPError as exc:
                    logger.warning("Webhook delivery failed (attempt %d): %s", attempt, exc)

                if attempt < max_retries:
                    await asyncio.sleep(1.5 * attempt)
        return False


webhook_service = WebhookService()
