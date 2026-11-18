import asyncio
import logging
from typing import Any, Dict, Optional
import httpx
from app.core.security import generate_hmac_signature

logger = logging.getLogger(__name__)


class WebhookService:
    """
    Dispatches asynchronous signed webhooks to client callback URLs.
    Includes HMAC-SHA256 signature header for webhook authenticity.
    """

    async def send_verification_callback(
        self,
        callback_url: str,
        payload: Dict[str, Any],
        max_retries: int = 3,
    ) -> bool:
        """
        Delivers verification result payload to client callback endpoint.
        """
        import json
        payload_str = json.dumps(payload, default=str)
        signature = generate_hmac_signature(payload_str)

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TrustID-VKYC-Webhook/1.0",
            "X-TrustID-Signature": signature,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    response = await client.post(callback_url, content=payload_str, headers=headers)
                    if response.status_code in [200, 201, 202, 204]:
                        logger.info(f"Webhook delivered to {callback_url} [Attempt {attempt}]")
                        return True
                    else:
                        logger.warning(
                            f"Webhook {callback_url} returned HTTP {response.status_code} [Attempt {attempt}]"
                        )
                except Exception as e:
                    logger.error(f"Webhook delivery failed [Attempt {attempt}]: {str(e)}")

                if attempt < max_retries:
                    await asyncio.sleep(1.5 * attempt)

        return False


webhook_service = WebhookService()
