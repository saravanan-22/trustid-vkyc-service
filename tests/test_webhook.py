import asyncio
import json
import socket
import time

import httpx

from app.core import url_safety
from app.core.config import settings
from app.services.webhook_service import WebhookService
from client_sdk.trustid import verify_webhook_signature


def _public_dns(monkeypatch):
    monkeypatch.setattr(
        url_safety.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", port))],
    )


def test_webhook_is_signed_and_verifiable(monkeypatch):
    _public_dns(monkeypatch)
    captured = {}

    def handler(request: httpx.Request):
        captured["body"] = request.content.decode()
        captured["ts"] = request.headers["X-TrustID-Timestamp"]
        captured["sig"] = request.headers["X-TrustID-Signature"]
        return httpx.Response(200)

    svc = WebhookService(transport=httpx.MockTransport(handler))
    ok = asyncio.run(svc.send_verification_callback("https://hooks.example.com/kyc", {"verdict": "REJECTED"}))
    assert ok
    assert json.loads(captured["body"]) == {"verdict": "REJECTED"}
    secret = settings.WEBHOOK_SIGNING_SECRET
    assert verify_webhook_signature(captured["body"], captured["ts"], captured["sig"], secret)
    # Tampered body, wrong secret, and replay after the tolerance window all fail.
    assert not verify_webhook_signature(captured["body"].replace("REJECTED", "APPROVED"), captured["ts"], captured["sig"], secret)
    assert not verify_webhook_signature(captured["body"], captured["ts"], captured["sig"], "wrong-secret")
    assert not verify_webhook_signature(captured["body"], captured["ts"], captured["sig"], secret, now=time.time() + 3600)


def test_webhook_signing_key_is_not_the_audit_key():
    assert settings.WEBHOOK_SIGNING_SECRET != settings.AUDIT_SIGNING_SECRET


def test_webhook_to_private_address_is_blocked_at_send_time(monkeypatch):
    monkeypatch.setattr(
        url_safety.socket, "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port))],
    )
    calls = []
    svc = WebhookService(transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200)))
    ok = asyncio.run(svc.send_verification_callback("https://rebinding.example.com/", {"a": 1}))
    assert ok is False
    assert calls == []


def test_webhook_does_not_retry_client_errors(monkeypatch):
    _public_dns(monkeypatch)
    calls = []
    svc = WebhookService(transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(404)))
    assert asyncio.run(svc.send_verification_callback("https://hooks.example.com/", {"a": 1})) is False
    assert len(calls) == 1
