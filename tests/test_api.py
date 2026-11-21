import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.middleware import BodySizeLimitMiddleware
from app.services import liveness_service as liveness_module
from app.services.audit_service import audit_service
from app.services.session_store import session_store
from tests.conftest import HEADERS, OTHER_HEADERS, jpeg_bytes, load_face, requires_models

SAMPLE_CARD_TEXT = (
    "SPECIMEN - GOVERNMENT OF INDIA - AADHAAR\nName: Demo Applicant\n"
    "DOB: 15/08/1994\nGender: FEMALE\n9841 2341 5679"
)


def create_session(client, headers=HEADERS, **extra):
    res = client.post("/api/v1/sessions/", json={"user_id": "usr_test", "document_type": "AADHAAR", **extra}, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()


def sample_card_bytes():
    with open("sample_assets/sample_id_card.jpg", "rb") as f:
        return f.read()


def full_verify(client, session_id, frames, doc=None, headers=HEADERS, **data):
    files = [("id_document", ("card.jpg", doc or sample_card_bytes(), "image/jpeg"))]
    files += [("live_frames", (f"f{i}.jpg", jpeg_bytes(fr), "image/jpeg")) for i, fr in enumerate(frames)]
    return client.post(
        "/api/v1/verify/full",
        data={"session_id": session_id, "extracted_ocr_text": SAMPLE_CARD_TEXT, **data},
        files=files,
        headers=headers,
    )


# ------------------------------------------------------------------ auth
def test_missing_api_key_is_401(client):
    assert client.post("/api/v1/sessions/", json={"user_id": "u"}).status_code == 401


def test_invalid_api_key_is_403(client):
    res = client.post("/api/v1/sessions/", json={"user_id": "u"}, headers={"X-API-KEY": "trustid_live_demo_key_2026"})
    assert res.status_code == 403


# -------------------------------------------------------------- sessions
def test_create_and_get_session(client):
    s = create_session(client)
    assert s["session_id"].startswith("vkyc_")
    assert len(s["challenges"]) == settings.REQUIRED_ACTIVE_STEPS
    assert "owner_key_id" not in s
    got = client.get(f"/api/v1/sessions/{s['session_id']}", headers=HEADERS)
    assert got.status_code == 200 and got.json()["status"] == "INITIALIZED"


def test_sessions_are_isolated_between_api_keys(client):
    s = create_session(client)
    assert client.get(f"/api/v1/sessions/{s['session_id']}", headers=OTHER_HEADERS).status_code == 404


def test_session_with_internal_callback_url_is_rejected(client):
    res = client.post(
        "/api/v1/sessions/",
        json={"user_id": "u1", "callback_url": "https://169.254.169.254/latest/meta-data"},
        headers=HEADERS,
    )
    assert res.status_code == 422


def test_invalid_user_id_is_rejected(client):
    assert client.post("/api/v1/sessions/", json={"user_id": "../../x y"}, headers=HEADERS).status_code == 422


def test_expired_session_is_gone(client):
    s = create_session(client)
    record = session_store.get(s["session_id"])
    record.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    session_store.save(record)
    res = full_verify(client, s["session_id"], [np.zeros((10, 10, 3), np.uint8)])
    assert res.status_code == 410
    assert client.get(f"/api/v1/sessions/{s['session_id']}", headers=HEADERS).json()["status"] == "EXPIRED"


# ---------------------------------------------------------------- uploads
def test_undecodable_image_is_400(client):
    s = create_session(client)
    res = client.post(
        "/api/v1/verify/full",
        data={"session_id": s["session_id"]},
        files=[("id_document", ("x.jpg", b"not an image", "image/jpeg")),
               ("live_frames", ("f.jpg", b"also not", "image/jpeg"))],
        headers=HEADERS,
    )
    assert res.status_code == 400


def test_too_many_frames_is_413(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_LIVE_FRAMES", 2)
    s = create_session(client)
    tiny = np.full((32, 32, 3), 128, np.uint8)
    assert full_verify(client, s["session_id"], [tiny] * 3).status_code == 413


def test_oversized_image_is_413(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_IMAGE_BYTES", 1000)
    s = create_session(client)
    big = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert full_verify(client, s["session_id"], [big]).status_code == 413


def test_body_size_middleware_rejects_large_bodies():
    mini = FastAPI()

    @mini.post("/echo")
    async def echo(payload: dict):
        return payload

    mini.add_middleware(BodySizeLimitMiddleware, max_bytes=100)
    with TestClient(mini) as c:
        assert c.post("/echo", json={"a": 1}).status_code == 200
        assert c.post("/echo", content=b"{" + b" " * 500 + b"}", headers={"content-type": "application/json"}).status_code == 413

        def chunks():  # no Content-Length header
            for _ in range(10):
                yield b" " * 50

        assert c.post("/echo", content=chunks(), headers={"content-type": "application/json"}).status_code == 413


# ------------------------------------------------------- full pipeline
@requires_models
def test_health_reports_ready(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["components"]["face_models"] == "ready"


@requires_models
def test_static_photo_frames_are_rejected_for_failed_liveness(client):
    # Regression: the same scenario used to be APPROVED-eligible because active
    # liveness failures never produced a rejection reason.
    s = create_session(client)
    frame = load_face("ride_1984.jpg")
    res = full_verify(client, s["session_id"], [frame] * 5)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["verdict"] == "REJECTED"
    reasons = body["dossier"]["rejection_reasons"]
    assert any(r.startswith("ACTIVE_LIVENESS_FAILED") for r in reasons)
    assert not any(r.startswith("BIOMETRIC_FACE_MISMATCH") for r in reasons)
    assert body["dossier"]["face_match"]["is_match"] is True


@requires_models
def test_frames_without_faces_are_rejected(client):
    s = create_session(client)
    blank = np.full((480, 640, 3), 190, np.uint8)
    body = full_verify(client, s["session_id"], [blank] * 3).json()
    assert body["verdict"] == "REJECTED"
    assert "NO_FACE_IN_LIVE_FRAMES" in body["dossier"]["rejection_reasons"]


@requires_models
def test_different_person_is_rejected_even_if_liveness_passes(client, monkeypatch):
    _force_challenges_pass(monkeypatch)
    s = create_session(client)
    body = full_verify(client, s["session_id"], [load_face("armstrong_1964.jpg")] * 3).json()
    assert body["verdict"] == "REJECTED"
    assert any(r.startswith("BIOMETRIC_FACE_MISMATCH") for r in body["dossier"]["rejection_reasons"])


@requires_models
def test_person_swap_between_frames_is_rejected(client, monkeypatch):
    _force_challenges_pass(monkeypatch)
    s = create_session(client)
    frames = [load_face("ride_1984.jpg")] * 3 + [load_face("collins.jpg")]
    body = full_verify(client, s["session_id"], frames).json()
    assert body["verdict"] == "REJECTED"
    assert "LIVE_FRAMES_IDENTITY_INCONSISTENT" in body["dossier"]["rejection_reasons"]


def _force_challenges_pass(monkeypatch):
    """
    Real head-turn / blink / smile sequences cannot be synthesised from still
    photos, so these tests stub the challenge evaluation (covered separately in
    test_liveness.py) to exercise the rest of the pipeline.
    """
    original = liveness_module.liveness_service.evaluate_challenges

    def passing(observations, challenges):
        _, steps = original(observations, challenges)
        return True, [s.model_copy(update={"is_completed": True, "failure_reason": None}) for s in steps]

    monkeypatch.setattr(liveness_module.liveness_service, "evaluate_challenges", passing)


@requires_models
def test_happy_path_approved_audited_and_single_use(client, monkeypatch):
    _force_challenges_pass(monkeypatch)
    s = create_session(client)
    res = full_verify(client, s["session_id"], [load_face("ride_1984.jpg")] * 3)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["verdict"] == "APPROVED", body["dossier"]["rejection_reasons"] + body["dossier"]["review_reasons"]
    assert body["dossier"]["document"]["text_source"] == "CLIENT_PROVIDED"
    assert body["dossier"]["document"]["extracted_data"]["document_number_checksum_valid"] is True
    vid = body["verification_id"]

    # Session is now terminal and cannot be reused.
    session = client.get(f"/api/v1/sessions/{s['session_id']}", headers=HEADERS).json()
    assert session["status"] == "VERIFIED" and session["verification_id"] == vid
    assert full_verify(client, s["session_id"], [load_face("ride_1984.jpg")] * 3).status_code == 409

    # Audit record: readable by owner, hidden from others, tamper-evident.
    audit = client.get(f"/api/v1/audit/{vid}", headers=HEADERS)
    assert audit.status_code == 200 and audit.json() == body["dossier"]
    assert client.get(f"/api/v1/audit/{vid}", headers=OTHER_HEADERS).status_code == 404

    path = audit_service.storage_dir / f"{vid}.json"
    record = json.loads(path.read_text())
    record["dossier"]["face_match"]["match_score"] = 0.99
    path.write_text(json.dumps(record))
    assert client.get(f"/api/v1/audit/{vid}", headers=HEADERS).status_code == 409


def test_audit_path_traversal_is_404(client):
    assert client.get("/api/v1/audit/..%2F..%2Fetc%2Fpasswd", headers=HEADERS).status_code == 404
    assert client.get("/api/v1/audit/verif_nothere", headers=HEADERS).status_code == 404


@requires_models
def test_face_match_endpoint(client):
    files = {
        "id_card_image": ("id.jpg", sample_card_bytes(), "image/jpeg"),
        "live_selfie_image": ("live.jpg", jpeg_bytes(load_face("ride_1984.jpg")), "image/jpeg"),
    }
    body = client.post("/api/v1/face/match", files=files, headers=HEADERS).json()
    assert body["is_match"] is True


@requires_models
def test_document_endpoint_without_text_does_not_invent_data(client):
    files = {"document_image": ("id.jpg", sample_card_bytes(), "image/jpeg")}
    body = client.post("/api/v1/document/verify", files=files, headers=HEADERS).json()
    assert body["text_source"] == "NONE"
    assert body["is_valid"] is False
    assert body["extracted_data"]["full_name"] is None


def test_pipeline_crash_releases_session_for_retry(client, monkeypatch):
    from app.api.v1.endpoints import verify as verify_module

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(verify_module, "run_pipeline", boom)
    s = create_session(client)
    tiny = np.full((32, 32, 3), 128, np.uint8)
    with pytest.raises(RuntimeError):
        full_verify(client, s["session_id"], [tiny])
    assert session_store.get(s["session_id"]).status.value == "INITIALIZED"
