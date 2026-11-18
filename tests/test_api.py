import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
API_KEY = "trustid_live_demo_key_2026"
HEADERS = {"X-API-KEY": API_KEY}


def create_mock_image_bytes(width=300, height=300, color=(100, 150, 200)):
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Add a pseudo face rectangle and circle
    cv2.circle(img, (width // 2, height // 2), 60, (220, 220, 220), -1)
    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models" in data


def test_create_session():
    payload = {
        "user_id": "applicant_test_101",
        "document_type": "AADHAAR",
    }
    response = client.post("/api/v1/sessions/", json=payload, headers=HEADERS)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert len(data["challenges"]) > 0
    assert "verification_code" in data


def test_unauthorized_request():
    response = client.post("/api/v1/sessions/", json={"user_id": "usr_fail"})
    assert response.status_code == 401


def test_full_vkyc_pipeline():
    # 1. Create Session
    session_res = client.post(
        "/api/v1/sessions/",
        json={"user_id": "usr_e2e_test"},
        headers=HEADERS,
    )
    assert session_res.status_code == 201
    session_id = session_res.json()["session_id"]

    # 2. Prepare mock document & live frames
    doc_bytes = create_mock_image_bytes(500, 350, (240, 240, 240))
    live_frame1 = create_mock_image_bytes(400, 400, (120, 140, 180))
    live_frame2 = create_mock_image_bytes(400, 400, (120, 140, 180))

    files = [
        ("id_document", ("aadhaar.jpg", doc_bytes, "image/jpeg")),
        ("live_frames", ("frame1.jpg", live_frame1, "image/jpeg")),
        ("live_frames", ("frame2.jpg", live_frame2, "image/jpeg")),
    ]
    data = {
        "session_id": session_id,
        "extracted_ocr_text": "GOVERNMENT OF INDIA\nAADHAAR\nRahul Sharma\nDOB: 15/08/1994\nMALE\n9841 2341 5678",
    }

    # 3. Call full verify
    response = client.post(
        "/api/v1/verify/full",
        data=data,
        files=files,
        headers=HEADERS,
    )
    assert response.status_code == 200
    res_json = response.json()
    assert "verification_id" in res_json
    assert "verdict" in res_json
    assert "dossier" in res_json

    verif_id = res_json["verification_id"]

    # 4. Fetch Audit Dossier
    audit_res = client.get(f"/api/v1/audit/{verif_id}", headers=HEADERS)
    assert audit_res.status_code == 200
    audit_json = audit_res.json()
    assert audit_json["verification_id"] == verif_id
    assert "tamper_proof_audit_hash" in audit_json
    assert "signature" in audit_json
