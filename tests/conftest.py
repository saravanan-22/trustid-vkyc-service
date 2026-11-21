"""
Test configuration. Environment variables are set *before* any `app` module is
imported, because settings are read at import time.
"""

import os
import tempfile
from pathlib import Path

TEST_API_KEY = "test-api-key-primary-000000000000"
OTHER_API_KEY = "test-api-key-secondary-11111111111"

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "TRUSTID_API_KEYS": f"{TEST_API_KEY},{OTHER_API_KEY}",
        "AUDIT_SIGNING_SECRET": "test-audit-signing-secret-" + "a" * 32,
        "WEBHOOK_SIGNING_SECRET": "test-webhook-signing-secret-" + "b" * 32,
        "DATA_DIR": tempfile.mkdtemp(prefix="trustid-test-"),
        "OCR_ENGINE": "none",
    }
)
os.environ.pop("REDIS_URL", None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402

FACES_DIR = Path(__file__).parent / "fixtures" / "faces"
HEADERS = {"X-API-KEY": TEST_API_KEY}
OTHER_HEADERS = {"X-API-KEY": OTHER_API_KEY}

MODELS_PRESENT = settings.face_detector_path.is_file() and settings.face_recognizer_path.is_file()
requires_models = pytest.mark.skipif(
    not MODELS_PRESENT, reason="face models missing: run `python scripts/download_models.py`"
)


def load_face(name: str) -> np.ndarray:
    img = cv2.imread(str(FACES_DIR / name))
    assert img is not None, f"missing fixture {name}"
    return img


def jpeg_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
