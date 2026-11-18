import numpy as np
import pytest
from app.schemas.session import ChallengeStep, ChallengeType
from app.services.liveness_service import liveness_service


def test_passive_texture_blur():
    # Blurry image
    blurry = np.full((120, 120, 3), 100, dtype=np.uint8)
    score, spoof_type = liveness_service.evaluate_passive_texture(blurry)
    assert score <= 0.35
    assert spoof_type == "BLURRY_OR_DEFOCUSED"


def test_passive_texture_skin():
    # Realistic skin chroma block in BGR
    # YCrCb of typical skin: Cr ~ 150, Cb ~ 100
    skin_bgr = np.zeros((150, 150, 3), dtype=np.uint8)
    skin_bgr[:, :, 0] = 110  # Blue
    skin_bgr[:, :, 1] = 130  # Green
    skin_bgr[:, :, 2] = 200  # Red
    # Add high-frequency noise
    noise = np.random.normal(0, 15, (150, 150, 3)).astype(np.int16)
    textured_skin = np.clip(skin_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    score, spoof_type = liveness_service.evaluate_passive_texture(textured_skin)
    assert score >= 0.50


def test_active_challenge_sequence_empty():
    passed, steps = liveness_service.verify_active_challenge_sequence([], [])
    assert passed is False
