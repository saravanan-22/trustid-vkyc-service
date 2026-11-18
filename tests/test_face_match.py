import numpy as np
import pytest
from app.services.face_matcher import face_matcher


def test_face_crop_fallback():
    # Synthetic canvas
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    crop = face_matcher.extract_face_crop(canvas)
    assert crop is not None
    assert crop.shape == (160, 160, 3)


def test_embedding_normalization():
    face_160 = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
    emb = face_matcher.compute_embedding(face_160)
    assert len(emb) == 512
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, 0.01) == 1.0


def test_face_comparison_identical():
    face = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    metrics = face_matcher.compare_faces(face, face)
    assert metrics.match_score >= 0.95
    assert metrics.is_match is True
