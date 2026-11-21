import numpy as np
from app.services.deepfake_detector import deepfake_detector


def test_fft_frequency_analysis_real_simulation():
    # Natural frame with Gaussian sensor noise
    img = np.random.normal(128, 20, (256, 256, 3)).astype(np.uint8)
    risk, var = deepfake_detector.analyze_frequency_spectrum(img)
    assert 0.0 <= risk <= 1.0
    assert var >= 0.0


def test_virtual_camera_detector():
    # Unnaturally flat frame (e.g. frozen digital camera buffer)
    flat_img = np.full((256, 256, 3), 128, dtype=np.uint8)
    is_virtual = deepfake_detector.detect_virtual_camera(flat_img)
    assert is_virtual is True

    # Real textured image
    natural_img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    is_virtual_natural = deepfake_detector.detect_virtual_camera(natural_img)
    assert is_virtual_natural is False


def test_deepfake_eval_metrics():
    img = np.random.randint(40, 220, (300, 300, 3), dtype=np.uint8)
    metrics = deepfake_detector.evaluate_frame(img)
    assert hasattr(metrics, "synthetic_risk_score")
    assert hasattr(metrics, "is_deepfake_suspected")
    assert hasattr(metrics, "virtual_camera_detected")
