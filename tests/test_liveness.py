import numpy as np
import pytest

from app.schemas.session import ChallengeStep, ChallengeType
from app.services.face_engine import Face
from app.services.liveness_service import (
    FrameObservation,
    evaluate_blink,
    evaluate_head_turn,
    evaluate_smile,
    liveness_service,
    yaw_proxy,
)
from tests.conftest import load_face, requires_models


def obs(yaw=0.0, mouth=0.85, eyes=40.0):
    return FrameObservation(face_count=1, yaw=yaw, mouth_ratio=mouth, eye_openness=eyes)


def step(ctype):
    return ChallengeStep(step_id=1, challenge_type=ctype, instruction="x")


# ------------------------------------------------------------- head turn
def test_head_turn_left_requires_frontal_then_turned_frame():
    ok, reason = evaluate_head_turn([obs(0.02), obs(0.15), obs(0.40)], "LEFT")
    assert ok and reason is None


def test_head_turn_wrong_direction_fails():
    ok, reason = evaluate_head_turn([obs(0.0), obs(-0.45)], "LEFT")
    assert not ok and reason == "NO_HEAD_TURN_LEFT_DETECTED"
    ok, _ = evaluate_head_turn([obs(0.0), obs(-0.45)], "RIGHT")
    assert ok


def test_static_turned_photo_fails_head_turn():
    # A photo of someone already looking sideways has no frontal frame.
    ok, reason = evaluate_head_turn([obs(0.5)] * 5, "LEFT")
    assert not ok and reason == "NO_FRONTAL_FRAME"


def test_static_frontal_frames_fail_head_turn():
    ok, reason = evaluate_head_turn([obs(0.01)] * 5, "RIGHT")
    assert not ok and reason == "NO_HEAD_TURN_RIGHT_DETECTED"


def _face(nose_x):
    return Face(box=(0, 0, 100, 100), right_eye=(30.0, 40.0), left_eye=(70.0, 40.0), nose=(nose_x, 60.0),
                mouth_right=(35.0, 80.0), mouth_left=(65.0, 80.0), score=0.9, raw=np.zeros(15, np.float32))


def test_yaw_proxy_sign_and_mirroring():
    assert yaw_proxy(_face(50.0)) == pytest.approx(0.0)
    assert yaw_proxy(_face(66.0)) == pytest.approx(0.4)
    assert yaw_proxy(_face(66.0), mirrored=True) == pytest.approx(-0.4)


# ------------------------------------------------------------------ smile
def test_smile_requires_mouth_widening():
    ok, _ = evaluate_smile([obs(mouth=0.85), obs(mouth=0.86), obs(mouth=1.00)])
    assert ok


def test_constant_mouth_fails_smile():
    ok, reason = evaluate_smile([obs(mouth=0.95)] * 5)
    assert not ok and reason == "NO_SMILE_DETECTED"


def test_smile_ignores_turned_frames():
    # The wide mouth only appears in a strongly turned frame, which is ignored.
    ok, reason = evaluate_smile([obs(mouth=0.85), obs(yaw=0.5, mouth=1.2)])
    assert not ok and reason == "INSUFFICIENT_FRONTAL_FRAMES"


# ------------------------------------------------------------------ blink
def test_blink_requires_open_closed_open():
    ok, _ = evaluate_blink([obs(eyes=40), obs(eyes=41), obs(eyes=18), obs(eyes=39)])
    assert ok


def test_static_eyes_fail_blink():
    ok, reason = evaluate_blink([obs(eyes=40)] * 6)
    assert not ok and reason == "NO_BLINK_DETECTED"


def test_eyes_closing_without_reopening_fails_blink():
    ok, _ = evaluate_blink([obs(eyes=40), obs(eyes=40), obs(eyes=15)])
    assert not ok


# ---------------------------------------------------- challenge evaluation
def test_too_few_frames_fails_every_challenge():
    passed, steps = liveness_service.evaluate_challenges([obs(), obs()], [step(ChallengeType.SMILE)])
    assert not passed
    assert steps[0].failure_reason == "INSUFFICIENT_FRAMES_WITH_FACE"


def test_unsupported_challenge_never_auto_passes():
    # Regression: READ_VERIFICATION_CODE used to be marked complete unconditionally.
    passed, steps = liveness_service.evaluate_challenges(
        [obs()] * 5, [step(ChallengeType.READ_VERIFICATION_CODE)]
    )
    assert not passed
    assert steps[0].failure_reason == "CHALLENGE_TYPE_NOT_SUPPORTED"


def test_no_challenges_does_not_pass():
    passed, _ = liveness_service.evaluate_challenges([obs()] * 5, [])
    assert passed is False


# -------------------------------------------------------------- passive
def test_passive_texture_blur():
    blurry = np.full((120, 120, 3), 100, dtype=np.uint8)
    score, spoof_type = liveness_service.evaluate_passive_texture(blurry)
    assert score <= 0.35
    assert spoof_type == "BLURRY_OR_DEFOCUSED"


def test_passive_texture_empty_crop():
    score, spoof_type = liveness_service.evaluate_passive_texture(np.zeros((0, 0, 3), np.uint8))
    assert score == 0.0 and spoof_type == "NO_FACE_DETECTED"


# ------------------------------------------------------- with real models
@requires_models
def test_repeated_static_photo_fails_all_challenges():
    frame = load_face("ride_1984.jpg")
    challenges = [step(t) for t in (ChallengeType.TURN_HEAD_LEFT, ChallengeType.BLINK_EYES, ChallengeType.SMILE)]
    passed, steps = liveness_service.verify_active_challenge_sequence([frame] * 6, challenges)
    assert not passed
    assert not any(s.is_completed for s in steps)


@requires_models
def test_frame_without_face_reports_no_face():
    metrics = liveness_service.evaluate_live_frame(np.full((480, 640, 3), 200, np.uint8))
    assert metrics.is_live_passive is False
    assert metrics.detected_spoof_type == "NO_FACE_DETECTED"
    assert metrics.faces_detected == 0


@requires_models
def test_real_face_passes_passive_heuristic():
    metrics = liveness_service.evaluate_live_frame(load_face("ride_1984.jpg"))
    assert metrics.faces_detected == 1
    assert metrics.is_live_passive
