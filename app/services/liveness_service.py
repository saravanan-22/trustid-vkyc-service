"""
Liveness checks.

Passive: single-frame texture/colour heuristics (blur, over-exposure, skin
chroma). These catch only crude spoofs; they are NOT a trained presentation
attack detection (PAD) model.

Active: randomized challenges checked from the five YuNet landmarks across the
submitted frame sequence. Every challenge requires a measurable *change*
between frames, so a static photo or a single repeated frame cannot pass.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.session import ChallengeStep, ChallengeType
from app.schemas.verification import LivenessMetrics
from app.services.face_engine import Face, face_engine

MIN_FRAMES_WITH_FACE = 3

# Head turn: horizontal nose offset from the eye midpoint, in inter-ocular distances.
FRONTAL_MAX_ABS_YAW = 0.15
TURNED_MIN_ABS_YAW = 0.30
# Smile: mouth width / inter-ocular distance must grow by this factor vs. neutral.
SMILE_MIN_WIDTH_GAIN = 1.10
SMILE_MAX_ABS_YAW = 0.20
# Blink: eye "openness" must drop below this fraction of the open level, then recover.
BLINK_CLOSED_FRACTION = 0.65
BLINK_REOPEN_FRACTION = 0.85


@dataclass(frozen=True)
class FrameObservation:
    face_count: int
    yaw: Optional[float] = None  # >0: subject turned to their LEFT (unmirrored image)
    mouth_ratio: Optional[float] = None
    eye_openness: Optional[float] = None

    @property
    def has_face(self) -> bool:
        return self.face_count > 0 and self.yaw is not None


# ---------------------------------------------------------------- measurements
def yaw_proxy(face: Face, mirrored: bool = False) -> float:
    """
    Signed horizontal offset of the nose tip from the eye midpoint, normalised by
    inter-ocular distance. ~0 when facing the camera. In an unmirrored camera
    image, a subject turning to *their* left moves the nose towards image-right,
    giving a positive value.
    """
    iod = face.inter_ocular_distance
    if iod < 1e-6:
        return 0.0
    eye_mid_x = (face.right_eye[0] + face.left_eye[0]) / 2.0
    value = (face.nose[0] - eye_mid_x) / iod
    return -value if mirrored else value


def mouth_width_ratio(face: Face) -> float:
    iod = face.inter_ocular_distance
    if iod < 1e-6:
        return 0.0
    width = float(np.hypot(face.mouth_left[0] - face.mouth_right[0], face.mouth_left[1] - face.mouth_right[1]))
    return width / iod


def eye_openness(img_bgr: np.ndarray, face: Face) -> float:
    """
    Intensity spread inside small patches centred on each eye. An open eye shows
    dark iris against bright sclera (high spread); a closed lid is mostly skin.
    Only meaningful *relative to other frames of the same sequence*.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    half = max(2, int(face.inter_ocular_distance * 0.22))
    values = []
    for cx, cy in (face.right_eye, face.left_eye):
        x1, y1 = max(0, int(cx) - half), max(0, int(cy) - half // 2)
        x2, y2 = min(gray.shape[1], int(cx) + half), min(gray.shape[0], int(cy) + half // 2)
        patch = gray[y1:y2, x1:x2]
        if patch.size:
            values.append(float(np.std(patch)))
    return float(np.mean(values)) if values else 0.0


# ------------------------------------------------------------ challenge rules
def evaluate_head_turn(observations: Sequence[FrameObservation], direction: str) -> Tuple[bool, Optional[str]]:
    """Needs at least one frontal frame AND one frame turned the requested way."""
    yaws = [o.yaw for o in observations if o.has_face]
    if len(yaws) < 2:
        return False, "INSUFFICIENT_FRAMES_WITH_FACE"
    sign = 1.0 if direction == "LEFT" else -1.0
    has_frontal = any(abs(y) <= FRONTAL_MAX_ABS_YAW for y in yaws)
    has_turn = any(sign * y >= TURNED_MIN_ABS_YAW for y in yaws)
    if not has_frontal:
        return False, "NO_FRONTAL_FRAME"
    if not has_turn:
        return False, f"NO_HEAD_TURN_{direction}_DETECTED"
    return True, None


def evaluate_smile(observations: Sequence[FrameObservation]) -> Tuple[bool, Optional[str]]:
    """Mouth must widen noticeably relative to the most neutral frontal frame."""
    ratios = [
        o.mouth_ratio
        for o in observations
        if o.has_face and o.mouth_ratio and abs(o.yaw) <= SMILE_MAX_ABS_YAW
    ]
    if len(ratios) < 2:
        return False, "INSUFFICIENT_FRONTAL_FRAMES"
    if max(ratios) < min(ratios) * SMILE_MIN_WIDTH_GAIN:
        return False, "NO_SMILE_DETECTED"
    return True, None


def evaluate_blink(observations: Sequence[FrameObservation]) -> Tuple[bool, Optional[str]]:
    """Requires an open -> closed -> open pattern in eye openness."""
    values = [o.eye_openness for o in observations if o.has_face and o.eye_openness is not None]
    if len(values) < MIN_FRAMES_WITH_FACE:
        return False, "INSUFFICIENT_FRAMES_WITH_FACE"
    open_level = float(np.percentile(values, 90))
    if open_level <= 1e-6:
        return False, "NO_BLINK_DETECTED"
    closed_at = [i for i, v in enumerate(values) if v < open_level * BLINK_CLOSED_FRACTION]
    is_open = [v >= open_level * BLINK_REOPEN_FRACTION for v in values]
    for i in closed_at:
        if any(is_open[:i]) and any(is_open[i + 1:]):
            return True, None
    return False, "NO_BLINK_DETECTED"


class LivenessService:
    def __init__(self, engine=face_engine):
        self.engine = engine

    # ----------------------------------------------------------------- passive
    def evaluate_passive_texture(self, face_crop_bgr: np.ndarray) -> Tuple[float, Optional[str]]:
        """
        Heuristic passive score from blur, over-exposure and skin chroma.
        Returns (score, spoof_type) where spoof_type is "NONE" when it passes.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return 0.0, "NO_FACE_DETECTED"

        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < settings.BLUR_MIN_LAPLACIAN_VAR:
            return 0.25, "BLURRY_OR_DEFOCUSED"

        hsv = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2HSV)
        overexposed_ratio = np.sum(hsv[:, :, 2] > 250) / float(gray.size)
        if overexposed_ratio > 0.15:
            return 0.30, "SCREEN_GLARE_SUSPECTED"

        ycrcb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2YCrCb)
        cr_mean = float(np.mean(ycrcb[:, :, 1]))
        cb_mean = float(np.mean(ycrcb[:, :, 2]))
        skin_chroma_valid = (125 <= cr_mean <= 180) and (70 <= cb_mean <= 135)

        score = 0.50
        if skin_chroma_valid:
            score += 0.30
        if laplacian_var > 80.0:
            score += 0.15
        score = float(np.clip(score, 0.0, 1.0))
        spoof_type = "NONE" if score >= settings.PASSIVE_LIVENESS_THRESHOLD else "ATYPICAL_SKIN_TEXTURE"
        return score, spoof_type

    def evaluate_live_frame(self, frame_bgr: np.ndarray, faces: Optional[List[Face]] = None) -> LivenessMetrics:
        if faces is None:
            faces = self.engine.detect_faces(frame_bgr)
        if not faces:
            return LivenessMetrics(
                passive_score=0.0,
                is_live_passive=False,
                active_challenges_passed=False,
                detected_spoof_type="NO_FACE_DETECTED",
                faces_detected=0,
            )
        score, spoof_type = self.evaluate_passive_texture(faces[0].crop(frame_bgr))
        return LivenessMetrics(
            passive_score=round(score, 4),
            is_live_passive=score >= settings.PASSIVE_LIVENESS_THRESHOLD,
            active_challenges_passed=False,
            detected_spoof_type=spoof_type,
            faces_detected=len(faces),
        )

    # ------------------------------------------------------------------ active
    def observe(self, frame_bgr: np.ndarray, faces: Optional[List[Face]] = None) -> FrameObservation:
        """Measures one frame. Pass `faces` if detection already ran on it."""
        if faces is None:
            faces = self.engine.detect_faces(frame_bgr)
        if not faces:
            return FrameObservation(face_count=0)
        face = faces[0]
        return FrameObservation(
            face_count=len(faces),
            yaw=yaw_proxy(face, mirrored=settings.CAMERA_FRAMES_MIRRORED),
            mouth_ratio=mouth_width_ratio(face),
            eye_openness=eye_openness(frame_bgr, face),
        )

    def evaluate_challenges(
        self, observations: Sequence[FrameObservation], challenges: Sequence[ChallengeStep]
    ) -> Tuple[bool, List[ChallengeStep]]:
        frames_with_face = sum(1 for o in observations if o.has_face)
        results: List[ChallengeStep] = []
        for step in challenges:
            completed, reason = False, None
            if frames_with_face < MIN_FRAMES_WITH_FACE:
                reason = "INSUFFICIENT_FRAMES_WITH_FACE"
            elif step.challenge_type == ChallengeType.TURN_HEAD_LEFT:
                completed, reason = evaluate_head_turn(observations, "LEFT")
            elif step.challenge_type == ChallengeType.TURN_HEAD_RIGHT:
                completed, reason = evaluate_head_turn(observations, "RIGHT")
            elif step.challenge_type == ChallengeType.SMILE:
                completed, reason = evaluate_smile(observations)
            elif step.challenge_type == ChallengeType.BLINK_EYES:
                completed, reason = evaluate_blink(observations)
            else:
                # e.g. READ_VERIFICATION_CODE needs audio/lip-sync analysis,
                # which this service does not implement. Never auto-pass it.
                reason = "CHALLENGE_TYPE_NOT_SUPPORTED"
            results.append(step.model_copy(update={"is_completed": completed, "failure_reason": reason}))

        all_passed = bool(results) and all(s.is_completed for s in results)
        return all_passed, results

    def verify_active_challenge_sequence(
        self, frames_bgr: Sequence[np.ndarray], challenges: Sequence[ChallengeStep]
    ) -> Tuple[bool, List[ChallengeStep]]:
        observations = [self.observe(f) for f in frames_bgr]
        return self.evaluate_challenges(observations, challenges)


liveness_service = LivenessService()
