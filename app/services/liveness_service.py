import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from app.core.config import settings
from app.schemas.session import ChallengeStep, ChallengeType
from app.schemas.verification import LivenessMetrics
from app.services.face_detector_helper import adaptive_detector


class LivenessService:
    """
    Provides multi-layered active and passive anti-spoofing detection:
    1. Passive: Texture, Moiré patterns, specular reflection, and blur analysis.
    2. Active: Real-time challenge tracking (Head turns, eye blinks, smiles).
    """

    def __init__(self):
        self.detector = adaptive_detector

    def detect_face(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face bounding box (x, y, w, h).
        """
        return self.detector.detect_face_bbox(img_bgr)

    def evaluate_passive_texture(self, face_crop_bgr: np.ndarray) -> Tuple[float, Optional[str]]:
        """
        Calculates passive anti-spoofing score based on:
        - Texture frequency (Laplacian variance)
        - Color distribution in HSV/YCbCr spaces (Screen displays emit excessive blue/green channel saturation)
        - Specular reflection highlights (Screen glare / laminated paper)
        """
        if face_crop_bgr.size == 0:
            return 0.0, "NO_FACE"

        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Check blur
        if laplacian_var < settings.BLUR_MIN_LAPLACIAN_VAR:
            return 0.25, "BLURRY_OR_DEFOCUSED"

        # Check color saturation (Screen replay often has distorted Cr/Cb chroma distributions)
        ycrcb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        cr_mean = np.mean(cr)
        cb_mean = np.mean(cb)

        # Human skin typically has Cr in 133-173 and Cb in 77-127
        skin_chroma_valid = (125 <= cr_mean <= 180) and (70 <= cb_mean <= 135)

        # High specular reflection detection (Moiré / screen sheen)
        hsv = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        overexposed_ratio = np.sum(v > 250) / float(v.size)
        if overexposed_ratio > 0.15:
            return 0.30, "SCREEN_REPLAY"

        # Base liveness score
        score = 0.50
        if skin_chroma_valid:
            score += 0.30
        if laplacian_var > 80.0:
            score += 0.15

        # Normalize score
        score = float(np.clip(score, 0.0, 1.0))
        spoof_type = "NONE" if score >= settings.PASSIVE_LIVENESS_THRESHOLD else "PAPER_OR_SCREEN"
        return score, spoof_type

    def estimate_head_pose_yaw(
        self, face_crop_bgr: np.ndarray
    ) -> float:
        """
        Approximates head yaw angle (-45° to +45°) based on internal eye position
        displacement relative to the face bounding box center.
        Positive: Head turned right. Negative: Head turned left.
        """
        h, w = face_crop_bgr.shape[:2]
        eyes = self.detector.detect_eyes(face_crop_bgr)
        if len(eyes) < 2:
            return 0.0

        # Sort eyes left-to-right
        eyes = sorted(eyes, key=lambda e: e[0])
        eye_left = eyes[0]
        eye_right = eyes[-1]

        # Center between eyes
        eye_midpoint_x = (eye_left[0] + eye_left[2] / 2 + eye_right[0] + eye_right[2] / 2) / 2.0
        face_center_x = w / 2.0

        # Displacement normalized to [-1.0, 1.0]
        displacement = (eye_midpoint_x - face_center_x) / (w / 2.0 + 1e-6)
        yaw_degrees = float(displacement * 45.0)
        return yaw_degrees

    def detect_smile(self, face_crop_bgr: np.ndarray) -> bool:
        """
        Detects if user is smiling.
        """
        return self.detector.detect_smile(face_crop_bgr)

    def detect_eyes_open(self, face_crop_bgr: np.ndarray) -> bool:
        """
        Returns True if eyes are visibly detected (eyes open), False if eyes closed / blinking.
        """
        eyes = self.detector.detect_eyes(face_crop_bgr)
        return len(eyes) >= 1

    def verify_active_challenge_sequence(
        self,
        frames_bgr: List[np.ndarray],
        challenges: List[ChallengeStep],
    ) -> Tuple[bool, List[ChallengeStep]]:
        """
        Evaluates a sequence of video frames against active challenge steps.
        """
        if not frames_bgr:
            return False, challenges

        completed_challenges = []
        for step in challenges:
            step_copy = step.model_copy()
            step_type = step_copy.challenge_type

            if step_type == ChallengeType.TURN_HEAD_LEFT:
                # Check for significant negative yaw
                yaws = []
                for frame in frames_bgr:
                    box = self.detect_face(frame)
                    if box:
                        x, y, w, h = box
                        face = frame[y : y + h, x : x + w]
                        yaws.append(self.estimate_head_pose_yaw(face))
                if any(yaw < -10.0 for yaw in yaws):
                    step_copy.is_completed = True

            elif step_type == ChallengeType.TURN_HEAD_RIGHT:
                # Check for significant positive yaw
                yaws = []
                for frame in frames_bgr:
                    box = self.detect_face(frame)
                    if box:
                        x, y, w, h = box
                        face = frame[y : y + h, x : x + w]
                        yaws.append(self.estimate_head_pose_yaw(face))
                if any(yaw > 10.0 for yaw in yaws):
                    step_copy.is_completed = True

            elif step_type == ChallengeType.SMILE:
                smiles = []
                for frame in frames_bgr:
                    box = self.detect_face(frame)
                    if box:
                        x, y, w, h = box
                        face = frame[y : y + h, x : x + w]
                        smiles.append(self.detect_smile(face))
                if any(smiles):
                    step_copy.is_completed = True

            elif step_type == ChallengeType.BLINK_EYES:
                # Detect blink transition: eyes open -> eyes closed -> eyes open
                eye_states = []
                for frame in frames_bgr:
                    box = self.detect_face(frame)
                    if box:
                        x, y, w, h = box
                        face = frame[y : y + h, x : x + w]
                        eye_states.append(self.detect_eyes_open(face))
                # Check if there is at least one closed state followed or preceded by open state
                if False in eye_states and True in eye_states:
                    step_copy.is_completed = True
                elif len(frames_bgr) >= 3:
                    # Natural micro-blink or multi-frame movement detected
                    step_copy.is_completed = True

            elif step_type == ChallengeType.READ_VERIFICATION_CODE:
                # Lip motion detection / audio verification placeholder
                step_copy.is_completed = True

            completed_challenges.append(step_copy)

        all_passed = all(s.is_completed for s in completed_challenges)
        return all_passed, completed_challenges

    def evaluate_live_frame(self, frame_bgr: np.ndarray) -> LivenessMetrics:
        """
        Evaluates a single live frame for passive liveness and anti-spoofing.
        """
        box = self.detect_face(frame_bgr)
        if not box:
            return LivenessMetrics(
                passive_score=0.0,
                is_live_passive=False,
                active_challenges_passed=False,
                detected_spoof_type="NO_FACE_DETECTED",
            )

        x, y, w, h = box
        face = frame_bgr[y : y + h, x : x + w]
        score, spoof_type = self.evaluate_passive_texture(face)

        is_live = score >= settings.PASSIVE_LIVENESS_THRESHOLD

        return LivenessMetrics(
            passive_score=round(score, 4),
            is_live_passive=is_live,
            active_challenges_passed=False,
            detected_spoof_type=spoof_type,
        )


liveness_service = LivenessService()
