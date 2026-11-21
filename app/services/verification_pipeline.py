"""
End-to-end VKYC pipeline and decision policy.

Policy (see `decide`):
* REJECTED if any hard check fails: no face, several faces, face mismatch,
  identity changes between frames, a failed active challenge, or a failed
  passive liveness check.
* REVIEW_REQUIRED if no hard check failed but a soft signal fired: document
  checks failed, or a synthetic-media / virtual-camera heuristic flagged.
* APPROVED only if every check passed.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.core.config import settings
from app.schemas.document import DocumentType, DocumentVerificationResult
from app.schemas.session import ChallengeStep
from app.schemas.verification import (
    DeepfakeMetrics,
    FaceMatchMetrics,
    LivenessMetrics,
    VerificationVerdict,
)
from app.services.deepfake_detector import deepfake_detector
from app.services.face_engine import Face, face_engine
from app.services.face_matcher import face_matcher
from app.services.liveness_service import liveness_service
from app.services.ocr_service import ocr_service

POLICY_VERSION = "2026.11-1"
IDENTITY_SAMPLE_FRAMES = 4


@dataclass
class PipelineResult:
    face_match: FaceMatchMetrics
    liveness: LivenessMetrics
    deepfake: DeepfakeMetrics
    document: DocumentVerificationResult
    identity_consistent: bool
    max_faces_in_frames: int
    frames_analyzed: int


def decide(
    face_match: FaceMatchMetrics,
    liveness: LivenessMetrics,
    deepfake: DeepfakeMetrics,
    document: DocumentVerificationResult,
    identity_consistent: bool = True,
    max_faces_in_frames: int = 1,
) -> Tuple[VerificationVerdict, List[str], List[str]]:
    rejection: List[str] = []
    review: List[str] = []

    if not face_match.face_detected_in_live:
        rejection.append("NO_FACE_IN_LIVE_FRAMES")
    if not face_match.face_detected_in_id:
        rejection.append("NO_FACE_ON_DOCUMENT")
    if max(max_faces_in_frames, face_match.faces_in_live) > 1:
        rejection.append("MULTIPLE_FACES_IN_LIVE_FRAMES")
    if face_match.face_detected_in_live and face_match.face_detected_in_id and not face_match.is_match:
        rejection.append(
            f"BIOMETRIC_FACE_MISMATCH: similarity {face_match.match_score:.3f} "
            f"below threshold {face_match.threshold:.3f}"
        )
    if not identity_consistent:
        rejection.append("LIVE_FRAMES_IDENTITY_INCONSISTENT")

    if not liveness.active_challenges_passed:
        failed = [
            f"{s.challenge_type.value}({s.failure_reason or 'NOT_COMPLETED'})"
            for s in liveness.completed_challenges
            if not s.is_completed
        ]
        rejection.append("ACTIVE_LIVENESS_FAILED: " + (", ".join(failed) or "NO_CHALLENGES"))
    if face_match.face_detected_in_live and not liveness.is_live_passive:
        rejection.append(f"PASSIVE_LIVENESS_FAILED: {liveness.detected_spoof_type}")

    if not document.is_valid:
        review.append("DOCUMENT_CHECKS_FAILED: " + ", ".join(document.tamper_flags))
    if deepfake.is_deepfake_suspected:
        review.append("SYNTHETIC_MEDIA_HEURISTIC_FLAGGED")
    if deepfake.virtual_camera_detected:
        review.append("VIRTUAL_CAMERA_HEURISTIC_FLAGGED")

    if rejection:
        verdict = VerificationVerdict.REJECTED
    elif review:
        verdict = VerificationVerdict.REVIEW_REQUIRED
    else:
        verdict = VerificationVerdict.APPROVED
    return verdict, rejection, review


def overall_confidence(result: PipelineResult) -> float:
    """Unweighted mean of the component scores. Informational only; `decide` sets the verdict."""
    scores = [
        result.face_match.match_score,
        result.liveness.passive_score,
        1.0 - result.deepfake.synthetic_risk_score,
        result.document.confidence,
    ]
    return round(float(np.mean(scores)), 4)


def _pick_primary_frame(frame_faces: Sequence[List[Face]]) -> int:
    """The frame with the most frontal single face is best for matching."""
    best_idx, best_yaw = None, None
    for idx, faces in enumerate(frame_faces):
        if len(faces) != 1:
            continue
        f = faces[0]
        iod = f.inter_ocular_distance
        if iod < 1e-6:
            continue
        yaw = abs(f.nose[0] - (f.left_eye[0] + f.right_eye[0]) / 2.0) / iod
        if best_yaw is None or yaw < best_yaw:
            best_idx, best_yaw = idx, yaw
    if best_idx is None:
        with_face = [i for i, faces in enumerate(frame_faces) if faces]
        best_idx = with_face[len(with_face) // 2] if with_face else len(frame_faces) // 2
    return best_idx


def _identity_consistent(
    frames: Sequence[np.ndarray], frame_faces: Sequence[List[Face]], primary_idx: int
) -> bool:
    """Every sampled frame must show the same person as the primary frame."""
    if not frame_faces[primary_idx]:
        return True  # nothing to compare; absence of a face is rejected elsewhere
    indices = [i for i, faces in enumerate(frame_faces) if faces and i != primary_idx]
    if len(indices) > IDENTITY_SAMPLE_FRAMES:
        step = len(indices) / IDENTITY_SAMPLE_FRAMES
        indices = [indices[int(k * step)] for k in range(IDENTITY_SAMPLE_FRAMES)]
    reference = face_engine.embedding(frames[primary_idx], frame_faces[primary_idx][0])
    for i in indices:
        emb = face_engine.embedding(frames[i], frame_faces[i][0])
        if face_engine.cosine_similarity(reference, emb) < settings.FACE_MATCH_THRESHOLD:
            return False
    return True


def run_pipeline(
    document_bgr: np.ndarray,
    live_frames: Sequence[np.ndarray],
    challenges: Sequence[ChallengeStep],
    provided_text: Optional[str] = None,
    expected_document_type: Optional[DocumentType] = None,
) -> PipelineResult:
    """CPU-bound; call from a worker thread, not the event loop."""
    doc_faces = face_engine.detect_faces(document_bgr)
    frame_faces = [face_engine.detect_faces(f) for f in live_frames]

    primary_idx = _pick_primary_frame(frame_faces)
    primary = live_frames[primary_idx]

    observations = [liveness_service.observe(f, faces) for f, faces in zip(live_frames, frame_faces)]
    active_passed, steps = liveness_service.evaluate_challenges(observations, challenges)

    liveness = liveness_service.evaluate_live_frame(primary, frame_faces[primary_idx])
    liveness = liveness.model_copy(
        update={"active_challenges_passed": active_passed, "completed_challenges": steps}
    )

    return PipelineResult(
        face_match=face_matcher.compare_detected(
            document_bgr, doc_faces, primary, frame_faces[primary_idx]
        ),
        liveness=liveness,
        deepfake=deepfake_detector.evaluate_frame(primary),
        document=ocr_service.verify_document(
            document_bgr, provided_text, expected_type=expected_document_type, faces=doc_faces
        ),
        identity_consistent=_identity_consistent(live_frames, frame_faces, primary_idx),
        max_faces_in_frames=max((len(f) for f in frame_faces), default=0),
        frames_analyzed=len(live_frames),
    )
