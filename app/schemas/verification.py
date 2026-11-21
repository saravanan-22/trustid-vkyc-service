from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.document import DocumentVerificationResult
from app.schemas.session import ChallengeStep


class VerificationVerdict(str, Enum):
    APPROVED = "APPROVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class LivenessMetrics(BaseModel):
    passive_score: float = Field(..., description="Heuristic texture score (0.0 to 1.0)")
    is_live_passive: bool
    active_challenges_passed: bool
    completed_challenges: List[ChallengeStep] = Field(default_factory=list)
    detected_spoof_type: Optional[str] = Field(
        None,
        description="NONE, NO_FACE_DETECTED, BLURRY_OR_DEFOCUSED, SCREEN_GLARE_SUSPECTED or ATYPICAL_SKIN_TEXTURE",
    )
    faces_detected: int = Field(0, description="Number of faces in the evaluated frame")


class DeepfakeMetrics(BaseModel):
    synthetic_risk_score: float = Field(
        ...,
        description="Heuristic FFT spectral anomaly score (0.0 to 1.0, lower is safer). "
        "Not a trained deepfake classifier; used only to route to manual review.",
    )
    is_deepfake_suspected: bool
    virtual_camera_detected: bool = Field(
        ..., description="Heuristic: unnaturally flat chroma noise, typical of synthetic or frozen frames"
    )
    spectral_artifact_score: float


class FaceMatchMetrics(BaseModel):
    match_score: float = Field(..., description="SFace cosine similarity, clamped to 0.0 to 1.0")
    is_match: bool
    face_detected_in_id: bool
    face_detected_in_live: bool
    faces_in_live: int = Field(0, description="Number of faces detected in the live frame")
    threshold: float = Field(..., description="Similarity threshold used for is_match")


class VerificationDossier(BaseModel):
    verification_id: str
    session_id: str
    user_id: str
    verdict: VerificationVerdict
    policy_version: str
    overall_confidence: float
    timestamp: datetime
    frames_analyzed: int
    face_match: FaceMatchMetrics
    liveness: LivenessMetrics
    deepfake: DeepfakeMetrics
    document: DocumentVerificationResult
    rejection_reasons: List[str] = Field(default_factory=list)
    review_reasons: List[str] = Field(default_factory=list)
    previous_hash: str = Field(..., description="Hash of the previous dossier in the audit chain")
    tamper_proof_audit_hash: str = Field(
        ..., description="SHA-256 over the canonical JSON of every other field"
    )
    signature: str = Field(..., description="HMAC-SHA256 of tamper_proof_audit_hash")


class FullVerificationResponse(BaseModel):
    success: bool
    verification_id: str
    verdict: VerificationVerdict
    summary: str
    dossier: VerificationDossier
