from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.document import DocumentVerificationResult
from app.schemas.session import ChallengeStep


class VerificationVerdict(str, Enum):
    APPROVED = "APPROVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class LivenessMetrics(BaseModel):
    passive_score: float = Field(..., description="Texture & anti-spoof score (0.0 to 1.0)")
    is_live_passive: bool
    active_challenges_passed: bool
    completed_challenges: List[ChallengeStep] = Field(default_factory=list)
    detected_spoof_type: Optional[str] = Field(None, description="SCREEN_REPLAY, PAPER_MASK, 3D_MASK, or NONE")


class DeepfakeMetrics(BaseModel):
    synthetic_risk_score: float = Field(..., description="FFT frequency anomaly score (0.0 to 1.0, lower is safer)")
    is_deepfake_suspected: bool
    virtual_camera_detected: bool
    spectral_artifact_score: float


class FaceMatchMetrics(BaseModel):
    match_score: float = Field(..., description="Biometric similarity score (0.0 to 1.0)")
    is_match: bool
    face_detected_in_id: bool
    face_detected_in_live: bool
    confidence_interval: float


class VerificationDossier(BaseModel):
    verification_id: str
    session_id: str
    user_id: str
    verdict: VerificationVerdict
    overall_confidence: float
    timestamp: datetime
    face_match: FaceMatchMetrics
    liveness: LivenessMetrics
    deepfake: DeepfakeMetrics
    document: DocumentVerificationResult
    rejection_reasons: List[str] = Field(default_factory=list)
    tamper_proof_audit_hash: str
    signature: str


class FullVerificationResponse(BaseModel):
    success: bool
    verification_id: str
    verdict: VerificationVerdict
    summary: str
    dossier: VerificationDossier
