from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.document import DocumentType


class ChallengeType(str, Enum):
    TURN_HEAD_LEFT = "TURN_HEAD_LEFT"
    TURN_HEAD_RIGHT = "TURN_HEAD_RIGHT"
    BLINK_EYES = "BLINK_EYES"
    SMILE = "SMILE"
    # Reserved: needs audio / lip-sync analysis, which is not implemented.
    # Never assigned to sessions and never auto-passed.
    READ_VERIFICATION_CODE = "READ_VERIFICATION_CODE"


class SessionStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFIED = "VERIFIED"
    PENDING_REVIEW = "PENDING_REVIEW"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


TERMINAL_SESSION_STATUSES = {
    SessionStatus.VERIFIED,
    SessionStatus.PENDING_REVIEW,
    SessionStatus.FAILED,
    SessionStatus.EXPIRED,
}


class ChallengeStep(BaseModel):
    step_id: int
    challenge_type: ChallengeType
    instruction: str
    target_value: Optional[str] = None
    time_limit_seconds: int = 15
    is_completed: bool = False
    failure_reason: Optional[str] = None


class CreateSessionRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:@\-]+$",
        description="Unique customer ID in client system",
        json_schema_extra={"example": "user_89410"},
    )
    callback_url: Optional[str] = Field(
        None, description="Default HTTPS webhook URL for the result of this session"
    )
    document_type: Optional[DocumentType] = Field(
        DocumentType.NATIONAL_ID, description="Expected document type"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary client metadata")

    @field_validator("metadata")
    @classmethod
    def _limit_metadata(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if len(value) > 32:
            raise ValueError("metadata may contain at most 32 keys")
        if len(str(value)) > 4096:
            raise ValueError("metadata is too large (max ~4 KB)")
        return value


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    challenges: List[ChallengeStep]
    verification_code: str = Field(
        ...,
        description="Code shown to the applicant. Informational only: spoken-code "
        "verification is not implemented.",
    )
    websocket_stream_url: Optional[str] = None
    upload_url: str
    verification_id: Optional[str] = None


class SessionRecord(SessionResponse):
    """Server-side session state. Never returned directly to clients."""

    owner_key_id: str
    callback_url: Optional[str] = None
    document_type: Optional[DocumentType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_response(self) -> SessionResponse:
        return SessionResponse.model_validate(self.model_dump(include=set(SessionResponse.model_fields)))
