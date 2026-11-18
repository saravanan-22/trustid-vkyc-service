import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ChallengeType(str, Enum):
    TURN_HEAD_LEFT = "TURN_HEAD_LEFT"
    TURN_HEAD_RIGHT = "TURN_HEAD_RIGHT"
    BLINK_EYES = "BLINK_EYES"
    SMILE = "SMILE"
    READ_VERIFICATION_CODE = "READ_VERIFICATION_CODE"


class SessionStatus(str, Enum):
    INITIALIZED = "INITIALIZED"
    IN_PROGRESS = "IN_PROGRESS"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class ChallengeStep(BaseModel):
    step_id: int
    challenge_type: ChallengeType
    instruction: str
    target_value: Optional[str] = None
    time_limit_seconds: int = 15
    is_completed: bool = False


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., description="Unique customer ID in client system", json_schema_extra={"example": "user_89410"})
    callback_url: Optional[str] = Field(None, description="Webhook URL for async result notification")
    document_type: Optional[str] = Field("NATIONAL_ID", description="Expected document type")
    metadata: Optional[dict] = Field(default_factory=dict, description="Arbitrary client metadata")


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    status: SessionStatus
    created_at: datetime
    expires_at: datetime
    challenges: List[ChallengeStep]
    verification_code: str
    websocket_stream_url: Optional[str] = None
    upload_url: str
