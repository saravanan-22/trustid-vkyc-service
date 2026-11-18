import os
from typing import List
from pydantic import BaseModel


class Settings(BaseModel):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "TrustID VKYC Verification Service")
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    API_KEY_HEADER_NAME: str = "X-API-KEY"
    DEFAULT_API_KEY: str = os.getenv("DEFAULT_API_KEY", "trustid_live_demo_key_2026")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "trustid-super-secret-hmac-audit-signing-key-v1")
    
    # Biometric Thresholds
    FACE_MATCH_THRESHOLD: float = float(os.getenv("FACE_MATCH_THRESHOLD", "0.70"))
    PASSIVE_LIVENESS_THRESHOLD: float = float(os.getenv("PASSIVE_LIVENESS_THRESHOLD", "0.65"))
    DEEPFAKE_MAX_RISK_SCORE: float = float(os.getenv("DEEPFAKE_MAX_RISK_SCORE", "0.40"))
    BLUR_MIN_LAPLACIAN_VAR: float = float(os.getenv("BLUR_MIN_LAPLACIAN_VAR", "40.0"))
    
    # Session Configuration
    SESSION_EXPIRY_SECONDS: int = int(os.getenv("SESSION_EXPIRY_SECONDS", "300"))
    REQUIRED_ACTIVE_STEPS: int = int(os.getenv("REQUIRED_ACTIVE_STEPS", "2"))
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Storage
    DATA_DIR: str = os.getenv(
        "TRUSTID_DATA_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"),
    )
    AUDIT_STORAGE_PATH: str = os.getenv(
        "TRUSTID_AUDIT_PATH",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "audits"),
    )


settings = Settings()
os.makedirs(settings.AUDIT_STORAGE_PATH, exist_ok=True)

