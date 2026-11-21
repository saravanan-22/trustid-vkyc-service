"""
Application settings.

All values are read from environment variables (or a local `.env` file).
Secrets have **no defaults**: the service refuses to start until real values
are configured. See `.env.example`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Values that shipped in earlier versions of this repo. They are public, so they
# must never be accepted as real secrets.
_KNOWN_LEAKED_SECRETS = {
    "trustid_live_demo_key_2026",
    "trustid-super-secret-hmac-audit-signing-key-v1",
}

MIN_SECRET_LENGTH = 32
MIN_API_KEY_LENGTH = 24


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    PROJECT_NAME: str = "TrustID VKYC Verification Service"
    VERSION: str = "1.1.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["development", "test", "production"] = "development"

    # ------------------------------------------------------------------ secrets
    # Comma-separated list of accepted API keys (allows rotation).
    TRUSTID_API_KEYS: Annotated[List[str], NoDecode] = Field(default_factory=list)
    # HMAC key used to sign audit dossiers. Never shared with API clients.
    AUDIT_SIGNING_SECRET: str = ""
    # HMAC key used to sign outgoing webhooks. Shared with integrators so they
    # can verify callbacks, so it must be different from AUDIT_SIGNING_SECRET.
    WEBHOOK_SIGNING_SECRET: str = ""

    # ------------------------------------------------------------------ models
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    FACE_DETECTOR_MODEL: str = "face_detection_yunet_2023mar.onnx"
    FACE_RECOGNIZER_MODEL: str = "face_recognition_sface_2021dec.onnx"
    FACE_DETECTION_MIN_CONFIDENCE: float = 0.80
    # Images are downscaled so the longest side is at most this before detection.
    FACE_DETECTION_MAX_SIDE: int = 1280

    # -------------------------------------------------------------- thresholds
    # SFace cosine similarity. 0.363 is the threshold published with the model
    # (OpenCV Zoo, evaluated on LFW). Tune on your own data before relying on it.
    FACE_MATCH_THRESHOLD: float = 0.363
    PASSIVE_LIVENESS_THRESHOLD: float = 0.65
    DEEPFAKE_MAX_RISK_SCORE: float = 0.40
    BLUR_MIN_LAPLACIAN_VAR: float = 40.0
    # FFT synthetic-media heuristic bounds (high/mid frequency energy ratio).
    FFT_RATIO_MIN: float = 0.02
    FFT_RATIO_MAX: float = 0.60
    FFT_SPECTRAL_VARIANCE_MAX: float = 1.8
    # Set true if the client mirrors frames (selfie view) before uploading them.
    CAMERA_FRAMES_MIRRORED: bool = False

    # ---------------------------------------------------------------- sessions
    SESSION_EXPIRY_SECONDS: int = 300
    REQUIRED_ACTIVE_STEPS: int = 2
    # When set, sessions are stored in Redis (required for >1 worker/replica).
    REDIS_URL: Optional[str] = None

    # ------------------------------------------------------------ upload limits
    MAX_REQUEST_BYTES: int = 40 * 1024 * 1024
    MAX_IMAGE_BYTES: int = 8 * 1024 * 1024
    MAX_IMAGE_PIXELS: int = 25_000_000
    MAX_LIVE_FRAMES: int = 30

    # ---------------------------------------------------------------- webhooks
    WEBHOOK_ALLOW_HTTP: bool = False
    # Only for local testing. Allows callbacks to localhost / private networks.
    WEBHOOK_ALLOW_PRIVATE_NETWORKS: bool = False
    WEBHOOK_TIMEOUT_SECONDS: float = 10.0

    # --------------------------------------------------------------------- OCR
    # auto: use Tesseract when installed, otherwise rely on client-supplied text.
    OCR_ENGINE: Literal["auto", "tesseract", "none"] = "auto"
    OCR_LANGUAGES: str = "eng"

    # -------------------------------------------------------------------- CORS
    # Empty = same-origin only (the bundled demo portal needs no CORS).
    BACKEND_CORS_ORIGINS: Annotated[List[str], NoDecode] = Field(default_factory=list)

    # ----------------------------------------------------------------- storage
    DATA_DIR: Path = PROJECT_ROOT / "data"
    AUDIT_STORAGE_PATH: Optional[Path] = None

    @field_validator("TRUSTID_API_KEYS", "BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def _split_csv(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _validate_secrets(self) -> "Settings":
        problems = []

        if not self.TRUSTID_API_KEYS:
            problems.append("TRUSTID_API_KEYS is not set")
        for key in self.TRUSTID_API_KEYS:
            if key in _KNOWN_LEAKED_SECRETS:
                problems.append("TRUSTID_API_KEYS contains the publicly known demo key")
            elif len(key) < MIN_API_KEY_LENGTH:
                problems.append(f"each API key must be at least {MIN_API_KEY_LENGTH} characters")

        for name in ("AUDIT_SIGNING_SECRET", "WEBHOOK_SIGNING_SECRET"):
            value = getattr(self, name)
            if not value:
                problems.append(f"{name} is not set")
            elif value in _KNOWN_LEAKED_SECRETS:
                problems.append(f"{name} is the publicly known demo secret")
            elif len(value) < MIN_SECRET_LENGTH:
                problems.append(f"{name} must be at least {MIN_SECRET_LENGTH} characters")

        if (
            self.AUDIT_SIGNING_SECRET
            and self.AUDIT_SIGNING_SECRET == self.WEBHOOK_SIGNING_SECRET
        ):
            problems.append("AUDIT_SIGNING_SECRET and WEBHOOK_SIGNING_SECRET must differ")

        if problems:
            raise ValueError(
                "Invalid TrustID configuration: "
                + "; ".join(dict.fromkeys(problems))
                + ". Copy .env.example to .env and generate values with "
                "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"`."
            )

        if self.AUDIT_STORAGE_PATH is None:
            self.AUDIT_STORAGE_PATH = self.DATA_DIR / "audits"
        return self

    @property
    def face_detector_path(self) -> Path:
        return self.MODEL_DIR / self.FACE_DETECTOR_MODEL

    @property
    def face_recognizer_path(self) -> Path:
        return self.MODEL_DIR / self.FACE_RECOGNIZER_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
