from datetime import date
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    AADHAAR = "AADHAAR"
    PAN_CARD = "PAN_CARD"
    PASSPORT = "PASSPORT"
    DRIVING_LICENSE = "DRIVING_LICENSE"
    NATIONAL_ID = "NATIONAL_ID"


class ExtractedDocumentData(BaseModel):
    document_type: DocumentType
    document_number: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    expiry_date: Optional[str] = None
    mrz_string: Optional[str] = None
    raw_text: Optional[str] = None


class DocumentQualityMetrics(BaseModel):
    sharpness_score: float = Field(..., description="Laplacian sharpness score")
    is_blurry: bool
    glare_detected: bool
    corner_edges_detected: bool
    photo_box_detected: bool


class DocumentVerificationResult(BaseModel):
    is_valid: bool
    document_type: DocumentType
    extracted_data: ExtractedDocumentData
    quality: DocumentQualityMetrics
    tamper_flags: List[str] = Field(default_factory=list)
    confidence: float
