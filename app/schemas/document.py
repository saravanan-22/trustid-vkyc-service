from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    AADHAAR = "AADHAAR"
    PAN_CARD = "PAN_CARD"
    PASSPORT = "PASSPORT"
    DRIVING_LICENSE = "DRIVING_LICENSE"
    NATIONAL_ID = "NATIONAL_ID"


class TextSource(str, Enum):
    SERVER_OCR = "SERVER_OCR"
    CLIENT_PROVIDED = "CLIENT_PROVIDED"
    NONE = "NONE"


class ExtractedDocumentData(BaseModel):
    document_type: DocumentType
    document_number: Optional[str] = None
    document_number_checksum_valid: Optional[bool] = Field(
        None,
        description="Aadhaar Verhoeff check, PAN structure check, or passport MRZ check digit. "
        "None when no checksum applies.",
    )
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
    photo_box_detected: bool = Field(..., description="A face was detected on the document")


class DocumentVerificationResult(BaseModel):
    is_valid: bool
    document_type: DocumentType
    text_source: TextSource = TextSource.NONE
    extracted_data: ExtractedDocumentData
    quality: DocumentQualityMetrics
    tamper_flags: List[str] = Field(
        default_factory=list, description="Quality, extraction and consistency issues found"
    )
    confidence: float
