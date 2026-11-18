import numpy as np
import pytest
from app.schemas.document import DocumentType
from app.services.ocr_service import ocr_service


def test_parse_aadhaar_card():
    text = """
    GOVERNMENT OF INDIA
    Unique Identification Authority of India
    Rahul Sharma
    DOB: 15/08/1994
    MALE
    9841 2341 5678
    """
    doc_type, extracted, flags = ocr_service.parse_document_text(text)
    assert doc_type == DocumentType.AADHAAR
    assert extracted.document_number == "9841 2341 5678"
    assert extracted.full_name == "Rahul Sharma"
    assert extracted.date_of_birth == "15/08/1994"
    assert extracted.gender == "MALE"
    assert len(flags) == 0


def test_parse_pan_card():
    text = """
    INCOME TAX DEPARTMENT
    GOVT. OF INDIA
    Permanent Account Number Card
    ABCDE1234F
    ANAND KUMAR
    DOB: 01/01/1990
    """
    doc_type, extracted, flags = ocr_service.parse_document_text(text)
    assert doc_type == DocumentType.PAN_CARD
    assert extracted.document_number == "ABCDE1234F"


def test_document_quality_assessment():
    img = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)
    quality = ocr_service.assess_quality(img)
    assert quality.sharpness_score > 0
    assert isinstance(quality.is_blurry, bool)
