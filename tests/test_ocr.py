import numpy as np

from app.schemas.document import DocumentType, TextSource
from app.services.ocr_service import DocumentOCRService, mrz_check_digit, verhoeff_valid


class StubFaceEngine:
    def __init__(self, faces=1):
        self.faces = faces

    def detect_faces(self, img):
        return [object()] * self.faces


class StubOCR:
    name = "stub"

    def __init__(self, text):
        self.text = text

    def read(self, img):
        return self.text


svc = DocumentOCRService(engine=StubFaceEngine())
SHARP_IMG = np.random.default_rng(0).integers(40, 210, (380, 600, 3), dtype=np.uint8)

AADHAAR_TEXT = """GOVERNMENT OF INDIA
Unique Identification Authority of India
Rahul Sharma
DOB: 15/08/1994
MALE
9841 2341 5679"""

ICAO_SPECIMEN_MRZ = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
)


# ---------------------------------------------------------------- checksums
def test_verhoeff_reference_values():
    assert verhoeff_valid("2363")
    assert not verhoeff_valid("2364")
    assert verhoeff_valid("9841 2341 5679")
    assert not verhoeff_valid("9841 2341 5678")


def test_mrz_check_digit_icao_specimen():
    assert mrz_check_digit("L898902C3") == 6
    assert mrz_check_digit("740812") == 2
    assert mrz_check_digit("120415") == 9


# ------------------------------------------------------------------ parsing
def test_parse_aadhaar_card():
    doc_type, extracted, flags = svc.parse_document_text(AADHAAR_TEXT)
    assert doc_type == DocumentType.AADHAAR
    assert extracted.document_number == "9841 2341 5679"
    assert extracted.document_number_checksum_valid is True
    assert extracted.full_name == "Rahul Sharma"
    assert extracted.date_of_birth == "15/08/1994"
    assert extracted.gender == "MALE"
    assert flags == []


def test_aadhaar_with_bad_checksum_is_flagged():
    _, extracted, flags = svc.parse_document_text(AADHAAR_TEXT.replace("5679", "5678"))
    assert extracted.document_number_checksum_valid is False
    assert "DOCUMENT_NUMBER_CHECKSUM_INVALID" in flags


def test_parse_pan_card():
    text = "INCOME TAX DEPARTMENT\nGOVT. OF INDIA\nPermanent Account Number Card\nABCPE1234F\nANAND KUMAR\nDOB: 01/01/1990"
    doc_type, extracted, flags = svc.parse_document_text(text)
    assert doc_type == DocumentType.PAN_CARD
    assert extracted.document_number == "ABCPE1234F"
    assert extracted.full_name == "Anand Kumar"
    assert flags == []


def test_pan_with_invalid_holder_type_is_not_accepted():
    _, extracted, flags = svc.parse_document_text("INCOME TAX DEPARTMENT\nABCDE1234F\nANAND KUMAR")
    assert extracted.document_number is None
    assert "DOCUMENT_NUMBER_NOT_DETECTED" in flags


def test_passport_mrz_is_validated_and_expiry_checked():
    doc_type, extracted, flags = svc.parse_document_text(ICAO_SPECIMEN_MRZ)
    assert doc_type == DocumentType.PASSPORT
    assert extracted.document_number == "L898902C3"
    assert extracted.document_number_checksum_valid is True
    assert extracted.expiry_date == "2012-04-15"
    assert "DOCUMENT_EXPIRED" in flags


def test_tampered_mrz_fails_check_digit():
    tampered = ICAO_SPECIMEN_MRZ.replace("L898902C36", "L898903C36")
    _, extracted, flags = svc.parse_document_text(tampered)
    assert extracted.document_number_checksum_valid is False
    assert "MRZ_CHECK_DIGIT_INVALID" in flags


def test_gender_parsing():
    _, e, _ = svc.parse_document_text("Name: Asha Rao\nGender: TRANSGENDER\n2363 2363 2363")
    assert e.gender == "TRANSGENDER"  # was previously mapped to FEMALE
    _, e, _ = svc.parse_document_text("Name: Asha Rao\nSex: F")
    assert e.gender == "FEMALE"
    _, e, _ = svc.parse_document_text("Name: Asha Rao\nBlock M Sector 4")
    assert e.gender is None  # a stray "M" is not a gender


def test_name_label_is_parsed():
    _, e, _ = svc.parse_document_text("AADHAAR\nName: Demo Applicant\n9841 2341 5679")
    assert e.full_name == "Demo Applicant"


def test_name_label_survives_ocr_noise_prefix():
    # Real Tesseract output for the sample card (noise from the portrait area).
    text = "GOVERNMENT OF INDIA - AADHAAR\n». f ‘Name: Demo Applicant\n>) DOB: 15/08/1994\n| 9841 2341 5679"
    _, e, flags = svc.parse_document_text(text)
    assert e.full_name == "Demo Applicant"
    assert flags == []


def test_expected_type_mismatch_is_flagged():
    _, _, flags = svc.parse_document_text(AADHAAR_TEXT, preferred_type=DocumentType.PASSPORT)
    assert "DOCUMENT_TYPE_MISMATCH" in flags


# ------------------------------------------------------------- verification
def test_no_text_available_is_flagged_not_invented():
    # Regression: missing text used to be replaced with a hard-coded valid Aadhaar.
    result = svc.verify_document(SHARP_IMG, provided_text=None)
    assert result.text_source == TextSource.NONE
    assert "OCR_TEXT_UNAVAILABLE" in result.tamper_flags
    assert result.is_valid is False
    assert result.extracted_data.document_number is None


def test_client_text_is_used_when_no_server_ocr():
    result = svc.verify_document(SHARP_IMG, provided_text=AADHAAR_TEXT)
    assert result.text_source == TextSource.CLIENT_PROVIDED
    assert result.is_valid, result.tamper_flags


def test_server_ocr_takes_precedence_and_mismatch_is_flagged():
    ocr_svc = DocumentOCRService(engine=StubFaceEngine(), ocr_engine=StubOCR(AADHAAR_TEXT))
    result = ocr_svc.verify_document(SHARP_IMG, provided_text=AADHAAR_TEXT.replace("9841 2341 5679", "2363 2363 2363"))
    assert result.text_source == TextSource.SERVER_OCR
    assert result.extracted_data.document_number == "9841 2341 5679"
    assert "CLIENT_TEXT_MISMATCH" in result.tamper_flags


def test_document_without_face_is_flagged():
    result = DocumentOCRService(engine=StubFaceEngine(faces=0)).verify_document(SHARP_IMG, AADHAAR_TEXT)
    assert "NO_FACE_ON_DOCUMENT" in result.tamper_flags
    assert not result.is_valid


def test_blank_image_is_blurry():
    result = svc.verify_document(np.full((380, 600, 3), 240, np.uint8), AADHAAR_TEXT)
    assert result.quality.is_blurry
    assert "IMAGE_TOO_BLURRY" in result.tamper_flags
