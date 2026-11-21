"""
ID document checks: image quality, text extraction, field parsing and
checksum validation.

Text comes from server-side Tesseract OCR when it is installed (and
OCR_ENGINE allows it), otherwise from text the API client supplies. If neither
is available the document is flagged; nothing is ever invented.
"""

import logging
import re
import shutil
from datetime import date, datetime, timezone
from typing import List, Optional, Protocol, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.document import (
    DocumentQualityMetrics,
    DocumentType,
    DocumentVerificationResult,
    ExtractedDocumentData,
    TextSource,
)
from app.services.face_engine import Face, face_engine

logger = logging.getLogger(__name__)

# Flags that do not by themselves make a document invalid.
SOFT_FLAGS = {"NAME_UNRESOLVED"}

# ----------------------------------------------------------------- checksums
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    """Aadhaar numbers carry a Verhoeff check digit as their last digit."""
    digits = re.sub(r"\D", "", number)
    if not digits:
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def mrz_check_digit(field: str) -> int:
    """ICAO 9303 check digit (weights 7, 3, 1)."""
    total = 0
    for i, ch in enumerate(field):
        if ch.isdigit():
            value = int(ch)
        elif "A" <= ch <= "Z":
            value = ord(ch) - ord("A") + 10
        else:  # '<' filler
            value = 0
        total += value * (7, 3, 1)[i % 3]
    return total % 10


def _mrz_date(yymmdd: str, future: bool) -> Optional[date]:
    try:
        yy, mm, dd = int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    except ValueError:
        return None
    this_year = datetime.now(timezone.utc).year % 100
    century = 2000 if (future or yy <= this_year) else 1900
    try:
        return date(century + yy, mm, dd)
    except ValueError:
        return None


# PAN: 3 letters, holder-type letter, initial letter, 4 digits, check letter.
PAN_REGEX = re.compile(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]\b")
AADHAAR_REGEX = re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")
INDIAN_PASSPORT_REGEX = re.compile(r"\b[A-Z][0-9]{7}\b")
DOB_REGEX = re.compile(r"\b(0[1-9]|[12]\d|3[01])[-/.](0[1-9]|1[0-2])[-/.](19\d\d|20\d\d)\b")
GENDER_LABEL_REGEX = re.compile(r"\b(?:GENDER|SEX)\s*[:/\-]?\s*(MALE|FEMALE|TRANSGENDER|M|F|T)\b", re.IGNORECASE)
GENDER_WORD_REGEX = re.compile(r"\b(MALE|FEMALE|TRANSGENDER)\b", re.IGNORECASE)
# Not anchored at line start: OCR often prefixes lines with noise from the portrait area.
NAME_LABEL_REGEX = re.compile(r"\bNAME\s*[:\-]\s*([A-Za-z][A-Za-z .']{1,80}?)\s*$", re.IGNORECASE | re.MULTILINE)
# ICAO 9303 TD3 line 2: number(9) check nationality(3) dob(6) check sex expiry(6) check optional(14) check composite.
MRZ_LINE2_REGEX = re.compile(r"^[A-Z0-9<]{9}[0-9][A-Z<]{3}[0-9]{6}[0-9][MFX<][0-9]{6}[0-9][A-Z0-9<]{14}[0-9<][0-9]$")

_GENDER_MAP = {"M": "MALE", "MALE": "MALE", "F": "FEMALE", "FEMALE": "FEMALE", "T": "TRANSGENDER", "TRANSGENDER": "TRANSGENDER"}
_NAME_IGNORE = (
    "GOVERNMENT", "INDIA", "INCOME", "TAX", "DEPARTMENT", "REPUBLIC", "CARD", "PASSPORT",
    "AADHAAR", "DOB", "DATE", "BIRTH", "MALE", "FEMALE", "AUTHORITY", "UNIQUE", "PERMANENT",
    "ACCOUNT", "NUMBER", "SPECIMEN",
)


# ----------------------------------------------------------------- OCR engine
class OCREngine(Protocol):
    name: str

    def read(self, img_bgr: np.ndarray) -> str: ...


class TesseractEngine:
    name = "tesseract"

    def __init__(self, languages: str = "eng"):
        import pytesseract  # optional dependency

        self._pytesseract = pytesseract
        self.languages = languages

    @staticmethod
    def is_available() -> bool:
        if shutil.which("tesseract") is None:
            return False
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return True

    def read(self, img_bgr: np.ndarray) -> str:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] < 1200:
            factor = 1200.0 / gray.shape[1]
            gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        return self._pytesseract.image_to_string(gray, lang=self.languages, config="--psm 6")


def build_default_ocr_engine() -> Optional[OCREngine]:
    if settings.OCR_ENGINE == "none":
        return None
    if TesseractEngine.is_available():
        return TesseractEngine(settings.OCR_LANGUAGES)
    if settings.OCR_ENGINE == "tesseract":
        logger.error("OCR_ENGINE=tesseract but the tesseract binary or pytesseract is missing")
    return None


class DocumentOCRService:
    def __init__(self, engine=face_engine, ocr_engine: Optional[OCREngine] = None):
        self.face_engine = engine
        self.ocr_engine = ocr_engine

    # ---------------------------------------------------------------- quality
    def assess_quality(self, img_bgr: np.ndarray, faces: Optional[List[Face]] = None) -> DocumentQualityMetrics:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        glare_ratio = np.sum(gray > 250) / float(gray.size)

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corner_edges_detected = False
        min_area = img_bgr.shape[0] * img_bgr.shape[1] * 0.3
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4 and cv2.contourArea(c) > min_area:
                corner_edges_detected = True
                break

        if faces is None:
            faces = self.face_engine.detect_faces(img_bgr)

        return DocumentQualityMetrics(
            sharpness_score=round(float(sharpness), 2),
            is_blurry=bool(sharpness < 45.0),
            glare_detected=bool(glare_ratio > 0.08),
            corner_edges_detected=corner_edges_detected,
            photo_box_detected=bool(faces),
        )

    # ---------------------------------------------------------------- parsing
    def parse_document_text(
        self, text: str, preferred_type: Optional[DocumentType] = None
    ) -> Tuple[DocumentType, ExtractedDocumentData, List[str]]:
        flags: List[str] = []
        upper = text.upper()
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        mrz_line2 = next(
            (re.sub(r"\s", "", l.upper()) for l in lines if MRZ_LINE2_REGEX.match(re.sub(r"\s", "", l.upper()))),
            None,
        )
        mrz_line1 = next((l.upper() for l in lines if l.upper().startswith("P<")), None)

        # Document type: explicit keywords first, then number patterns.
        if "INCOME TAX" in upper or "PERMANENT ACCOUNT NUMBER" in upper:
            doc_type = DocumentType.PAN_CARD
        elif "AADHAAR" in upper or "UNIQUE IDENTIFICATION" in upper:
            doc_type = DocumentType.AADHAAR
        elif "PASSPORT" in upper or mrz_line1 or mrz_line2:
            doc_type = DocumentType.PASSPORT
        elif "DRIVING LICEN" in upper:
            doc_type = DocumentType.DRIVING_LICENSE
        elif PAN_REGEX.search(upper):
            doc_type = DocumentType.PAN_CARD
        elif AADHAAR_REGEX.search(text):
            doc_type = DocumentType.AADHAAR
        else:
            doc_type = preferred_type or DocumentType.NATIONAL_ID

        doc_number: Optional[str] = None
        checksum_valid: Optional[bool] = None
        expiry: Optional[str] = None

        if doc_type == DocumentType.AADHAAR:
            m = AADHAAR_REGEX.search(text)
            if m:
                doc_number = m.group(0)
                checksum_valid = verhoeff_valid(doc_number)
        elif doc_type == DocumentType.PAN_CARD:
            m = PAN_REGEX.search(upper)
            if m:
                doc_number = m.group(0)
                checksum_valid = True  # structure validated by the regex
        elif doc_type == DocumentType.PASSPORT:
            if mrz_line2:
                number_field = mrz_line2[0:9]
                doc_number = number_field.replace("<", "")
                checks = [
                    mrz_check_digit(number_field) == int(mrz_line2[9]) if mrz_line2[9].isdigit() else False,
                    mrz_check_digit(mrz_line2[13:19]) == int(mrz_line2[19]) if mrz_line2[19].isdigit() else False,
                    mrz_check_digit(mrz_line2[21:27]) == int(mrz_line2[27]) if mrz_line2[27].isdigit() else False,
                ]
                checksum_valid = all(checks)
                if not checksum_valid:
                    flags.append("MRZ_CHECK_DIGIT_INVALID")
                expiry_date = _mrz_date(mrz_line2[21:27], future=True)
                if expiry_date:
                    expiry = expiry_date.isoformat()
                    if expiry_date < datetime.now(timezone.utc).date():
                        flags.append("DOCUMENT_EXPIRED")
            else:
                m = INDIAN_PASSPORT_REGEX.search(upper)
                if m:
                    doc_number = m.group(0)

        if doc_number is None:
            flags.append("DOCUMENT_NUMBER_NOT_DETECTED")
        elif checksum_valid is False and "MRZ_CHECK_DIGIT_INVALID" not in flags:
            flags.append("DOCUMENT_NUMBER_CHECKSUM_INVALID")

        if preferred_type and preferred_type != DocumentType.NATIONAL_ID and preferred_type != doc_type:
            flags.append("DOCUMENT_TYPE_MISMATCH")

        dob_match = DOB_REGEX.search(text)
        gender = None
        g = GENDER_LABEL_REGEX.search(text) or GENDER_WORD_REGEX.search(text)
        if g:
            gender = _GENDER_MAP.get(g.group(1).upper())

        full_name = None
        label = NAME_LABEL_REGEX.search(text)
        if label:
            full_name = " ".join(label.group(1).split()).title()
        else:
            for line in lines:
                if any(k in line.upper() for k in _NAME_IGNORE):
                    continue
                if len(line.split()) >= 2 and re.fullmatch(r"[A-Za-z .']+", line):
                    full_name = " ".join(line.split()).title()
                    break
        if not full_name:
            flags.append("NAME_UNRESOLVED")

        extracted = ExtractedDocumentData(
            document_type=doc_type,
            document_number=doc_number,
            document_number_checksum_valid=checksum_valid,
            full_name=full_name,
            date_of_birth=dob_match.group(0) if dob_match else None,
            gender=gender,
            expiry_date=expiry,
            mrz_string="\n".join(x for x in (mrz_line1, mrz_line2) if x) or None,
            raw_text=text[:500],
        )
        return doc_type, extracted, flags

    # ------------------------------------------------------------------ main
    def verify_document(
        self,
        img_bgr: np.ndarray,
        provided_text: Optional[str] = None,
        expected_type: Optional[DocumentType] = None,
        faces: Optional[List[Face]] = None,
    ) -> DocumentVerificationResult:
        quality = self.assess_quality(img_bgr, faces=faces)
        flags: List[str] = []
        if quality.is_blurry:
            flags.append("IMAGE_TOO_BLURRY")
        if quality.glare_detected:
            flags.append("EXCESSIVE_GLARE")
        if not quality.photo_box_detected:
            flags.append("NO_FACE_ON_DOCUMENT")

        provided_text = (provided_text or "").strip() or None
        text: Optional[str] = None
        source = TextSource.NONE
        if self.ocr_engine is not None:
            try:
                text = self.ocr_engine.read(img_bgr).strip() or None
                source = TextSource.SERVER_OCR if text else TextSource.NONE
            except Exception:  # OCR failure should degrade, not crash the request
                logger.exception("Server-side OCR failed")
                text = None
        if text is None and provided_text:
            text, source = provided_text, TextSource.CLIENT_PROVIDED

        if text is None:
            flags.append("OCR_TEXT_UNAVAILABLE")
            doc_type = expected_type or DocumentType.NATIONAL_ID
            extracted = ExtractedDocumentData(document_type=doc_type)
        else:
            doc_type, extracted, text_flags = self.parse_document_text(text, preferred_type=expected_type)
            flags.extend(text_flags)
            if source == TextSource.SERVER_OCR and provided_text:
                _, client_extracted, _ = self.parse_document_text(provided_text, preferred_type=expected_type)
                if (
                    client_extracted.document_number
                    and extracted.document_number
                    and re.sub(r"\s", "", client_extracted.document_number)
                    != re.sub(r"\s", "", extracted.document_number)
                ):
                    flags.append("CLIENT_TEXT_MISMATCH")

        hard_flags = [f for f in flags if f not in SOFT_FLAGS]
        confidence = max(0.0, 0.95 - 0.2 * len(hard_flags) - 0.05 * (len(flags) - len(hard_flags)))

        return DocumentVerificationResult(
            is_valid=not hard_flags,
            document_type=doc_type,
            text_source=source,
            extracted_data=extracted,
            quality=quality,
            tamper_flags=flags,
            confidence=round(confidence, 4),
        )


ocr_service = DocumentOCRService(ocr_engine=build_default_ocr_engine())
