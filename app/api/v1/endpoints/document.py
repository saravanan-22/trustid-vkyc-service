from typing import Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from app.core.security import verify_api_key
from app.schemas.document import DocumentVerificationResult
from app.services.ocr_service import ocr_service

router = APIRouter()


def decode_image_bytes(data: bytes) -> np.ndarray:
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format.",
        )
    return img


@router.post("/verify", response_model=DocumentVerificationResult)
async def verify_id_document(
    document_image: UploadFile = File(..., description="ID card or passport image"),
    extracted_text: Optional[str] = Form(
        None, description="Optional pre-extracted OCR text string from client OCR pipeline"
    ),
    api_key: str = Depends(verify_api_key),
):
    """
    Validates document image quality, extracts structured fields, and inspects for tampering.
    """
    content = await document_image.read()
    img_bgr = decode_image_bytes(content)

    result = ocr_service.verify_document(img_bgr, provided_text=extracted_text)
    return result
