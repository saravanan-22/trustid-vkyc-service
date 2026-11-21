from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.images import read_image
from app.core.security import verify_api_key
from app.schemas.document import DocumentType, DocumentVerificationResult
from app.services.ocr_service import ocr_service

router = APIRouter()


@router.post("/verify", response_model=DocumentVerificationResult)
async def verify_id_document(
    document_image: UploadFile = File(..., description="ID card or passport image"),
    extracted_text: Optional[str] = Form(
        None, max_length=10_000, description="Document text from your own OCR (used when server OCR is unavailable)"
    ),
    expected_type: Optional[DocumentType] = Form(None, description="Expected document type"),
    key_id: str = Depends(verify_api_key),
):
    """
    Checks image quality and face presence, extracts fields (server-side OCR when
    installed, otherwise the supplied text) and validates document-number checksums.
    """
    img = await read_image(document_image, "document_image")
    return await run_in_threadpool(ocr_service.verify_document, img, extracted_text, expected_type)
