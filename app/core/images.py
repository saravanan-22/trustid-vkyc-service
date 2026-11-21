import io
from typing import List

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import settings


async def read_upload_bytes(upload: UploadFile, field_name: str) -> bytes:
    """Reads an uploaded file, enforcing the per-image size limit."""
    data = await upload.read(settings.MAX_IMAGE_BYTES + 1)
    if len(data) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"'{field_name}' exceeds the {settings.MAX_IMAGE_BYTES} byte limit.",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{field_name}' is empty.",
        )
    return data


def decode_image(data: bytes, field_name: str) -> np.ndarray:
    """
    Decodes image bytes into a BGR array. The pixel count is checked from the
    header first, so a small "decompression bomb" file cannot exhaust memory.
    """
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{field_name}' is not a supported image.",
        )
    if width * height > settings.MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"'{field_name}' exceeds the {settings.MAX_IMAGE_PIXELS} pixel limit.",
        )

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{field_name}' could not be decoded.",
        )
    return img


async def read_image(upload: UploadFile, field_name: str) -> np.ndarray:
    return decode_image(await read_upload_bytes(upload, field_name), field_name)


async def read_frames(uploads: List[UploadFile], field_name: str) -> List[np.ndarray]:
    if not uploads:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"At least one '{field_name}' image is required.",
        )
    if len(uploads) > settings.MAX_LIVE_FRAMES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many '{field_name}' images (max {settings.MAX_LIVE_FRAMES}).",
        )
    return [await read_image(u, f"{field_name}[{i}]") for i, u in enumerate(uploads)]
