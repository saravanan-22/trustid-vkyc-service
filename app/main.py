import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.api import api_router
from app.core.config import PROJECT_ROOT, settings
from app.core.middleware import BodySizeLimitMiddleware
from app.services.face_engine import ModelUnavailableError, face_engine
from app.services.ocr_service import ocr_service
from app.services.session_store import InMemorySessionStore, session_store

logger = logging.getLogger("trustid")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not face_engine.load():
        logger.error("Starting in DEGRADED mode: %s", face_engine.load_error)
    workers = int(os.getenv("WEB_CONCURRENCY", "1") or 1)
    if isinstance(session_store, InMemorySessionStore) and workers > 1:
        logger.warning(
            "In-memory session store with %d workers: sessions will not be shared "
            "between workers. Set REDIS_URL.", workers
        )
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
# TrustID: Video KYC verification service (prototype)

REST API for document checks, active liveness challenges, 1:1 face matching
and a signed, hash-chained audit trail.

* **Face detection / recognition:** YuNet + SFace neural networks (OpenCV Model Zoo).
* **Active liveness:** randomized head-turn / blink / smile challenges checked from facial landmarks across frames.
* **Passive liveness and synthetic-media signals:** heuristics only. They route cases to review; they are not certified PAD or deepfake models.
* **Documents:** server-side Tesseract OCR when installed, Aadhaar Verhoeff / PAN structure / passport MRZ check-digit validation.
* **Audit:** SHA-256 over the full dossier, hash-chained and HMAC-signed, re-verified on every read.

This is a reference implementation, not a certified KYC product.
    """,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=False,  # API-key auth; no cookies involved
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-KEY", "Content-Type"],
    )
app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.MAX_REQUEST_BYTES)


@app.exception_handler(ModelUnavailableError)
async def _model_unavailable(_: Request, exc: ModelUnavailableError):
    return JSONResponse(status_code=503, content={"detail": f"Face models unavailable: {exc}"})


app.include_router(api_router, prefix=settings.API_V1_STR)

static_dir = PROJECT_ROOT / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health", tags=["System"])
def health_check():
    """Reports which components are actually loaded and usable."""
    models_ready = face_engine.is_ready
    return JSONResponse(
        status_code=200 if models_ready else 503,
        content={
            "status": "healthy" if models_ready else "degraded",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "components": {
                "face_models": "ready" if models_ready else "unavailable",
                "server_ocr": ocr_service.ocr_engine.name if ocr_service.ocr_engine else "unavailable",
                "session_store": type(session_store).__name__,
            },
        },
    )


@app.get("/", include_in_schema=False)
def serve_portal():
    """Serves the interactive demo portal."""
    index_file = static_dir / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "TrustID VKYC API is running. See /docs."})
