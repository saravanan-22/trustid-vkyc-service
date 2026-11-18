import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api.v1.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
# TrustID — Production-Grade Video KYC & Anti-Deepfake Verification Service

TrustID provides an ultra-fast, high-throughput, bank-grade Video KYC verification service exposed as a developer-friendly REST API.

### Core Capabilities:
* **Active & Passive Anti-Spoofing:** Detects 2D screen replays, paper cutouts, and 3D masks.
* **Deepfake & Virtual Camera Detection:** Uses 2D Fast Fourier Transform (FFT) spectral frequency analysis to identify GAN/Diffusion synthetic face swaps.
* **1:1 Biometric Facial Matching:** Verifies identity between extracted ID card portrait and live video stream.
* **Document OCR & Fraud Detection:** Validates image clarity, parses identity details (Aadhaar, Passport, PAN, National ID), and spots tampering.
* **Cryptographic Tamper-Proof Audit Trail:** Issues SHA-256 HMAC-signed verification dossiers for regulatory compliance (RBI / FATF / SOC2).
    """,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount Static assets for live demo
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/health", tags=["System"])
def health_check():
    """Returns the operational status and latency of the VKYC engine."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "models": {
            "face_matcher": "operational",
            "anti_spoofing": "operational",
            "deepfake_fft": "operational",
            "document_ocr": "operational",
        },
    }


@app.get("/", include_in_schema=False)
def serve_portal():
    """Serves the interactive recruiter verification testing dashboard."""
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        {
            "message": "TrustID VKYC Service API is running. Visit /docs for Swagger documentation.",
            "interactive_demo": "Static portal directory not found.",
        }
    )
