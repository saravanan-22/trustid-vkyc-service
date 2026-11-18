from fastapi import APIRouter
from app.api.v1.endpoints import audit, document, face_match, liveness, sessions, verify

api_router = APIRouter()

api_router.include_router(sessions.router, prefix="/sessions", tags=["Sessions & Challenges"])
api_router.include_router(liveness.router, prefix="/liveness", tags=["Liveness & Anti-Spoofing"])
api_router.include_router(face_match.router, prefix="/face", tags=["1:1 Biometric Face Match"])
api_router.include_router(document.router, prefix="/document", tags=["Document OCR & Tampering"])
api_router.include_router(verify.router, prefix="/verify", tags=["Full End-to-End VKYC"])
api_router.include_router(audit.router, prefix="/audit", tags=["Cryptographic Audit Trail"])
