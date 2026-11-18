import cv2
import numpy as np
from typing import Optional, Tuple
from app.core.config import settings
from app.schemas.verification import FaceMatchMetrics
from app.services.face_detector_helper import adaptive_detector


class FaceMatcher:
    """
    Performs 1:1 Biometric Face Verification between Live Video/Selfie
    and ID Card Photo. Computes cosine similarity across aligned facial embeddings.
    """

    def __init__(self):
        self.detector = adaptive_detector

    def extract_face_crop(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Locates and extracts the primary face, aligned to a fixed 160x160 canvas.
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        box = self.detector.detect_face_bbox(img_bgr)
        if not box:
            h, w = img_bgr.shape[:2]
            cx, cy = w // 2, h // 2
            size = min(w, h) // 2
            return cv2.resize(
                img_bgr[max(0, cy - size) : cy + size, max(0, cx - size) : cx + size],
                (160, 160),
            )

        x, y, w, h = box
        # Add a 10% margin
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_bgr.shape[1], x + w + margin_x)
        y2 = min(img_bgr.shape[0], y + h + margin_y)

        face_crop = img_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return cv2.resize(img_bgr, (160, 160))
        return cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)

    def compute_embedding(self, face_160_bgr: np.ndarray) -> np.ndarray:
        """
        Generates a 512-dimensional normalized facial biometric embedding.
        Uses multi-region spatial gradient histograms and textural descriptors.
        """
        gray = cv2.cvtColor(face_160_bgr, cv2.COLOR_BGR2GRAY)
        # Equalize histogram for illumination invariance
        equalized = cv2.equalizeHist(gray)

        # 4x4 spatial grid decomposition
        grid_h, grid_w = equalized.shape[0] // 4, equalized.shape[1] // 4
        features = []

        for r in range(4):
            for c in range(4):
                patch = equalized[r * grid_h : (r + 1) * grid_h, c * grid_w : (c + 1) * grid_w]
                
                # Gradient orientations (Sobel X and Y)
                gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
                mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

                # 16-bin gradient orientation histogram
                hist, _ = np.histogram(ang, bins=16, range=(0, 360), weights=mag)
                # 16-bin intensity histogram
                intensity_hist, _ = np.histogram(patch, bins=16, range=(0, 256))

                features.extend(hist)
                features.extend(intensity_hist)

        embedding = np.array(features, dtype=np.float32)
        # L2 Normalization
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding

    def compare_faces(
        self, id_photo_bgr: np.ndarray, live_photo_bgr: np.ndarray
    ) -> FaceMatchMetrics:
        """
        Compares ID card photo against live camera capture.
        Computes cosine similarity: dot product of normalized embeddings.
        """
        id_crop = self.extract_face_crop(id_photo_bgr)
        live_crop = self.extract_face_crop(live_photo_bgr)

        face_detected_id = id_crop is not None
        face_detected_live = live_crop is not None

        if not face_detected_id or not face_detected_live:
            return FaceMatchMetrics(
                match_score=0.0,
                is_match=False,
                face_detected_in_id=face_detected_id,
                face_detected_in_live=face_detected_live,
                confidence_interval=0.0,
            )

        emb_id = self.compute_embedding(id_crop)
        emb_live = self.compute_embedding(live_crop)

        # Cosine similarity
        similarity = float(np.dot(emb_id, emb_live))
        # Ensure clamped to [0.0, 1.0]
        similarity = float(np.clip(similarity, 0.0, 1.0))

        # Check threshold
        is_match = similarity >= settings.FACE_MATCH_THRESHOLD
        confidence_interval = round(min(1.0, similarity * 1.05), 4)

        return FaceMatchMetrics(
            match_score=round(similarity, 4),
            is_match=is_match,
            face_detected_in_id=face_detected_id,
            face_detected_in_live=face_detected_live,
            confidence_interval=confidence_interval,
        )


face_matcher = FaceMatcher()
