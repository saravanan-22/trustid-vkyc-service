"""
Face detection and recognition backed by pretrained neural networks from the
OpenCV Model Zoo (https://github.com/opencv/opencv_zoo):

* YuNet  (face_detection_yunet_2023mar.onnx, MIT licence): face boxes plus five
  landmarks (both eyes, nose tip, both mouth corners).
* SFace  (face_recognition_sface_2021dec.onnx, Apache-2.0): 128-D face
  embeddings compared with cosine similarity.

Run `python scripts/download_models.py` to fetch the weights.
"""

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the face models are not installed or failed to load."""


@dataclass(frozen=True)
class Face:
    """A detected face in image coordinates."""

    box: Tuple[int, int, int, int]  # x, y, w, h
    # Landmarks as YuNet reports them, in image coordinates.
    right_eye: Tuple[float, float]  # the subject's right eye (image left)
    left_eye: Tuple[float, float]
    nose: Tuple[float, float]
    mouth_right: Tuple[float, float]
    mouth_left: Tuple[float, float]
    score: float
    raw: np.ndarray  # original 15-value YuNet row (needed by SFace alignment)

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    @property
    def inter_ocular_distance(self) -> float:
        return float(np.hypot(self.left_eye[0] - self.right_eye[0], self.left_eye[1] - self.right_eye[1]))

    def crop(self, img_bgr: np.ndarray, margin: float = 0.0) -> np.ndarray:
        x, y, w, h = self.box
        mx, my = int(w * margin), int(h * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(img_bgr.shape[1], x + w + mx), min(img_bgr.shape[0], y + h + my)
        return img_bgr[y1:y2, x1:x2]


class FaceEngine:
    def __init__(self) -> None:
        self._detector = None
        self._recognizer = None
        self._load_error: Optional[str] = None
        # OpenCV DNN objects are not safe to call from several threads at once.
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- loading
    def load(self) -> bool:
        """Loads both models. Returns False (and records why) if unavailable."""
        with self._lock:
            if self._detector is not None and self._recognizer is not None:
                return True
            det_path = settings.face_detector_path
            rec_path = settings.face_recognizer_path
            missing = [str(p) for p in (det_path, rec_path) if not p.is_file()]
            if missing:
                self._load_error = (
                    "Face model files not found: " + ", ".join(missing)
                    + ". Run `python scripts/download_models.py`."
                )
                logger.error(self._load_error)
                return False
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(det_path), "", (320, 320),
                    settings.FACE_DETECTION_MIN_CONFIDENCE, 0.3, 5000,
                )
                self._recognizer = cv2.FaceRecognizerSF.create(str(rec_path), "")
            except (cv2.error, AttributeError) as exc:
                self._detector = self._recognizer = None
                self._load_error = f"Failed to load face models: {exc}"
                logger.error(self._load_error)
                return False
            self._load_error = None
            return True

    @property
    def is_ready(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _require(self) -> None:
        if not self.is_ready and not self.load():
            raise ModelUnavailableError(self._load_error or "Face models unavailable")

    # --------------------------------------------------------------- detection
    def detect_faces(self, img_bgr: np.ndarray) -> List[Face]:
        """Returns all faces found, largest first. Empty list if none."""
        self._require()
        if img_bgr is None or img_bgr.size == 0:
            return []

        h, w = img_bgr.shape[:2]
        scale = min(1.0, settings.FACE_DETECTION_MAX_SIDE / float(max(h, w)))
        work = img_bgr if scale == 1.0 else cv2.resize(
            img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
        wh, ww = work.shape[:2]

        with self._lock:
            self._detector.setInputSize((ww, wh))
            _, rows = self._detector.detect(work)

        if rows is None:
            return []

        faces = []
        for row in rows:
            row = row.astype(np.float32).copy()
            row[:14] /= scale  # back to original image coordinates
            x, y, bw, bh = row[:4]
            x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
            x2, y2 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
            if x2 <= x1 or y2 <= y1:
                continue
            pts = row[4:14].reshape(5, 2)
            faces.append(
                Face(
                    box=(x1, y1, x2 - x1, y2 - y1),
                    right_eye=tuple(map(float, pts[0])),
                    left_eye=tuple(map(float, pts[1])),
                    nose=tuple(map(float, pts[2])),
                    mouth_right=tuple(map(float, pts[3])),
                    mouth_left=tuple(map(float, pts[4])),
                    score=float(row[14]),
                    raw=row,
                )
            )
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def primary_face(self, img_bgr: np.ndarray) -> Optional[Face]:
        faces = self.detect_faces(img_bgr)
        return faces[0] if faces else None

    # ------------------------------------------------------------- recognition
    def embedding(self, img_bgr: np.ndarray, face: Face) -> np.ndarray:
        """L2-normalised 128-D SFace embedding of an aligned face crop."""
        self._require()
        with self._lock:
            aligned = self._recognizer.alignCrop(img_bgr, face.raw)
            feature = self._recognizer.feature(aligned)
        vec = feature.flatten().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-9 else vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


face_engine = FaceEngine()
