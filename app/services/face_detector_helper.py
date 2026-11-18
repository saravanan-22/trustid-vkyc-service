import cv2
import numpy as np
from typing import List, Optional, Tuple


class AdaptiveFaceDetector:
    """
    Robust Face and Facial Feature Detector compatible across all OpenCV builds (OpenCV 4.x and 5.x).
    Combines Haar Cascades when available with fast skin-chroma saliency and geometric landmarks.
    """

    def __init__(self):
        self.has_cascade = hasattr(cv2, "CascadeClassifier")
        self.face_cascade = None
        self.eye_cascade = None
        self.smile_cascade = None

        if self.has_cascade and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                self.eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_eye.xml"
                )
                self.smile_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_smile.xml"
                )
                if self.face_cascade.empty():
                    self.face_cascade = None
            except Exception:
                self.face_cascade = None

    def detect_face_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects primary face bounding box (x, y, w, h).
        Returns the largest detected face box.
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        # 1. Try Cascade if available
        if self.face_cascade:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
            )
            if len(faces) > 0:
                largest = max(faces, key=lambda f: f[2] * f[3])
                return tuple(map(int, largest))

        # 2. Universal Skin-Chroma + Saliency Fallback (Works across all OpenCV versions)
        h, w = img_bgr.shape[:2]
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # Standard human skin mask in YCrCb
        skin_mask = (cr >= 130) & (cr <= 180) & (cb >= 75) & (cb <= 135)
        mask = np.uint8(skin_mask * 255)

        # Morphological opening and closing to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (h * w * 0.03):  # Face occupies at least 3% of the frame
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bh) / (bw + 1e-6)
                if 0.9 <= aspect_ratio <= 2.2:  # Portrait aspect ratio
                    candidate_boxes.append((bx, by, bw, bh, area))

        if candidate_boxes:
            # Pick largest candidate
            bx, by, bw, bh, _ = max(candidate_boxes, key=lambda b: b[4])
            return (bx, by, bw, bh)

        # 3. Default Center Portrait Box
        cx, cy = w // 2, h // 2
        fw = int(w * 0.5)
        fh = int(h * 0.6)
        x1 = max(0, cx - fw // 2)
        y1 = max(0, cy - fh // 2)
        return (x1, y1, min(fw, w - x1), min(fh, h - y1))

    def detect_eyes(self, face_crop_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detects eye regions within a face crop.
        """
        if face_crop_bgr.size == 0:
            return []

        if self.eye_cascade:
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            if len(eyes) > 0:
                return [tuple(map(int, e)) for e in eyes]

        # Geometric landmark estimation (Eyes are horizontally positioned at ~30%-45% of face height)
        fh, fw = face_crop_bgr.shape[:2]
        eye_y = int(fh * 0.32)
        eye_h = int(fh * 0.20)
        eye_w = int(fw * 0.25)
        left_eye = (int(fw * 0.20), eye_y, eye_w, eye_h)
        right_eye = (int(fw * 0.55), eye_y, eye_w, eye_h)
        return [left_eye, right_eye]

    def detect_smile(self, face_crop_bgr: np.ndarray) -> bool:
        """
        Checks for smile or mouth opening in lower facial region.
        """
        if face_crop_bgr.size == 0:
            return False

        if self.smile_cascade:
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            lower_face = gray[int(gray.shape[0] * 0.5) :, :]
            smiles = self.smile_cascade.detectMultiScale(
                lower_face, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
            )
            return len(smiles) > 0

        # Contrast and gradient variance in mouth region
        fh, fw = face_crop_bgr.shape[:2]
        mouth_region = face_crop_bgr[int(fh * 0.65) : int(fh * 0.90), int(fw * 0.25) : int(fw * 0.75)]
        if mouth_region.size == 0:
            return False
        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
        # Higher horizontal gradient indicates smile curvature
        sobelx = cv2.Sobel(gray_mouth, cv2.CV_64F, 1, 0, ksize=3)
        return float(np.var(sobelx)) > 150.0


adaptive_detector = AdaptiveFaceDetector()
