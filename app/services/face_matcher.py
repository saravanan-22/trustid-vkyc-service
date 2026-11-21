from typing import List, Optional

import numpy as np

from app.core.config import settings
from app.schemas.verification import FaceMatchMetrics
from app.services.face_engine import Face, face_engine


class FaceMatcher:
    """
    1:1 face verification between the ID document portrait and a live frame,
    using SFace embeddings and cosine similarity.
    """

    def __init__(self, engine=face_engine):
        self.engine = engine

    def compare_faces(
        self, id_photo_bgr: Optional[np.ndarray], live_photo_bgr: Optional[np.ndarray]
    ) -> FaceMatchMetrics:
        id_faces = self.engine.detect_faces(id_photo_bgr) if id_photo_bgr is not None else []
        live_faces = self.engine.detect_faces(live_photo_bgr) if live_photo_bgr is not None else []
        return self.compare_detected(id_photo_bgr, id_faces, live_photo_bgr, live_faces)

    def compare_detected(
        self,
        id_photo_bgr: Optional[np.ndarray],
        id_faces: List[Face],
        live_photo_bgr: Optional[np.ndarray],
        live_faces: List[Face],
    ) -> FaceMatchMetrics:
        """Same as compare_faces, reusing detections that already ran."""
        if not id_faces or not live_faces:
            return FaceMatchMetrics(
                match_score=0.0,
                is_match=False,
                face_detected_in_id=bool(id_faces),
                face_detected_in_live=bool(live_faces),
                faces_in_live=len(live_faces),
                threshold=settings.FACE_MATCH_THRESHOLD,
            )

        emb_id = self.engine.embedding(id_photo_bgr, id_faces[0])
        emb_live = self.engine.embedding(live_photo_bgr, live_faces[0])
        similarity = self.engine.cosine_similarity(emb_id, emb_live)

        return FaceMatchMetrics(
            # Cosine similarity can be negative; report 0 for "not similar at all".
            match_score=round(max(0.0, similarity), 4),
            is_match=similarity >= settings.FACE_MATCH_THRESHOLD,
            face_detected_in_id=True,
            face_detected_in_live=True,
            faces_in_live=len(live_faces),
            threshold=settings.FACE_MATCH_THRESHOLD,
        )


face_matcher = FaceMatcher()
