import itertools

import cv2
import numpy as np
import pytest

from app.services.face_matcher import face_matcher
from tests.conftest import load_face, requires_models

pytestmark = requires_models

SAME_PERSON = [("ride_1984.jpg", "ride_1979.jpg"), ("armstrong_1964.jpg", "armstrong_pose.jpg")]
PEOPLE = {
    "ride": ["ride_1984.jpg", "ride_1979.jpg"],
    "armstrong": ["armstrong_1964.jpg", "armstrong_pose.jpg"],
    "collins": ["collins.jpg"],
}


@pytest.mark.parametrize("a,b", SAME_PERSON)
def test_two_different_photos_of_same_person_match(a, b):
    metrics = face_matcher.compare_faces(load_face(a), load_face(b))
    assert metrics.face_detected_in_id and metrics.face_detected_in_live
    assert metrics.is_match, metrics


def _cross_person_pairs():
    for (_, imgs1), (_, imgs2) in itertools.combinations(PEOPLE.items(), 2):
        for a in imgs1:
            for b in imgs2:
                yield a, b


@pytest.mark.parametrize("a,b", list(_cross_person_pairs()))
def test_different_people_do_not_match(a, b):
    metrics = face_matcher.compare_faces(load_face(a), load_face(b))
    assert not metrics.is_match, metrics
    assert metrics.match_score < metrics.threshold


def test_random_noise_has_no_face_and_does_not_match():
    # Regression: the old histogram matcher scored two noise images at 0.99.
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    metrics = face_matcher.compare_faces(a, b)
    assert not metrics.face_detected_in_id and not metrics.face_detected_in_live
    assert not metrics.is_match and metrics.match_score == 0.0


def test_cartoon_shapes_are_not_faces():
    img = np.full((480, 640, 3), 200, np.uint8)
    cv2.circle(img, (320, 220), 80, (130, 150, 185), -1)
    cv2.ellipse(img, (320, 420), (110, 80), 0, 180, 360, (130, 150, 185), -1)
    metrics = face_matcher.compare_faces(img, load_face("ride_1984.jpg"))
    assert not metrics.face_detected_in_id
    assert not metrics.is_match


def test_face_vs_noise_does_not_match():
    noise = np.random.default_rng(1).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    metrics = face_matcher.compare_faces(load_face("ride_1984.jpg"), noise)
    assert metrics.face_detected_in_id and not metrics.face_detected_in_live
    assert not metrics.is_match
