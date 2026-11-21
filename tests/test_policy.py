from app.schemas.document import (
    DocumentQualityMetrics,
    DocumentType,
    DocumentVerificationResult,
    ExtractedDocumentData,
)
from app.schemas.session import ChallengeStep, ChallengeType
from app.schemas.verification import (
    DeepfakeMetrics,
    FaceMatchMetrics,
    LivenessMetrics,
    VerificationVerdict,
)
from app.services.verification_pipeline import decide


def face(**kw):
    base = dict(match_score=0.8, is_match=True, face_detected_in_id=True,
                face_detected_in_live=True, faces_in_live=1, threshold=0.363)
    base.update(kw)
    return FaceMatchMetrics(**base)


def liveness(active=True, passive=True):
    step = ChallengeStep(step_id=1, challenge_type=ChallengeType.SMILE, instruction="smile",
                         is_completed=active, failure_reason=None if active else "NO_SMILE_DETECTED")
    return LivenessMetrics(passive_score=0.95 if passive else 0.3, is_live_passive=passive,
                           active_challenges_passed=active, completed_challenges=[step],
                           detected_spoof_type="NONE" if passive else "BLURRY_OR_DEFOCUSED",
                           faces_detected=1)


def deepfake(suspected=False, virtual=False):
    return DeepfakeMetrics(synthetic_risk_score=0.5 if suspected else 0.0, is_deepfake_suspected=suspected,
                           virtual_camera_detected=virtual, spectral_artifact_score=0.4)


def document(valid=True):
    return DocumentVerificationResult(
        is_valid=valid, document_type=DocumentType.AADHAAR,
        extracted_data=ExtractedDocumentData(document_type=DocumentType.AADHAAR),
        quality=DocumentQualityMetrics(sharpness_score=200, is_blurry=False, glare_detected=False,
                                       corner_edges_detected=True, photo_box_detected=True),
        tamper_flags=[] if valid else ["DOCUMENT_NUMBER_CHECKSUM_INVALID"],
        confidence=0.95 if valid else 0.75,
    )


def test_all_checks_pass_is_approved():
    verdict, rej, rev = decide(face(), liveness(), deepfake(), document())
    assert verdict == VerificationVerdict.APPROVED
    assert rej == [] and rev == []


def test_failed_active_liveness_is_rejected_even_if_everything_else_passes():
    # Regression: previously this combination was APPROVED.
    verdict, rej, _ = decide(face(), liveness(active=False), deepfake(), document())
    assert verdict == VerificationVerdict.REJECTED
    assert any(r.startswith("ACTIVE_LIVENESS_FAILED") and "SMILE(NO_SMILE_DETECTED)" in r for r in rej)


def test_failed_passive_liveness_is_rejected():
    verdict, rej, _ = decide(face(), liveness(passive=False), deepfake(), document())
    assert verdict == VerificationVerdict.REJECTED
    assert any(r.startswith("PASSIVE_LIVENESS_FAILED") for r in rej)


def test_face_mismatch_is_rejected():
    verdict, rej, _ = decide(face(match_score=0.1, is_match=False), liveness(), deepfake(), document())
    assert verdict == VerificationVerdict.REJECTED
    assert any(r.startswith("BIOMETRIC_FACE_MISMATCH") for r in rej)


def test_no_live_face_is_rejected_without_duplicate_mismatch_reason():
    verdict, rej, _ = decide(
        face(match_score=0, is_match=False, face_detected_in_live=False, faces_in_live=0),
        liveness(), deepfake(), document(),
    )
    assert verdict == VerificationVerdict.REJECTED
    assert "NO_FACE_IN_LIVE_FRAMES" in rej
    assert not any(r.startswith("BIOMETRIC_FACE_MISMATCH") for r in rej)


def test_multiple_faces_is_rejected():
    verdict, rej, _ = decide(face(), liveness(), deepfake(), document(), max_faces_in_frames=2)
    assert verdict == VerificationVerdict.REJECTED
    assert "MULTIPLE_FACES_IN_LIVE_FRAMES" in rej


def test_identity_change_between_frames_is_rejected():
    verdict, rej, _ = decide(face(), liveness(), deepfake(), document(), identity_consistent=False)
    assert verdict == VerificationVerdict.REJECTED
    assert "LIVE_FRAMES_IDENTITY_INCONSISTENT" in rej


def test_document_problem_alone_goes_to_review():
    verdict, rej, rev = decide(face(), liveness(), deepfake(), document(valid=False))
    assert verdict == VerificationVerdict.REVIEW_REQUIRED
    assert rej == []
    assert rev[0].startswith("DOCUMENT_CHECKS_FAILED")


def test_heuristic_media_signals_go_to_review_not_approval():
    verdict, _, rev = decide(face(), liveness(), deepfake(suspected=True, virtual=True), document())
    assert verdict == VerificationVerdict.REVIEW_REQUIRED
    assert "SYNTHETIC_MEDIA_HEURISTIC_FLAGGED" in rev
    assert "VIRTUAL_CAMERA_HEURISTIC_FLAGGED" in rev


def test_hard_failure_wins_over_review_signals():
    verdict, rej, rev = decide(face(is_match=False, match_score=0.1), liveness(), deepfake(True), document(False))
    assert verdict == VerificationVerdict.REJECTED
    assert rej and rev
