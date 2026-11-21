import json
import uuid

import pytest

from app.schemas.verification import VerificationVerdict
from app.services.audit_service import AuditIntegrityError, AuditService
from tests.test_policy import deepfake, document, face, liveness

OWNER = "owner-fingerprint-1"


def new_id():
    return f"verif_{uuid.uuid4().hex[:24]}"


def store(svc, verdict=VerificationVerdict.REJECTED, owner=OWNER):
    return svc.create_and_store_dossier(
        verification_id=new_id(), session_id="vkyc_test", user_id="user_1", owner_key_id=owner,
        verdict=verdict, policy_version="test", overall_confidence=0.5, frames_analyzed=3,
        face_match=face(), liveness=liveness(), deepfake=deepfake(), document=document(),
        rejection_reasons=["ACTIVE_LIVENESS_FAILED: SMILE(NO_SMILE_DETECTED)"], review_reasons=[],
    )


@pytest.fixture
def svc(tmp_path):
    return AuditService(storage_dir=tmp_path)


def _edit(svc, verification_id, mutate):
    path = svc.storage_dir / f"{verification_id}.json"
    record = json.loads(path.read_text())
    mutate(record["dossier"])
    path.write_text(json.dumps(record))


def test_round_trip_verifies(svc):
    d = store(svc)
    loaded = svc.get_dossier(d.verification_id, OWNER)
    assert loaded == d


def test_editing_verdict_is_detected(svc):
    # Regression: the old check only compared HMAC(stored hash) to the stored
    # signature, so editing any field without touching the hash went unnoticed.
    d = store(svc)
    _edit(svc, d.verification_id, lambda x: x.update(verdict="APPROVED"))
    with pytest.raises(AuditIntegrityError):
        svc.get_dossier(d.verification_id, OWNER)


def test_editing_fields_outside_the_old_hash_is_detected(svc):
    d = store(svc)
    _edit(svc, d.verification_id, lambda x: x.update(rejection_reasons=[]))
    with pytest.raises(AuditIntegrityError):
        svc.get_dossier(d.verification_id, OWNER)


def test_recomputed_hash_without_secret_is_detected(svc):
    from app.services.audit_service import compute_dossier_hash
    from app.schemas.verification import VerificationDossier

    d = store(svc)

    def forge(x):
        x["verdict"] = "APPROVED"
        x["tamper_proof_audit_hash"] = compute_dossier_hash(VerificationDossier.model_validate(x))

    _edit(svc, d.verification_id, forge)
    with pytest.raises(AuditIntegrityError, match="signature"):
        svc.get_dossier(d.verification_id, OWNER)


def test_other_owner_cannot_read(svc):
    d = store(svc)
    assert svc.get_dossier(d.verification_id, "someone-else") is None


@pytest.mark.parametrize("bad_id", ["../../etc/passwd", "verif_xyz", "verif_" + "g" * 24, ""])
def test_malformed_ids_are_not_found(svc, bad_id):
    assert svc.get_dossier(bad_id, OWNER) is None


def test_chain_links_and_verifies(svc):
    a, b, c = store(svc), store(svc), store(svc)
    assert b.previous_hash == a.tamper_proof_audit_hash
    assert c.previous_hash == b.tamper_proof_audit_hash
    assert svc.verify_chain() == 3


def test_deleting_a_record_breaks_the_chain(svc):
    store(svc)
    middle = store(svc)
    store(svc)
    (svc.storage_dir / f"{middle.verification_id}.json").unlink()
    with pytest.raises(AuditIntegrityError):
        svc.verify_chain()


def test_empty_chain_verifies(svc):
    assert svc.verify_chain() == 0
