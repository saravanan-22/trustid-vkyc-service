"""
Tamper-evident audit records.

* Each dossier's `tamper_proof_audit_hash` is SHA-256 over the canonical JSON
  of *every* other field (verdict, all scores, document result, reasons,
  timestamps, and the previous dossier's hash).
* `previous_hash` links dossiers into a hash chain, so deleting or reordering
  records is detectable with `verify_chain()`.
* `signature` is HMAC-SHA256 of the hash with AUDIT_SIGNING_SECRET, so the
  hash cannot simply be recomputed by someone who edits a file.
* On every read the hash is recomputed from the content and the signature is
  checked.

Storage is JSON files on local disk with an exclusive file lock, which suits
a single host. A multi-host deployment needs a shared append-only store
(e.g. a database table with the same fields, or WORM object storage).
"""

import contextlib
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from app.core.config import settings
from app.core.security import sign_audit_hash, verify_audit_signature
from app.schemas.document import DocumentVerificationResult
from app.schemas.verification import (
    DeepfakeMetrics,
    FaceMatchMetrics,
    LivenessMetrics,
    VerificationDossier,
    VerificationVerdict,
)

try:  # POSIX file locking; falls back to an in-process lock elsewhere.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

GENESIS_HASH = "0" * 64
VERIFICATION_ID_RE = re.compile(r"^verif_[0-9a-f]{24}$")
_HASH_EXCLUDED_FIELDS = {"tamper_proof_audit_hash", "signature"}


class AuditIntegrityError(Exception):
    """Stored dossier failed hash or signature verification."""


def canonical_hash(dossier_fields: dict) -> str:
    payload = json.dumps(dossier_fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_dossier_hash(dossier: VerificationDossier) -> str:
    return canonical_hash(dossier.model_dump(mode="json", exclude=_HASH_EXCLUDED_FIELDS))


class AuditService:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir or settings.AUDIT_STORAGE_PATH)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.Lock()

    # ------------------------------------------------------------- internals
    @property
    def _head_path(self) -> Path:
        return self.storage_dir / "_chain_head.json"

    def _path_for(self, verification_id: str) -> Path:
        return self.storage_dir / f"{verification_id}.json"

    @contextlib.contextmanager
    def _chain_lock(self) -> Iterator[None]:
        with self._thread_lock:
            if fcntl is None:  # pragma: no cover
                yield
                return
            with open(self.storage_dir / "_chain.lock", "a+") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _read_head(self) -> dict:
        if not self._head_path.exists():
            return {"hash": GENESIS_HASH, "verification_id": None, "count": 0}
        return json.loads(self._head_path.read_text(encoding="utf-8"))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # ---------------------------------------------------------------- public
    def create_and_store_dossier(
        self,
        *,
        verification_id: str,
        session_id: str,
        user_id: str,
        owner_key_id: str,
        verdict: VerificationVerdict,
        policy_version: str,
        overall_confidence: float,
        frames_analyzed: int,
        face_match: FaceMatchMetrics,
        liveness: LivenessMetrics,
        deepfake: DeepfakeMetrics,
        document: DocumentVerificationResult,
        rejection_reasons: List[str],
        review_reasons: List[str],
    ) -> VerificationDossier:
        if not VERIFICATION_ID_RE.match(verification_id):
            raise ValueError("invalid verification_id")

        with self._chain_lock():
            head = self._read_head()
            draft = VerificationDossier(
                verification_id=verification_id,
                session_id=session_id,
                user_id=user_id,
                verdict=verdict,
                policy_version=policy_version,
                overall_confidence=overall_confidence,
                timestamp=datetime.now(timezone.utc),
                frames_analyzed=frames_analyzed,
                face_match=face_match,
                liveness=liveness,
                deepfake=deepfake,
                document=document,
                rejection_reasons=rejection_reasons,
                review_reasons=review_reasons,
                previous_hash=head["hash"],
                tamper_proof_audit_hash="",
                signature="",
            )
            # Normalise through JSON first so the hash is computed on exactly
            # what will be read back from disk.
            dossier = VerificationDossier.model_validate_json(draft.model_dump_json())
            audit_hash = compute_dossier_hash(dossier)
            dossier = dossier.model_copy(
                update={"tamper_proof_audit_hash": audit_hash, "signature": sign_audit_hash(audit_hash)}
            )

            record = {
                "owner_key_id": owner_key_id,
                # Navigation pointer only; the link itself is proven by previous_hash.
                "previous_verification_id": head["verification_id"],
                "dossier": dossier.model_dump(mode="json"),
            }
            self._atomic_write(self._path_for(verification_id), json.dumps(record, indent=2))
            self._atomic_write(
                self._head_path,
                json.dumps(
                    {"hash": audit_hash, "verification_id": verification_id, "count": head["count"] + 1}
                ),
            )
        return dossier

    def _load_record(self, verification_id: str) -> Optional[dict]:
        if not VERIFICATION_ID_RE.match(verification_id):
            return None
        path = self._path_for(verification_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _verify(dossier: VerificationDossier) -> None:
        recomputed = compute_dossier_hash(dossier)
        if recomputed != dossier.tamper_proof_audit_hash:
            raise AuditIntegrityError("dossier content does not match its audit hash")
        if not verify_audit_signature(dossier.tamper_proof_audit_hash, dossier.signature):
            raise AuditIntegrityError("audit hash signature is invalid")

    def get_dossier(self, verification_id: str, owner_key_id: str) -> Optional[VerificationDossier]:
        """
        Returns the verified dossier, or None if it does not exist or belongs to
        another API key. Raises AuditIntegrityError if it was modified.
        """
        record = self._load_record(verification_id)
        if record is None or record.get("owner_key_id") != owner_key_id:
            return None
        try:
            dossier = VerificationDossier.model_validate(record["dossier"])
        except (KeyError, ValueError) as exc:
            raise AuditIntegrityError(f"dossier is malformed: {exc}") from exc
        self._verify(dossier)
        return dossier

    def verify_chain(self) -> int:
        """
        Walks the chain backwards from the head, verifying every dossier and link.
        Returns the number of verified records. Raises AuditIntegrityError on any break.
        """
        with self._chain_lock():
            head = self._read_head()
        expected_hash, current_id, count = head["hash"], head["verification_id"], 0
        while current_id is not None:
            record = self._load_record(current_id)
            if record is None:
                raise AuditIntegrityError(f"dossier {current_id} is missing from the chain")
            dossier = VerificationDossier.model_validate(record["dossier"])
            self._verify(dossier)
            if dossier.tamper_proof_audit_hash != expected_hash:
                raise AuditIntegrityError(f"chain link broken at {current_id}")
            count += 1
            expected_hash = dossier.previous_hash
            current_id = record.get("previous_verification_id")
            if (current_id is None) != (expected_hash == GENESIS_HASH):
                raise AuditIntegrityError(f"chain pointer inconsistent after {dossier.verification_id}")
        if count != head["count"]:
            raise AuditIntegrityError(f"chain has {count} records but head expects {head['count']}")
        return count


audit_service = AuditService()
