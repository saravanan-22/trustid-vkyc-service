import pytest

from app.core.config import Settings

GOOD = dict(
    TRUSTID_API_KEYS="k" * 30,
    AUDIT_SIGNING_SECRET="a" * 40,
    WEBHOOK_SIGNING_SECRET="b" * 40,
)


def make(monkeypatch, **overrides):
    for name in ("TRUSTID_API_KEYS", "AUDIT_SIGNING_SECRET", "WEBHOOK_SIGNING_SECRET"):
        monkeypatch.delenv(name, raising=False)
    values = {**GOOD, **overrides}
    for name, value in values.items():
        if value is not None:
            monkeypatch.setenv(name, value)
    return Settings(_env_file=None)


def test_valid_configuration_loads(monkeypatch):
    s = make(monkeypatch, TRUSTID_API_KEYS=f"{'k' * 30}, {'j' * 30}")
    assert s.TRUSTID_API_KEYS == ["k" * 30, "j" * 30]


@pytest.mark.parametrize("missing", ["TRUSTID_API_KEYS", "AUDIT_SIGNING_SECRET", "WEBHOOK_SIGNING_SECRET"])
def test_missing_secrets_refuse_to_start(monkeypatch, missing):
    with pytest.raises(ValueError, match=missing):
        make(monkeypatch, **{missing: None})


def test_old_public_demo_values_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="demo key"):
        make(monkeypatch, TRUSTID_API_KEYS="trustid_live_demo_key_2026")
    with pytest.raises(ValueError, match="demo secret"):
        make(monkeypatch, AUDIT_SIGNING_SECRET="trustid-super-secret-hmac-audit-signing-key-v1")


def test_short_secrets_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match="at least"):
        make(monkeypatch, AUDIT_SIGNING_SECRET="short")


def test_audit_and_webhook_secrets_must_differ(monkeypatch):
    with pytest.raises(ValueError, match="must differ"):
        make(monkeypatch, WEBHOOK_SIGNING_SECRET="a" * 40)


def test_cors_defaults_to_same_origin_only(monkeypatch):
    assert make(monkeypatch).BACKEND_CORS_ORIGINS == []
