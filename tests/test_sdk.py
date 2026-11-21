import pytest

from client_sdk.trustid import TrustIDClient


def test_sdk_initialization():
    client = TrustIDClient(base_url="http://testserver/", api_key="test_key_123")
    assert client.base_url == "http://testserver"
    assert client.headers["X-API-KEY"] == "test_key_123"


def test_sdk_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("TRUSTID_API_KEY", "env_key_456")
    assert TrustIDClient().api_key == "env_key_456"


def test_sdk_has_no_default_key(monkeypatch):
    monkeypatch.delenv("TRUSTID_API_KEY", raising=False)
    with pytest.raises(ValueError):
        TrustIDClient()
