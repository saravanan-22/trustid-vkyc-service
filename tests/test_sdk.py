import pytest
from client_sdk.trustid import TrustIDClient
from app.main import app
from fastapi.testclient import TestClient

# Verify client SDK initialization and methods
def test_sdk_initialization():
    client = TrustIDClient(base_url="http://testserver", api_key="test_key_123")
    assert client.base_url == "http://testserver"
    assert client.api_key == "test_key_123"
    assert client.headers["X-API-KEY"] == "test_key_123"
