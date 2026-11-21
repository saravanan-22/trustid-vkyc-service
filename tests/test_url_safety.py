import socket

import pytest

from app.core import url_safety
from app.core.url_safety import UnsafeURLError, validate_callback_url


def fake_resolver(ip):
    def _resolve(host, port, *args, **kwargs):
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]
    return _resolve


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.1.2.3", "172.16.0.5", "192.168.1.10", "169.254.169.254", "0.0.0.0", "::1",
     "fd00::1", "::ffff:127.0.0.1", "100.64.0.1"],
)
def test_private_and_internal_addresses_are_rejected(monkeypatch, ip):
    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_resolver(ip))
    with pytest.raises(UnsafeURLError):
        validate_callback_url("https://hooks.example.com/kyc")


def test_public_https_url_is_accepted(monkeypatch):
    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_resolver("93.184.215.14"))
    assert validate_callback_url("https://hooks.example.com/kyc") == "https://hooks.example.com/kyc"


@pytest.mark.parametrize(
    "url",
    ["http://hooks.example.com/kyc", "ftp://hooks.example.com", "file:///etc/passwd",
     "https://user:pass@hooks.example.com/", "https:///no-host", "gopher://x"],
)
def test_bad_schemes_hosts_and_credentials_are_rejected(monkeypatch, url):
    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_resolver("93.184.215.14"))
    with pytest.raises(UnsafeURLError):
        validate_callback_url(url)


def test_empty_url_is_none():
    assert validate_callback_url(None) is None
    assert validate_callback_url("") is None
