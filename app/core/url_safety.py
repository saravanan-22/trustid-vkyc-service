"""
Validation for client-supplied webhook URLs (SSRF protection).

The server makes outbound requests to these URLs, so without checks an API
client could make it call internal services (cloud metadata endpoints,
databases, admin panels on the private network).
"""

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlsplit

from app.core.config import settings


class UnsafeURLError(ValueError):
    pass


def _is_public_address(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def validate_callback_url(url: Optional[str]) -> Optional[str]:
    """
    Returns the URL if it is safe to call, otherwise raises UnsafeURLError.

    Checks: http(s) scheme (https only unless WEBHOOK_ALLOW_HTTP), a hostname,
    no embedded credentials, and every resolved address must be publicly
    routable unless WEBHOOK_ALLOW_PRIVATE_NETWORKS is set.

    Call it again right before sending: DNS can change between the request and
    the delivery. This narrows DNS-rebinding attacks but cannot fully close them;
    for strict environments, route webhooks through an egress proxy.
    """
    if url is None or url == "":
        return None
    if len(url) > 2048:
        raise UnsafeURLError("callback_url is too long")

    parts = urlsplit(url)
    allowed_schemes = {"https", "http"} if settings.WEBHOOK_ALLOW_HTTP else {"https"}
    if parts.scheme not in allowed_schemes:
        raise UnsafeURLError(f"callback_url scheme must be one of {sorted(allowed_schemes)}")
    if not parts.hostname:
        raise UnsafeURLError("callback_url must include a hostname")
    if parts.username or parts.password:
        raise UnsafeURLError("callback_url must not contain credentials")

    if settings.WEBHOOK_ALLOW_PRIVATE_NETWORKS:
        return url

    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise UnsafeURLError("callback_url has an invalid port")

    try:
        infos = socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise UnsafeURLError("callback_url hostname does not resolve")
    if not infos:
        raise UnsafeURLError("callback_url hostname does not resolve")

    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not _is_public_address(ip):
            raise UnsafeURLError("callback_url must resolve to a public internet address")
    return url
