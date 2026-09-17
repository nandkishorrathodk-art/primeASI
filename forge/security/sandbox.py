"""Defensive network helpers, lab-scope only.

Scope contract enforced in code:
  * targets must be private / loopback / link-local / RFC5737 documentation
    ranges, or explicitly listed in an allowlist,
  * only low-rate HTTP HEAD/GET reachability checks,
  * no port sweeping, no exploitation, no payload delivery.

If a user wants to test something public, they must put it in the allowlist
themselves, which turns it into an explicit, deliberate act.
"""
from __future__ import annotations

import ipaddress
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlparse

MAX_REQUESTS_PER_MINUTE = 30


class ScopeError(Exception):
    pass


def is_private_target(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast
        )
    except ValueError:
        if host in ("localhost",):
            return True
        # Resolve once, then judge the address.
        try:
            info = socket.getaddrinfo(host, None)
            return all(is_private_target(s[4][0]) for s in info)
        except socket.gaierror:
            return False


def check_scope(url: str, allowlist: Iterable[str] = ()) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host:
        raise ScopeError(f"no host in {url!r}")
    if any(host == a or host.endswith("." + a) for a in allowlist):
        return host
    if not is_private_target(host):
        raise ScopeError(
            f"{host} is not in a private/lab range. Add it to the allowlist "
            "explicitly if you are authorised to test it."
        )
    return host


@dataclass
class Probe:
    url: str
    reachable: bool
    status: Optional[int]
    latency_ms: float
    note: str = ""


class RateLimiter:
    def __init__(self, per_minute: int = MAX_REQUESTS_PER_MINUTE) -> None:
        self.per_minute = per_minute
        self._times: list[float] = []

    def allow(self) -> bool:
        now = time.time()
        self._times = [t for t in self._times if now - t < 60]
        if len(self._times) >= self.per_minute:
            return False
        self._times.append(now)
        return True


def probe(
    url: str,
    allowlist: Iterable[str] = (),
    timeout: float = 5.0,
    limiter: Optional[RateLimiter] = None,
    method: str = "HEAD",
) -> Probe:
    """Single low-rate reachability check against an in-scope target."""
    check_scope(url, allowlist)
    limiter = limiter or RateLimiter()
    if not limiter.allow():
        raise ScopeError("rate limit exceeded (30 requests/minute)")
    if method not in ("HEAD", "GET"):
        raise ScopeError("only HEAD and GET are permitted")

    req = urllib.request.Request(url, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Probe(url, True, resp.status,
                         (time.time() - t0) * 1000,
                         f"headers: {dict(resp.headers)}")
    except urllib.error.HTTPError as exc:
        return Probe(url, True, exc.code, (time.time() - t0) * 1000, "http error")
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        return Probe(url, False, None, (time.time() - t0) * 1000, str(exc))