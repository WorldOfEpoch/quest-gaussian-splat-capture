"""Authenticated readiness checks for the existing desktop bridge launcher."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .config import load_config


def data_root_id(root: Path) -> str:
    # An opaque identity lets the launcher detect the wrong storage without
    # disclosing a Windows user path on the headset-facing listener.
    return hashlib.sha256(os.path.normcase(str(root.resolve())).encode()).hexdigest()


def probe(root: Path, url: str, *, dashboard: bool, wait: float = 10.) -> dict:
    config = load_config(root)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Expected a bridge HTTP(S) listener URL")
    loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    if parsed.scheme == "http" and not loopback:
        raise ValueError("Non-loopback readiness requires HTTPS")
    context = ssl.create_default_context(cafile=str(root / "server.crt")) if parsed.scheme == "https" else None
    if context and loopback:
        # USB is a local reverse tunnel to a certificate issued for the paired
        # LAN IP. Keep certificate verification against that exact local CA;
        # only its host-name check differs on the loopback transport.
        context.check_hostname = False
    expected = data_root_id(root)
    deadline = time.monotonic() + max(0., wait)
    # No redirects: a readiness check must never send the pairing token to
    # a different service or host.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    handlers = [NoRedirect(), urllib.request.ProxyHandler({})]
    if context:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    while True:
        request = urllib.request.Request(url.rstrip("/") + "/api/v1/runtime",
                                         headers={"Authorization": f"Bearer {config.token}"})
        try:
            with opener.open(request, timeout=2.) as response:
                state = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if time.monotonic() >= deadline:
                raise RuntimeError("Listener not ready; inspect its bridge log and port ownership") from error
            time.sleep(.25)
            continue
        if state.get("data_root_id") != expected or state.get("dashboard_enabled") is not dashboard:
            raise RuntimeError("Listener has the wrong data directory or dashboard role; no process was stopped")
        if state.get("status") != "ok" or state.get("protocol_version") != "0.1":
            raise RuntimeError("Listener does not implement the expected bridge protocol")
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--wait", type=float, default=10.)
    args = parser.parse_args()
    try:
        print(json.dumps(probe(args.data_dir, args.url, dashboard=args.dashboard, wait=args.wait)))
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
