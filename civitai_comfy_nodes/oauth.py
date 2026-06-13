"""Civitai OAuth 2.0 (Authorization Code + PKCE) for interactive logins.

Tokens persist in ~/.civitai/comfy-oauth.json; access tokens last 1h and are
auto-refreshed, refresh tokens last 30 days, after which a new interactive
login is required.
"""

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import requests

from .errors import CivitaiNodeError

OAUTH_BASE = os.environ.get("CIVITAI_OAUTH_BASE", "https://civitai.com")
# UserRead | AIServicesRead | AIServicesWrite | BuzzRead = 1 + 16384 + 32768 + 65536
SCOPE = 114689
# Registered for the official "Civitai ComfyUI Nodes" OAuth app; override for your own app.
CLIENT_ID = os.environ.get("CIVITAI_OAUTH_CLIENT_ID", "")
REDIRECT_PORT = int(os.environ.get("CIVITAI_OAUTH_REDIRECT_PORT", "18188"))
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/civitai/callback"
LOGIN_TIMEOUT_SECONDS = 300


def token_store_path() -> Path:
    override = os.environ.get("CIVITAI_COMFY_OAUTH_STORE")
    if override:
        return Path(override)
    return Path.home() / ".civitai" / "comfy-oauth.json"


def _load_tokens() -> dict | None:
    path = token_store_path()
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_tokens(tokens: dict) -> None:
    path = token_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokens, indent=2))
    path.chmod(0o600)


def _store_token_response(payload: dict) -> dict:
    tokens = {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token"),
        "expires_at": time.time() + float(payload.get("expires_in", 3600)),
        "scope": payload.get("scope"),
    }
    _save_tokens(tokens)
    return tokens


def _refresh(tokens: dict) -> dict | None:
    if not tokens.get("refresh_token") or not CLIENT_ID:
        return None
    response = requests.post(
        f"{OAUTH_BASE}/api/auth/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": tokens["refresh_token"],
            "client_id": CLIENT_ID,
        },
        timeout=30,
    )
    if response.status_code != 200:
        return None
    return _store_token_response(response.json())


def get_valid_access_token() -> str | None:
    """Return a usable stored access token, refreshing if expired; None if unavailable."""
    tokens = _load_tokens()
    if tokens is None:
        return None
    if tokens.get("expires_at", 0) > time.time() + 60:
        return tokens["access_token"]
    refreshed = _refresh(tokens)
    return refreshed["access_token"] if refreshed else None


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):  # noqa: N802 - http.server API
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/civitai/callback":
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.result = {k: v[0] for k, v in params.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Civitai login complete \xe2\x80\x94 you can close this tab.</h2></body></html>"
        )

    def log_message(self, *args):
        pass


def interactive_login() -> str:
    """Run the loopback PKCE flow: open a browser, capture the code, exchange and store tokens."""
    if not CLIENT_ID:
        raise CivitaiNodeError(
            "No Civitai OAuth client id configured. Either set the CIVITAI_API_TOKEN environment variable "
            "with an API key from https://civitai.com/user/account, or set CIVITAI_OAUTH_CLIENT_ID to a "
            "registered OAuth app id."
        )

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)

    authorize_url = f"{OAUTH_BASE}/api/auth/oauth/authorize?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )

    _CallbackHandler.result = {}
    try:
        server = http.server.HTTPServer(("127.0.0.1", REDIRECT_PORT), _CallbackHandler)
    except OSError as e:
        raise CivitaiNodeError(
            f"Could not open localhost port {REDIRECT_PORT} for the Civitai OAuth callback ({e}). "
            "Close whatever is using it, or set CIVITAI_API_TOKEN instead."
        ) from e

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print(f"[civitai-comfy-nodes] Opening browser for Civitai login: {authorize_url}")
        webbrowser.open(authorize_url)
        deadline = time.time() + LOGIN_TIMEOUT_SECONDS
        while time.time() < deadline and not _CallbackHandler.result:
            time.sleep(0.25)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    result = _CallbackHandler.result
    if not result:
        raise CivitaiNodeError(
            f"Civitai login timed out after {LOGIN_TIMEOUT_SECONDS}s. If this ComfyUI runs on a remote machine, "
            "the browser flow cannot reach it — set CIVITAI_API_TOKEN instead."
        )
    if result.get("state") != state:
        raise CivitaiNodeError("Civitai login failed: OAuth state mismatch (possible CSRF or stale flow).")
    if "error" in result:
        raise CivitaiNodeError(f"Civitai login failed: {result.get('error_description') or result['error']}")

    response = requests.post(
        f"{OAUTH_BASE}/api/auth/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": result["code"],
            "code_verifier": verifier,
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise CivitaiNodeError(f"Civitai token exchange failed ({response.status_code}): {response.text}")
    tokens = _store_token_response(response.json())
    print("[civitai-comfy-nodes] Civitai login successful; tokens stored.")
    return tokens["access_token"]
