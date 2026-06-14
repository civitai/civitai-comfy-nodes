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

from . import comfy_compat
from .errors import CivitaiNodeError

OAUTH_BASE = os.environ.get("CIVITAI_OAUTH_BASE", "https://civitai.com")
# UserRead | AIServicesRead | AIServicesWrite | BuzzRead = 1 + 16384 + 32768 + 65536
SCOPE = 114689
# Registered for the official "Civitai ComfyUI Nodes" OAuth app; override for your own app.
CLIENT_ID = os.environ.get("CIVITAI_OAUTH_CLIENT_ID", "2d61872c-9aa9-4dbc-93c3-899c222842c1")
# Loopback callback ports the client tries, in order. EVERY one must be a registered redirect URI
# on the OAuth app. They're spread across the range so a Windows reserved block (excluded port
# range, the cause of WinError 10013) is very unlikely to cover them all. Override with
# CIVITAI_OAUTH_REDIRECT_PORTS (comma-separated) if you register a different set.
DEFAULT_PORTS = [18188, 7853, 12793, 23117, 31247, 41983]
LOGIN_TIMEOUT_SECONDS = 300


def _candidate_ports() -> list[int]:
    override = os.environ.get("CIVITAI_OAUTH_REDIRECT_PORTS") or os.environ.get("CIVITAI_OAUTH_REDIRECT_PORT")
    if override:
        return [int(p) for p in override.replace(",", " ").split()]
    return DEFAULT_PORTS


def _bind_callback_server():
    """Bind the loopback callback server on the first free candidate port. Returns (server, port)."""
    ports = _candidate_ports()
    last_error = None
    for port in ports:
        try:
            server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
            return server, port
        except OSError as e:
            last_error = e
    raise CivitaiNodeError(
        f"Could not bind any of the Civitai OAuth callback ports {ports} ({last_error}). On Windows these "
        "may all be in a reserved range (`netsh interface ipv4 show excludedportrange protocol=tcp`). "
        "Set CIVITAI_API_TOKEN or use a Civitai Auth node with mode=api_key instead."
    )


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


def _result_page(success: bool, heading: str, subtext: str) -> bytes:
    accent = "#4dabf7" if success else "#ff6b6b"
    glyph = "&#10003;" if success else "&#10005;"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Civitai &middot; ComfyUI</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{
    margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
    background: #16181c; color: #e9ecef;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  .card {{
    background: #1f2227; border: 1px solid #2c2f36; border-radius: 16px;
    padding: 40px 48px; text-align: center; max-width: 420px;
    box-shadow: 0 12px 40px rgba(0,0,0,.45);
  }}
  .badge {{
    width: 64px; height: 64px; border-radius: 50%; margin: 0 auto 24px;
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; color: #fff; background: {accent};
  }}
  h1 {{ font-size: 20px; margin: 0 0 8px; font-weight: 600; }}
  p {{ margin: 0; color: #adb5bd; font-size: 14px; line-height: 1.5; }}
</style>
</head>
<body>
  <div class="card">
    <div class="badge">{glyph}</div>
    <h1>{heading}</h1>
    <p>{subtext}</p>
  </div>
</body>
</html>""".encode()


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
        if "error" in params:
            body = _result_page(
                False,
                "Login failed",
                params.get("error_description", params["error"])[0],
            )
        elif "code" in params:
            body = _result_page(
                True,
                "You're signed in to Civitai",
                "Authorization complete. You can close this tab and return to ComfyUI.",
            )
        else:
            body = _result_page(False, "Login failed", "No authorization code was returned.")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

    _CallbackHandler.result = {}
    server, port = _bind_callback_server()
    redirect_uri = f"http://localhost:{port}/civitai/callback"

    authorize_url = f"{OAUTH_BASE}/api/auth/oauth/authorize?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": SCOPE,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print(f"[civitai-comfy-nodes] Opening browser for Civitai login: {authorize_url}")
        webbrowser.open(authorize_url)
        deadline = time.time() + LOGIN_TIMEOUT_SECONDS
        while time.time() < deadline and not _CallbackHandler.result:
            # Honor ComfyUI's Cancel button while we wait on the browser, otherwise the node
            # is wedged for the full timeout and the whole queue is unstoppable.
            comfy_compat.check_interrupted()
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
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    if response.status_code != 200:
        raise CivitaiNodeError(f"Civitai token exchange failed ({response.status_code}): {response.text}")
    tokens = _store_token_response(response.json())
    print("[civitai-comfy-nodes] Civitai login successful; tokens stored.")
    return tokens["access_token"]
