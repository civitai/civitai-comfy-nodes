import os
from dataclasses import dataclass

from . import oauth
from .errors import CivitaiAuthError, CivitaiNodeError

DEFAULT_BASE_URL = "https://orchestration.civitai.com"

_NO_CREDS_MESSAGE = (
    "No Civitai credentials. Set the CIVITAI_API_TOKEN environment variable to a token from "
    "https://civitai.com/user/account, or add a Civitai Auth node and paste your token."
)


@dataclass
class ClientConfig:
    base_url: str
    token: str
    allow_mature_content: bool = False
    timeout_minutes: float = 30.0


def base_url() -> str:
    return (os.environ.get("CIVITAI_ORCHESTRATION_URL") or DEFAULT_BASE_URL).rstrip("/")


def auth_state() -> tuple[str | None, str | None]:
    """Return (token, source) from non-interactive credential sources, or (None, None).

    source is one of "env", "apikey", "oauth". Never opens a browser / interactive login, so it's
    safe for the server-side status route.
    """
    env = os.environ.get("CIVITAI_API_TOKEN")
    if env:
        return env, "env"
    key = oauth.stored_api_key()
    if key:
        return key, "apikey"
    token = oauth.get_valid_access_token()  # refreshes a stored OAuth login if present, no browser
    if token:
        return token, "oauth"
    return None, None


def resolve_config(api_config: dict | None = None, *, interactive: bool = True) -> ClientConfig:
    """Resolve auth + endpoint: CivitaiAuth node input > env var > stored API key > stored OAuth >
    (when `interactive`) browser login. With `interactive=False` (server routes), raise
    CivitaiAuthError instead of opening a browser."""
    resolved_base = (
        (api_config or {}).get("base_url") or os.environ.get("CIVITAI_ORCHESTRATION_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    allow_mature = bool((api_config or {}).get("allow_mature_content", False))
    timeout_minutes = float((api_config or {}).get("timeout_minutes") or os.environ.get("CIVITAI_COMFY_TIMEOUT", 30))

    mode = (api_config or {}).get("mode", "auto")
    token = (api_config or {}).get("api_token")
    if not token:
        token, _source = auth_state()
    # Automatic OAuth: sign in via the browser unless disabled or the user pinned api_key mode.
    if not token and interactive and mode != "api_key":
        token = oauth.interactive_login()
    if not token:
        raise (CivitaiAuthError if not interactive else CivitaiNodeError)(_NO_CREDS_MESSAGE)

    return ClientConfig(
        base_url=resolved_base,
        token=token,
        allow_mature_content=allow_mature,
        timeout_minutes=timeout_minutes,
    )
