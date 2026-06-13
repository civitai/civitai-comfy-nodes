import os
from dataclasses import dataclass

from . import oauth
from .errors import CivitaiNodeError

DEFAULT_BASE_URL = "https://orchestration.civitai.com"


@dataclass
class ClientConfig:
    base_url: str
    token: str
    allow_mature_content: bool = False
    timeout_minutes: float = 30.0


def resolve_config(api_config: dict | None = None) -> ClientConfig:
    """Resolve auth + endpoint: CivitaiAuth node input > env vars > stored OAuth tokens > interactive login."""
    base_url = (
        (api_config or {}).get("base_url") or os.environ.get("CIVITAI_ORCHESTRATION_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    allow_mature = bool((api_config or {}).get("allow_mature_content", False))
    timeout_minutes = float((api_config or {}).get("timeout_minutes") or os.environ.get("CIVITAI_COMFY_TIMEOUT", 30))

    token = (api_config or {}).get("api_token") or os.environ.get("CIVITAI_API_TOKEN")
    if not token:
        token = oauth.get_valid_access_token()
    if not token and (api_config or {}).get("mode", "auto") != "api_key":
        token = oauth.interactive_login()
    if not token:
        raise CivitaiNodeError(
            "No Civitai credentials found. Connect a Civitai Auth node, set CIVITAI_API_TOKEN, or log in via OAuth."
        )

    return ClientConfig(
        base_url=base_url,
        token=token,
        allow_mature_content=allow_mature,
        timeout_minutes=timeout_minutes,
    )
