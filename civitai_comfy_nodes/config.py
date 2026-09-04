import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import oauth, prompt_context
from .errors import CivitaiAuthError, CivitaiNodeError

DEFAULT_BASE_URL = "https://orchestration.civitai.com"
DEFAULT_LINK_URL = "https://link.civitai.com"

# Workflows submitted by this pack carry two indexed tags so the gallery can scope its listing:
# SOURCE_TAG (any workflow from this pack) and a per-session tag identifying the submitter.
SOURCE_TAG = "civitai-comfy-nodes"


def session_id_store_path() -> Path:
    override = os.environ.get("CIVITAI_COMFY_SESSION_STORE")
    if override:
        return Path(override)
    return Path.home() / ".civitai" / "comfy-session-id"


def resolve_session_id() -> str:
    """The submitting session's stable id. A host (e.g. comfy-cloud) pins it via
    CIVITAI_COMFY_SESSION_ID so submissions link to its own session; a standalone install
    instead mints one and persists it, so it survives ComfyUI restarts (identifies the instance,
    not just the process). Resolved per call so a host that sets the env var is always honoured."""
    ctx = prompt_context.current()
    if ctx and (ctx.get("session_id") or "").strip():
        return ctx["session_id"].strip()
    provided = os.environ.get("CIVITAI_COMFY_SESSION_ID")
    if provided and provided.strip():
        return provided.strip()
    path = session_id_store_path()
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except OSError:
        pass
    new_id = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id)
        path.chmod(0o600)
    except OSError:
        pass  # read-only FS: not persisted, but still usable for this run
    return new_id


def session_tag() -> str:
    return f"{SOURCE_TAG}:session:{resolve_session_id()}"


def submit_tags() -> list[str]:
    return [SOURCE_TAG, session_tag()]


def is_hosted_session() -> bool:
    """A host (comfy-cloud) pins CIVITAI_COMFY_SESSION_ID on pooled containers, where per-user
    features like Civitai Link must stay off."""
    return bool((os.environ.get("CIVITAI_COMFY_SESSION_ID") or "").strip())


def settings_store_path() -> Path:
    override = os.environ.get("CIVITAI_COMFY_SETTINGS_STORE")
    if override:
        return Path(override)
    return Path.home() / ".civitai" / "comfy-settings.json"


def load_pack_settings() -> dict:
    """Persisted pack settings written by the Settings dialog ({} when absent/corrupt)."""
    try:
        data = json.loads(settings_store_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_pack_settings(data: dict) -> None:
    path = settings_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def stored_enable_link() -> bool:
    return bool(load_pack_settings().get("enableLink", True))


def stored_link_url() -> str | None:
    url = (load_pack_settings().get("linkUrl") or "").strip()
    return url or None


def link_url() -> str:
    return (os.environ.get("CIVITAI_LINK_URL") or stored_link_url() or DEFAULT_LINK_URL).rstrip("/")


def link_key_store_path() -> Path:
    override = os.environ.get("CIVITAI_COMFY_LINK_STORE")
    if override:
        return Path(override)
    return Path.home() / ".civitai" / "comfy-link.json"


def load_link_key() -> dict | None:
    """The persisted Civitai Link pairing: ``{"key", "activated", "instance_id", "paired_at"}`` or None.
    ``instance_id`` is set only for pairings made through the account (OAuth) flow."""
    try:
        data = json.loads(link_key_store_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    key = (data.get("key") or "").strip() if isinstance(data, dict) else ""
    if not key:
        return None
    return {
        "key": key,
        "activated": bool(data.get("activated")),
        "instance_id": data.get("instance_id"),
        "paired_at": data.get("paired_at"),
    }


def save_link_key(key: str, *, activated: bool, instance_id: int | None = None) -> None:
    path = link_key_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "key": key,
        "activated": activated,
        "instance_id": instance_id,
        "paired_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(record))
    path.chmod(0o600)


def install_id_store_path() -> Path:
    override = os.environ.get("CIVITAI_COMFY_INSTALL_ID_STORE")
    if override:
        return Path(override)
    return Path.home() / ".civitai" / "comfy-install-id"


def install_id() -> str:
    """A UUID v4 identifying this install to link-service, generated once and kept across unpair and
    logout: re-pairing with the same id re-keys the same instance instead of adding one towards the
    per-user instance limit."""
    path = install_id_store_path()
    try:
        candidate = uuid.UUID(path.read_text().strip())
        if candidate.version == 4:
            return str(candidate)
    except (FileNotFoundError, OSError, ValueError):
        pass
    fresh = str(uuid.uuid4())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fresh)
    path.chmod(0o600)
    return fresh


def clear_link_key() -> None:
    try:
        link_key_store_path().unlink()
    except FileNotFoundError:
        pass


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

    source is one of "prompt", "env", "apikey", "oauth". Never opens a browser / interactive login,
    so it's safe for the server-side status route.
    """
    ctx = prompt_context.current()
    if ctx and ctx.get("api_token"):
        return ctx["api_token"], "prompt"
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
