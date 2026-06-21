import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import oauth, prompt_context
from .errors import CivitaiAuthError, CivitaiNodeError

DEFAULT_BASE_URL = "https://orchestration.civitai.com"

# The GPU generation offloaded jobs currently run on. Surfaced read-only in the settings panel;
# not yet a selectable control (the consumer API has no field for it).
GPU_GENERATION_LABEL = "Ada"

# Allowed required-VRAM tiers (GB) offered by the settings panel.
VRAM_TIERS = [24]

MATURE_CONTENT_MODES = ("auto", "true", "false")

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

_NO_CREDS_MESSAGE = (
    "No Civitai credentials. Set the CIVITAI_API_TOKEN environment variable to a token from "
    "https://civitai.com/user/account, or add a Civitai Auth node and paste your token."
)


@dataclass
class ClientConfig:
    base_url: str
    token: str
    mature_content: str = "auto"
    timeout_minutes: float = 30.0

    @property
    def allow_mature_content(self) -> bool:
        return self.mature_content == "true"


def settings_store_path() -> Path:
    override = os.environ.get("CIVITAI_COMFY_SETTINGS_STORE")
    if override:
        return Path(override)
    return Path.home() / ".civitai" / "comfy-settings.json"


def load_pack_settings() -> dict:
    """Persisted pack settings written by the sidebar Settings panel ({} when absent/corrupt)."""
    try:
        data = json.loads(settings_store_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_pack_settings(data: dict) -> None:
    path = settings_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    path.chmod(0o600)


def stored_orchestrator_url() -> str | None:
    url = (load_pack_settings().get("orchestratorUrl") or "").strip()
    return url or None


def stored_min_vram_gb() -> int | None:
    value = load_pack_settings().get("minVramGb")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def stored_mature_content() -> str:
    mode = load_pack_settings().get("allowMatureContent")
    return mode if mode in MATURE_CONTENT_MODES else "auto"


def stored_use_sage_attention() -> bool:
    return bool(load_pack_settings().get("useSageAttention", True))


def base_url() -> str:
    return (os.environ.get("CIVITAI_ORCHESTRATION_URL") or stored_orchestrator_url() or DEFAULT_BASE_URL).rstrip("/")


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
        (api_config or {}).get("base_url")
        or os.environ.get("CIVITAI_ORCHESTRATION_URL")
        or stored_orchestrator_url()
        or DEFAULT_BASE_URL
    ).rstrip("/")
    if api_config is not None and "allow_mature_content" in api_config:
        mature_content = "true" if api_config["allow_mature_content"] else "false"
    else:
        mature_content = stored_mature_content()
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
        mature_content=mature_content,
        timeout_minutes=timeout_minutes,
    )
