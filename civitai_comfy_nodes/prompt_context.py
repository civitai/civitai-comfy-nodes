"""Per-prompt Civitai credentials.

A hosted runner (spine-controller) puts the session owner's token + session id
in the ComfyUI /prompt `extra_data` under "civitai" rather than a container env
var, because the comfy container is pooled across users. An on-prompt handler
captures and strips it per prompt; nodes read it during their own execution,
matched to the running prompt id (so a pooled container never crosses users).
"""

import threading
import uuid

try:
    from server import PromptServer

    _IN_COMFY = True
except Exception:  # imported under plain pytest, no ComfyUI runtime
    _IN_COMFY = False

_MAX = 64
_lock = threading.Lock()
_by_prompt: dict[str, dict] = {}
_registered = False


def _on_prompt(json_data):
    extra = json_data.get("extra_data")
    civitai = extra.pop("civitai", None) if isinstance(extra, dict) else None
    if civitai:
        prompt_id = str(json_data.get("prompt_id") or uuid.uuid4().hex)
        json_data["prompt_id"] = prompt_id  # the server honors a supplied id
        with _lock:
            _by_prompt[prompt_id] = civitai
            while len(_by_prompt) > _MAX:
                _by_prompt.pop(next(iter(_by_prompt)))
    return json_data


def register() -> None:
    global _registered
    if not _IN_COMFY or _registered:
        return
    try:
        PromptServer.instance.add_on_prompt_handler(_on_prompt)
        _registered = True
    except Exception:
        pass


def _running_prompt_id() -> str | None:
    if not _IN_COMFY:
        return None
    try:
        for item in PromptServer.instance.prompt_queue.currently_running.values():
            return item[1]  # (number, prompt_id, prompt, ...)
    except Exception:
        return None
    return None


def current() -> dict | None:
    """The session owner's {api_token, session_id} for the executing prompt, or
    None. Matched to the running prompt id — no fallback, so a credential is
    never served to the wrong prompt."""
    prompt_id = _running_prompt_id()
    if prompt_id is None:
        return None
    with _lock:
        return _by_prompt.get(prompt_id)
