"""Pure Civitai Link wire-format helpers (no socket, no ComfyUI): resource-type/folder maps, command
response shapes, progress math, the activity ring. The relay itself is a dumb pipe; the shapes here
mirror the web app's `src/components/CivitaiLink/shared-types.ts`."""

import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timezone

from . import local_models

# Civitai ModelType (what `resources:add` carries) -> ComfyUI model folder.
LINK_TYPE_FOLDERS = {
    "Checkpoint": "checkpoints",
    "CheckpointConfig": "checkpoints",
    "LORA": "loras",
    "LoCon": "loras",
    "DoRA": "loras",
    "TextualInversion": "embeddings",
    "VAE": "vae",
    "Controlnet": "controlnet",
    "Upscaler": "upscale_models",
    "Hypernetwork": "hypernetworks",
    "UNet": "diffusion_models",
    "TextEncoder": "text_encoders",
    "CLIPVision": "clip_vision",
    "MotionModule": "animatediff_models",
}

FOLDER_RESOURCE_TYPES = {
    "checkpoints": "Checkpoint",
    "loras": "LORA",
    "embeddings": "TextualInversion",
    "vae": "VAE",
    "controlnet": "Controlnet",
    "upscale_models": "Upscaler",
    "hypernetworks": "Hypernetwork",
    "diffusion_models": "UNet",
    "text_encoders": "TextEncoder",
    "clip_vision": "CLIPVision",
    "animatediff_models": "MotionModule",
}

LIST_FOLDERS = tuple(FOLDER_RESOURCE_TYPES)
ACTIVITY_TYPES = ("resources:add", "resources:remove")
ACTIVITY_LIMIT = 60
PROGRESS_INTERVAL = 1.0
# The relay's socket.io maxHttpBufferSize is 1 MB; keep list pushes safely under it.
MAX_LIST_BYTES = 900_000

PAIR_CODE_RE = re.compile(r"^[0-9a-f]{6}$", re.I)
UPGRADED_KEY_RE = re.compile(r"^[0-9a-f]{128}$", re.I)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.I)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_key(value: str | None) -> str:
    key = (value or "").strip().lower()
    if not (PAIR_CODE_RE.match(key) or UPGRADED_KEY_RE.match(key)):
        raise ValueError("Enter the 6-character pairing code shown on civitai.com")
    return key


def normalize_sha256(value: str | None) -> str:
    sha = (value or "").strip().lower()
    if not _SHA256_RE.match(sha):
        raise ValueError("Resource hash is not a SHA256")
    return sha


def safe_filename(name: str | None) -> str:
    """The bare filename from a Link resource name; rejects traversal and empty names."""
    cleaned = (name or "").replace("\\", "/").strip()
    base = os.path.basename(cleaned).strip()
    if not base or base in (".", "..") or _CONTROL_RE.search(base):
        raise ValueError(f"Invalid resource filename {name!r}")
    return base


def folder_for_resource(resource_type: str | None, file_type: str | None = None) -> str | None:
    """ComfyUI folder for a Link resource, or None when the type has no local home. The file's own
    Civitai type wins when more specific (a Checkpoint whose file is a Diffusion Model)."""
    default = LINK_TYPE_FOLDERS.get((resource_type or "").strip())
    if default is None:
        return None
    return local_models.folder_for_file_type(file_type, default)


def make_response(command: dict, status: str, **fields) -> dict:
    """A `commandStatus` payload: the command echoed (the site keys activities by `id` and renders
    `resource`), plus status/timestamps and any extra fields (progress, error, resources, …)."""
    response = dict(command)
    response["status"] = status
    response.setdefault("createdAt", now_iso())
    response["updatedAt"] = now_iso()
    for key, value in fields.items():
        if value is not None:
            response[key] = value
    return response


def progress_fields(written: int, total: int, started: float, now: float) -> dict:
    """`progress` (0-100), `speed` (bytes/s) and `remainingTime` (s) for an in-flight download."""
    elapsed = max(now - started, 1e-6)
    speed = written / elapsed
    fields = {"speed": round(speed, 1)}
    if total > 0:
        fields["progress"] = round(min(written / total * 100.0, 100.0), 2)
        if speed > 0:
            fields["remainingTime"] = round(max(total - written, 0) / speed, 1)
    return fields


def resource_entry(folder: str, relative_name: str, sha256: str, *, downloading: bool = False) -> dict:
    entry = {
        "type": FOLDER_RESOURCE_TYPES.get(folder, "Other"),
        "hash": sha256.lower(),
        "name": os.path.basename(relative_name),
        "path": f"{folder}/{relative_name.replace(os.sep, '/')}",
        "hasPreview": "",
    }
    if downloading:
        entry["downloading"] = True
    return entry


def parse_resource_path(path: str | None) -> tuple[str, str] | None:
    """Split our own `folder/relative` list entry path; None for unknown folders or traversal."""
    folder, _, relative = (path or "").replace("\\", "/").partition("/")
    if folder not in FOLDER_RESOURCE_TYPES or not relative:
        return None
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return folder, relative


def build_resource_list(records, cache: dict[str, dict]) -> tuple[list[dict], list]:
    """Entries for records whose cache entry carries a SHA256, plus the records still to resolve."""
    entries: list[dict] = []
    pending = []
    for record in records:
        sha256 = ((cache.get(str(record.path)) or {}).get("hashes") or {}).get("SHA256")
        if sha256:
            entries.append(resource_entry(record.folder, record.name, str(sha256)))
        else:
            pending.append(record)
    return entries, pending


def trim_resource_list(entries: list[dict], max_bytes: int = MAX_LIST_BYTES) -> list[dict]:
    import json

    if len(json.dumps(entries)) <= max_bytes:
        return entries
    slim = [{k: v for k, v in e.items() if k != "path"} for e in entries]
    while slim and len(json.dumps(slim)) > max_bytes:
        slim.pop()
    return slim


def unique_destination(dest_dir: str, name: str, sha256: str, exists: Callable[[str], bool] = os.path.exists) -> str:
    """`dest_dir/name`, or a hash-suffixed sibling when that file already exists (never clobber a
    file whose contents we haven't verified)."""
    path = os.path.join(dest_dir, name)
    if not exists(path):
        return path
    stem, ext = os.path.splitext(name)
    candidate = os.path.join(dest_dir, f"{stem}_{sha256[:8]}{ext}")
    counter = 2
    while exists(candidate):
        candidate = os.path.join(dest_dir, f"{stem}_{sha256[:8]}_{counter}{ext}")
        counter += 1
    return candidate


class Activities:
    """Recent add/remove responses keyed by command id (newest last), served to `activities:list`."""

    def __init__(self, limit: int = ACTIVITY_LIMIT):
        self._limit = limit
        self._items: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def upsert(self, response: dict) -> None:
        key = str(response.get("id") or "")
        if not key:
            return
        with self._lock:
            self._items[key] = response
            while len(self._items) > self._limit:
                self._items.popitem(last=False)

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._items.get(key)

    def list(self) -> list[dict]:
        with self._lock:
            return list(self._items.values())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
