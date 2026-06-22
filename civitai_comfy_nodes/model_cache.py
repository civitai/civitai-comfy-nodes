"""Persistent cache of local-model -> Civitai AIR resolutions, keyed on file identity.

Hashing a multi-GB checkpoint is CPU-bound (SHA256 runs at ~1-2 GB/s regardless of disk speed) and
runs on every offload submit. Caching the computed hashes and the resolved AIR keyed on
``(size, mtime_ns)`` lets repeat submits skip it. Positive AIR results are stable for a model
version, so they are cached indefinitely; a changed file gets a new identity and is re-resolved.
Best-effort: any IO/JSON error degrades to "no cache" rather than failing the offload.
"""

import json
import os
import threading
from pathlib import Path

_lock = threading.Lock()


def cache_store_path() -> Path:
    override = os.environ.get("CIVITAI_COMFY_MODEL_CACHE")
    if override:
        return Path(override)
    from .config import settings_store_path

    return settings_store_path().parent / "model-air-cache.json"


def _identity(path: str | os.PathLike[str]) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _read() -> dict:
    try:
        data = json.loads(cache_store_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict) -> None:
    path = cache_store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except OSError:
        pass


def get(path: str | os.PathLike[str]) -> dict | None:
    """Return the cached entry for ``path`` when the file is unchanged (size + mtime match), else
    None. Entry shape: ``{"size", "mtime_ns", "hashes": {...}, "air": str|None, "model_version_id"}``.
    """
    identity = _identity(path)
    if identity is None:
        return None
    with _lock:
        entry = _read().get(str(path))
    if not isinstance(entry, dict):
        return None
    if entry.get("size") != identity[0] or entry.get("mtime_ns") != identity[1]:
        return None
    return entry


def put(
    path: str | os.PathLike[str],
    *,
    hashes: dict,
    air: str | None = None,
    model_version_id=None,
) -> None:
    identity = _identity(path)
    if identity is None:
        return
    entry: dict = {"size": identity[0], "mtime_ns": identity[1], "hashes": hashes or {}}
    if air:
        entry["air"] = air
        entry["model_version_id"] = model_version_id
    with _lock:
        data = _read()
        data[str(path)] = entry
        _write(data)
