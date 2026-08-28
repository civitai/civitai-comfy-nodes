"""Registers same-origin proxy routes so the catalog picker JS can search Civitai (no CORS), the
Civitai sidebar can list the user's generations, and each node learns its expected ecosystem.
No-op when imported outside ComfyUI (e.g. pytest)."""

import asyncio
import logging
import os
import re
import threading
import time
import uuid

import requests

from . import catalog, link
from .errors import CivitaiAuthError, CivitaiNodeError

_log = logging.getLogger("civitai_comfy_nodes.server_routes")

try:
    from aiohttp import web
    from server import PromptServer

    _server = PromptServer.instance
except Exception:
    _server = None


# ── Generation gallery: flatten workflows → media items (pure; unit-tested without ComfyUI) ──────

_BLOB_KINDS = {"image", "video", "audio", "model3d"}
TRACE_URL_TERMINAL_GRACE_SECONDS = 10.0
TRACE_URL_POLL_DELAY_SECONDS = 0.5


def _kind_from_media_ref(value: str | None) -> str | None:
    if not value:
        return None
    lower = value.lower()
    if lower.startswith(("http://", "https://")):
        lower = requests.utils.urlparse(lower).path
    if lower.endswith((".mp4", ".webm", ".mov", ".mkv")):
        return "video"
    if lower.endswith((".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a")):
        return "audio"
    if lower.endswith((".glb", ".gltf", ".fbx")):
        return "model3d"
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
        return "image"
    return None

_EXT_KINDS = {
    ext: kind
    for kind, exts in {
        "image": (".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif"),
        "video": (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"),
        "audio": (".mp3", ".flac", ".wav", ".ogg", ".opus", ".m4a", ".aac"),
        "model3d": (".glb", ".gltf", ".fbx", ".obj", ".stl", ".ply", ".usdz"),
    }.items()
    for ext in exts
}


def _walk_blobs(node, key=None):
    """Yield (blob, containing_key) for every blob anywhere in a step output. A blob is any dict with
    the required Blob fields (`id` + `available`); the `type` discriminator is NOT reliable because
    System.Text.Json only writes it when a property is declared as the base `Blob` — concrete
    `ImageBlob`/`VideoBlob` outputs carry no `type` field, so kind comes from the property name."""
    if isinstance(node, dict):
        if "id" in node and "available" in node:
            yield node, key
            return
        for k, value in node.items():
            yield from _walk_blobs(value, k)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_blobs(value, key)


def _blob_kind(blob: dict, key: str | None) -> str:
    """image | video | audio | model3d | other — from the polymorphic `type` if present, else the
    file extension in the blob id (customComfy asset ids keep the original output filename, and the
    containing key — `blobs`/`tempBlobs` — says nothing about the media type), else the property
    name. `other` = non-media or unidentifiable blobs (nodepack snapshot layers, extensionless
    customComfy assets); the UI shows them as plain files."""
    declared = blob.get("type")
    if declared in _BLOB_KINDS:
        return declared
    ext = os.path.splitext(str(blob.get("id") or ""))[1].lower()
    if ext in _EXT_KINDS:
        return _EXT_KINDS[ext]
    name = (key or "").lower()
    if "video" in name:
        return "video"
    if "audio" in name:
        return "audio"
    if "model" in name or "fbx" in name or "3d" in name:
        return "model3d"
    # frames, thumbnails, samples, and the singular ImageBlob-typed `blob` fields (convertImage,
    # imageUpload, humanoidImageMask) — concretely declared, so they never carry `type`.
    if "image" in name or "frame" in name or "thumb" in name or "sample" in name or name == "blob":
        return "image"
    return "other"


def flatten_generations(workflows: list, kinds: set | None = None) -> list:
    """Slim a workflow list down to displayable media items, dropping blocked/unavailable blobs and
    workflows with no usable media. Generic blob-walk handles image/video/audio/3D step outputs."""
    items = []
    for workflow in workflows:
        media = []
        for step in workflow.get("steps") or []:
            for blob, key in _walk_blobs(step.get("output")):
                if blob.get("available") is False or blob.get("blockedReason"):
                    continue
                url = blob.get("url")
                preview = blob.get("previewUrl") or url
                if not (url or preview):
                    continue
                kind = _blob_kind(blob, key)
                if kinds and kind not in kinds:
                    continue
                media.append(
                    {
                        "kind": kind,
                        "url": url,
                        "previewUrl": preview,
                        "width": blob.get("width"),
                        "height": blob.get("height"),
                        "blobId": blob.get("id"),
                    }
                )
        if not media:
            continue
        items.append(
            {
                "workflowId": workflow.get("id"),
                "createdAt": workflow.get("createdAt"),
                "status": workflow.get("status"),
                "cost": (workflow.get("cost") or {}).get("total"),
                "media": media,
                "meta": workflow.get("metadata") or {},
            }
        )
    return items


def _guess_ext(kind: str, data: bytes) -> str:
    head = data[:12]
    if head.startswith(b"\x89PNG"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:4] == b"GIF8":
        return ".gif"
    if head[4:8] == b"ftyp":
        return ".mp4"
    if head.startswith(b"fLaC"):
        return ".flac"
    if head.startswith(b"ID3") or head[:2] == b"\xff\xfb":
        return ".mp3"
    if head[:4] == b"glTF":
        return ".glb"
    return {"image": ".png", "video": ".mp4", "audio": ".flac", "model3d": ".glb"}.get(kind, ".bin")


def _new_client(*, interactive: bool = False):
    from .client import OrchestrationClient
    from .config import resolve_config

    return OrchestrationClient(resolve_config(interactive=interactive))


def _scope_tags(scope: str | None) -> list[str] | None:
    """Map a gallery scope to the tag filter: 'session' = this ComfyUI process's generations,
    'source' = any from this pack across the user's sessions, anything else = no filter."""
    from .config import SOURCE_TAG, session_tag

    if scope == "session":
        return [SOURCE_TAG, session_tag()]
    if scope == "source":
        return [SOURCE_TAG]
    return None


def _list_generations(cursor: str | None, take: int, tags: list[str] | None = None) -> dict:
    # The gallery shows the user's OWN history, so don't hide their mature content. The list API
    # defaults hideMatureContent=true, which nulls the url + sets blockedReason on every R+ blob —
    # that dropped fully-mature workflows entirely and showed only the SFW frames of a batch.
    return _new_client().query_workflows(cursor=cursor, take=take, hide_mature=False, tags=tags)


def _validate_and_save_key(key: str) -> None:
    from . import oauth
    from .client import OrchestrationClient
    from .config import ClientConfig, base_url

    OrchestrationClient(ClientConfig(base_url=base_url(), token=key)).query_workflows(take=1)  # 401s if invalid
    oauth.save_api_key(key)


def _import_blob(blob_id: str | None, url: str | None, kind: str) -> dict:
    import folder_paths  # ComfyUI runtime

    client = _new_client()
    data = client.download_blob({"id": blob_id, "url": url})
    safe = "".join(c for c in (blob_id or uuid.uuid4().hex) if c.isalnum() or c in "-_")[:48]
    name = f"civitai_{safe}{_guess_ext(kind, data)}"
    path = os.path.join(folder_paths.get_input_directory(), name)
    with open(path, "wb") as handle:
        handle.write(data)
    return {"name": name, "subfolder": "", "type": "input"}


def _import_model(air: str) -> dict:
    """Model Library import: download a version's primary file — plus its *required* CLIP/VAE
    component files — into the matching ComfyUI model folders (same folder rules as the Model
    Selector node). Returns {"files": [{"folder", "name"}, …]} in download order, primary first."""
    from . import local_models
    from .config import auth_state

    token = auth_state()[0]
    files = catalog.components(air, token=token)
    primary = files.get("primary") or {}
    folder = local_models.folder_for_file_type(primary.get("type"), local_models.folder_for_air(air))
    path = local_models.download_model(air, folder=folder, token=token, in_execution=False)
    downloaded = [{"folder": folder, "name": os.path.basename(path)}]
    for bucket, fallback in (("clip", "text_encoders"), ("vae", "vae")):
        for f in files.get(bucket) or []:
            if not f.get("isRequired"):
                continue
            comp_folder = local_models.folder_for_file_type(f.get("type"), fallback)
            path = local_models.download_model(
                air,
                folder=comp_folder,
                token=token,
                download_url=f["downloadUrl"],
                file_id=f["id"],
                in_execution=False,
            )
            downloaded.append({"folder": comp_folder, "name": os.path.basename(path)})
    return {"files": downloaded}
def _download_url_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=300)
    if response.status_code >= 400:
        raise CivitaiNodeError(f"Asset download failed ({response.status_code})")
    return response.content


def _write_bytes_to_input(data: bytes, kind: str = "image") -> dict:
    import folder_paths  # ComfyUI runtime

    safe = uuid.uuid4().hex[:16]
    name = f"civitai_offload_{safe}{_guess_ext(kind, data)}"
    path = os.path.join(folder_paths.get_input_directory(), name)
    with open(path, "wb") as handle:
        handle.write(data)
    return {"name": name, "subfolder": "", "type": "input"}


def _write_bytes_to_output(data: bytes, kind: str = "image") -> dict:
    import folder_paths  # ComfyUI runtime

    safe = uuid.uuid4().hex[:16]
    name = f"civitai_offload_{safe}{_guess_ext(kind, data)}"
    path = os.path.join(folder_paths.get_output_directory(), name)
    with open(path, "wb") as handle:
        handle.write(data)
    result = {"filename": name, "subfolder": "", "type": "output", "kind": kind}
    asset = _register_output_asset(path, name)
    if asset:
        result["asset"] = asset
    return result


def _register_output_asset(path: str, name: str) -> dict | None:
    try:
        from app.assets.services.ingest import register_file_in_place  # ComfyUI runtime

        result = register_file_in_place(abs_path=path, name=name, tags=["output"])
        return {
            "id": result.ref.id,
            "name": result.ref.name,
            "asset_hash": result.asset.hash,
            "size": result.asset.size_bytes,
            "mime_type": result.asset.mime_type,
            "tags": result.tags,
        }
    except Exception:
        _log.debug("Could not register offload output asset", exc_info=True)
        return None


def _workflow_asset_urls(workflow: dict) -> list[str]:
    return [item["url"] for item in _workflow_asset_items(workflow)]


def _workflow_asset_items(workflow: dict) -> list[dict]:
    urls: list[str] = []
    items: list[dict] = []
    for step in workflow.get("steps") or []:
        output = step.get("output") or {}
        assets = output.get("assets")
        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, str) and asset:
                    kind = _kind_from_media_ref(asset) or "image"
                    if asset not in urls:
                        urls.append(asset)
                        items.append({"url": asset, "kind": kind})
                elif isinstance(asset, dict):
                    url = asset.get("url") or asset.get("previewUrl")
                    if url:
                        declared = asset.get("kind") or asset.get("type")
                        kind = declared if declared in _BLOB_KINDS else None
                        kind = kind or _kind_from_media_ref(url) or _kind_from_media_ref(asset.get("name")) or "image"
                        if url not in urls:
                            urls.append(url)
                            items.append({"url": url, "kind": kind})
        for blob, _key in _walk_blobs(output):
            url = blob.get("url") or blob.get("previewUrl")
            if url:
                kind = _blob_kind(blob, _key)
                if url not in urls:
                    urls.append(url)
                    items.append({"url": url, "kind": kind})
    return items


def _offload_output_node_ids(offload_result: dict) -> list[str]:
    from . import offload

    workflow = (offload_result.get("offload") or {}).get("workflow") or {}
    output_ids = []
    for node_id, node in workflow.items():
        class_type = str((node or {}).get("class_type") or "")
        if offload._is_output_node(class_type):  # keep output detection aligned with the builder
            output_ids.append(str(node_id))
    return sorted(output_ids, key=offload._node_sort_key)


def _publish_local_output_preview(
    output_nodes: list[str],
    outputs: list[dict],
    *,
    prompt_id: str | None,
    sid: str | None,
) -> None:
    if not output_nodes or not outputs:
        return
    try:
        from server import PromptServer  # ComfyUI runtime
    except Exception:
        return

    # The first offloaded output node gets the returned customComfy assets. This mirrors Comfy's
    # SaveImage websocket shape, but uses local filenames created in this user's output directory.
    node_id = output_nodes[0]
    preview_outputs = _preview_output_items(outputs)
    if not preview_outputs:
        return
    output_key = _preview_output_key(outputs)
    PromptServer.instance.send_sync(
        "executed",
        {
            "node": node_id,
            "display_node": node_id,
            "output": {output_key: preview_outputs},
            "prompt_id": prompt_id,
        },
        sid,
    )


def _preview_output_key(outputs: list[dict]) -> str:
    return "audio" if (outputs[0].get("kind") if outputs else None) == "audio" else "images"


def _preview_output_items(outputs: list[dict]) -> list[dict]:
    return [
        {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
        for item in outputs
        if item.get("filename")
    ]


def _publish_local_job_history(
    prompt: dict,
    output_nodes: list[str],
    outputs: list[dict],
    *,
    prompt_id: str,
    workflow_id: str | None,
    started_ms: int | None = None,
    completed_ms: int | None = None,
) -> None:
    if not output_nodes or not outputs or not prompt_id:
        return
    try:
        from server import PromptServer  # ComfyUI runtime
    except Exception:
        return

    preview_outputs = _preview_output_items(outputs)
    if not preview_outputs:
        return

    now_ms = int(time.time() * 1000)
    # Report the real remote runtime: the frontend derives duration from these two timestamps, so
    # start == end (both now_ms) renders as "0.00s". Fall back to now only when usage lacks times.
    start_ms = started_ms if started_ms is not None else now_ms
    end_ms = completed_ms if completed_ms is not None and completed_ms >= start_ms else now_ms
    node_id = output_nodes[0]
    output_key = _preview_output_key(outputs)
    prompt_queue = getattr(PromptServer.instance, "prompt_queue", None)
    if prompt_queue is None:
        return

    extra_data = {
        "create_time": start_ms,
        "extra_pnginfo": {
            "workflow": {
                "id": workflow_id or prompt_id,
                "source": "civitai_offload",
            }
        },
    }
    history_item = {
        "prompt": (0, prompt_id, prompt, extra_data, output_nodes),
        "outputs": {node_id: {output_key: preview_outputs}},
        "status": {
            "status_str": "success",
            "completed": True,
            "messages": [
                ("execution_start", {"prompt_id": prompt_id, "timestamp": start_ms}),
                ("execution_success", {"prompt_id": prompt_id, "timestamp": end_ms}),
            ],
        },
    }
    with prompt_queue.mutex:
        prompt_queue.history[prompt_id] = history_item
    try:
        PromptServer.instance.queue_updated()
    except Exception:
        _log.debug("Could not notify Comfy queue update for offload history", exc_info=True)


def _poll_workflow_to_terminal(client, workflow: dict, timeout_minutes: float, on_update=None) -> dict:
    workflow_id = workflow.get("id") or workflow.get("workflowId")
    if not workflow_id:
        return workflow
    deadline = time.monotonic() + max(1.0, timeout_minutes) * 60
    current = workflow
    if on_update is not None:
        on_update(current)
    while str(current.get("status") or "").lower() not in {"succeeded", "failed", "expired", "canceled"}:
        if time.monotonic() > deadline:
            raise CivitaiNodeError(f"Civitai workflow {workflow_id} timed out")
        current = client.get_workflow(workflow_id, wait=10)
        if on_update is not None:
            on_update(current)
    return current


_TERMINAL_STATUSES = {"succeeded", "failed", "expired", "canceled"}


def _epoch_ms_from_iso(value) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        from datetime import datetime

        return int(datetime.fromisoformat(text).timestamp() * 1000)
    except ValueError:
        return None


def _extract_usage(workflow: dict) -> dict | None:
    for step in workflow.get("steps") or []:
        usage = (step.get("output") or {}).get("usage")
        if isinstance(usage, dict):
            return usage
    return None


def _workflow_run_times(workflow: dict) -> tuple[int | None, int | None]:
    """Real (start_ms, end_ms) of the offloaded compute, from the customComfy usage — so the local
    history entry reports the actual GPU runtime instead of a zero-length (start == end) span."""
    usage = _extract_usage(workflow) or {}
    start = _epoch_ms_from_iso(usage.get("startedAt"))
    end = _epoch_ms_from_iso(usage.get("computedAt"))
    if start is not None and end is None:
        runtime = usage.get("runtimeSeconds")
        if isinstance(runtime, (int, float)) and not isinstance(runtime, bool):
            end = start + int(runtime * 1000)
    return start, end


# Buzz wallet (accountType) -> display name; the colour IS the currency (mirrors base._BUZZ_WALLETS).
_BUZZ_WALLETS = {"yellow": "Yellow", "blue": "Blue", "green": "Green", "fakeRed": "Red"}


def _transactions(workflow: dict) -> list[dict]:
    """Settled per-wallet charges as [{amount, currency, refund}] (credit = refund), for the
    editor's job-details popup."""
    result = []
    for t in ((workflow.get("transactions") or {}).get("list")) or []:
        amount = t.get("amount")
        if amount is None:
            continue
        account = t.get("accountType")
        result.append(
            {"amount": amount, "currency": _BUZZ_WALLETS.get(account, account), "refund": t.get("type") == "credit"}
        )
    return result


def _buzz_message(workflow: dict) -> dict | None:
    """The `civitai.buzz` ws payload for a workflow snapshot — the pinned rate while running and the
    final ceiled charge at terminal — or None when there's no usage/cost to show yet. Snake_case keys
    + epoch-ms dates mirror the comfy-cloud meter so both consumers share one frontend shape."""
    status = str(workflow.get("status") or "").lower()
    terminal = status in _TERMINAL_STATUSES
    usage = _extract_usage(workflow) or {}
    rate = usage.get("buzzPerSecond")
    estimated = usage.get("estimatedCost")
    total = (workflow.get("cost") or {}).get("total")
    if terminal and total is not None:
        # Prefer the authoritative settled charge so the meter snaps to the real bill.
        estimated = total
    if rate is None and estimated is None:
        return None
    message = {
        "prompt_id": workflow.get("id") or workflow.get("workflowId"),
        "status": status,
        "terminal": terminal,
        "buzz_per_second": rate or 0,
        "runtime_seconds": usage.get("runtimeSeconds") or 0,
        "estimated_cost": estimated if estimated is not None else 0,
        "started_at": _epoch_ms_from_iso(usage.get("startedAt")),
        "computed_at": _epoch_ms_from_iso(usage.get("computedAt")),
    }
    # The settled per-wallet charge rides the terminal frame for the job-details popup.
    if terminal:
        message["transactions"] = _transactions(workflow)
        if total is not None:
            message["cost_total"] = total
    return message


def _send_buzz(sid: str | None, workflow: dict) -> None:
    if not sid:
        return
    message = _buzz_message(workflow)
    if message is None:
        return
    try:
        from server import PromptServer  # ComfyUI runtime

        PromptServer.instance.send_sync("civitai.buzz", message, sid)
    except Exception:
        _log.debug("Could not push civitai.buzz frame", exc_info=True)


_RUNNING_ENTRY_TTL_SECONDS = 60 * 60
_running_lock = threading.Lock()
_running_seq = 0
_running_injected_at: dict[int, float] = {}
# workflow_id -> {"task_id": int|None, "sid": str|None, "queue_state": dict|None}
_active_offloads: dict[str, dict] = {}


def _prompt_queue():
    try:
        from server import PromptServer  # ComfyUI runtime
    except Exception:
        return None
    return getattr(PromptServer.instance, "prompt_queue", None)


def _notify_queue_updated() -> None:
    try:
        from server import PromptServer  # ComfyUI runtime

        PromptServer.instance.queue_updated()
    except Exception:
        _log.debug("Could not notify Comfy queue update", exc_info=True)


def _reap_stale_running_entries() -> None:
    """Backstop for a finalize thread that died before its `finally` could remove its running entry."""
    now = time.monotonic()
    with _running_lock:
        stale = [tid for tid, at in _running_injected_at.items() if now - at >= _RUNNING_ENTRY_TTL_SECONDS]
        for tid in stale:
            _running_injected_at.pop(tid, None)
    if not stale:
        return
    pq = _prompt_queue()
    if pq is None or not hasattr(pq, "currently_running"):
        return
    with pq.mutex:
        for tid in stale:
            pq.currently_running.pop(tid, None)
    _notify_queue_updated()


def _inject_running_queue(
    prompt: dict, output_nodes: list[str], *, prompt_id: str, workflow_id: str | None
) -> int | None:
    """Make the offloaded job show as running: the queue sidebar lists prompt_queue.currently_running,
    so insert an entry there (negative key — never collides with ComfyUI's counter, no executor touches
    it). Returns the key to remove at terminal."""
    if not prompt_id:
        return None
    pq = _prompt_queue()
    if pq is None or not hasattr(pq, "currently_running"):
        return None
    extra_data = {
        "create_time": int(time.time() * 1000),
        "extra_pnginfo": {"workflow": {"id": workflow_id or prompt_id, "source": "civitai_offload"}},
    }
    item = (0, prompt_id, prompt, extra_data, list(output_nodes or []))
    try:
        global _running_seq
        with _running_lock:
            _running_seq -= 1
            task_id = _running_seq
            _running_injected_at[task_id] = time.monotonic()
        with pq.mutex:
            pq.currently_running[task_id] = item
    except Exception:
        _log.debug("Could not inject offload running entry", exc_info=True)
        return None
    _notify_queue_updated()
    return task_id


def _remove_running_queue(task_id: int | None, *, notify: bool = True) -> None:
    # `notify=False` at terminal: the caller publishes history immediately after and notifies once,
    # so the frontend's single /api/jobs refetch sees the job already in history — never in neither
    # list (a two-notify remove-then-publish leaves a window where a refetch drops the row entirely).
    if task_id is None:
        return
    with _running_lock:
        _running_injected_at.pop(task_id, None)
    pq = _prompt_queue()
    if pq is None or not hasattr(pq, "currently_running"):
        return
    try:
        with pq.mutex:
            existed = pq.currently_running.pop(task_id, None) is not None
    except Exception:
        _log.debug("Could not remove offload running entry", exc_info=True)
        return
    if existed and notify:
        _notify_queue_updated()


def _emit_event(sid: str | None, event: str, data: dict) -> None:
    if not sid:
        return
    try:
        from server import PromptServer  # ComfyUI runtime

        PromptServer.instance.send_sync(event, data, sid)
    except Exception:
        _log.debug("Could not emit %s frame", event, exc_info=True)


def _broadcast_event(event: str, data: dict) -> None:
    try:
        from server import PromptServer  # ComfyUI runtime

        PromptServer.instance.send_sync(event, data)
    except Exception:
        _log.debug("Could not broadcast %s frame", event, exc_info=True)


def _execution_error_data(prompt_id: str, status: str, *, message: str | None = None) -> dict:
    return {
        "prompt_id": prompt_id,
        "node_id": "",
        "node_type": "",
        "executed": [],
        "exception_message": message or f"The workflow {status.lower()} on Civitai.",
        "exception_type": "CivitaiOrchestration",
        "traceback": [],
        "timestamp": int(time.time() * 1000),
    }


def _emit_lifecycle_transition(
    sid: str | None, prompt_id: str, old_status: str, new_status: str, *, error: dict | None = None
) -> None:
    """Drive the queued row's progress bar: the frontend binds it to execution_start's prompt_id, so
    emit the native lifecycle frames on each status edge."""
    if not sid or not prompt_id:
        return
    old = (old_status or "").lower()
    new = (new_status or "").lower()
    if old == new:
        return
    now = int(time.time() * 1000)
    was_terminal = old in _TERMINAL_STATUSES
    if new == "processing" and old != "processing" and not was_terminal:
        _emit_event(sid, "execution_start", {"prompt_id": prompt_id, "timestamp": now})
        _emit_event(sid, "executing", {"node": None, "prompt_id": prompt_id})
        return
    if new in _TERMINAL_STATUSES and not was_terminal:
        if new == "succeeded":
            _emit_event(sid, "executing", {"node": None, "prompt_id": prompt_id})
            _emit_event(sid, "execution_success", {"prompt_id": prompt_id, "timestamp": now})
        elif new in {"canceled", "cancelled"}:
            _emit_event(
                sid,
                "execution_interrupted",
                {"prompt_id": prompt_id, "node_id": "", "node_type": "", "executed": [], "timestamp": now},
            )
        else:  # failed / expired
            _emit_event(sid, "execution_error", error or _execution_error_data(prompt_id, new))


def _max_progress_rate(workflow: dict) -> float | None:
    best = None
    for step in workflow.get("steps") or []:
        for job in step.get("jobs") or []:
            rate = job.get("estimatedProgressRate")
            if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                best = rate if best is None else max(best, rate)
    return best


def _emit_progress(sid: str | None, prompt_id: str, workflow: dict) -> None:
    """A coarse baseline `progress` frame from estimatedProgressRate; rewritten trace frames refine it."""
    if not sid:
        return
    rate = _max_progress_rate(workflow)
    if rate is None:
        return
    rate = max(0.0, min(1.0, rate))
    _emit_event(sid, "progress", {"value": int(round(rate * 1000)), "max": 1000, "prompt_id": prompt_id, "node": None})


def _preparation_progress(workflow: dict) -> float | None:
    """The download fraction shown during `preparing`: the max estimatedProgressRate among jobs still
    preparing (the size-weighted fraction of the job's resources already local to the worker)."""
    best = None
    for step in workflow.get("steps") or []:
        for job in step.get("jobs") or []:
            if str(job.get("status") or "").lower() != "preparing":
                continue
            rate = job.get("estimatedProgressRate")
            if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                best = rate if best is None else max(best, rate)
    return best


def _queue_phase(status: str) -> str | None:
    """Map an orchestration WorkflowStatus to the queue-state overlay phase, or None if it carries no
    phase label (the native lifecycle frames handle terminal rows)."""
    s = (status or "").lower()
    if s in _TERMINAL_STATUSES or s == "cancelled":
        return s
    if s == "processing":
        return "processing"
    if s in {"unassigned", "preparing", "scheduled"}:
        return "preparing"
    return None


def _queue_state_data(workflow: dict, prompt_id: str) -> dict | None:
    phase = _queue_phase(str(workflow.get("status") or ""))
    if phase is None:
        return None
    data = {"prompt_id": prompt_id, "status": phase}
    if phase == "preparing":
        progress = _preparation_progress(workflow)
        if progress is not None:
            data["progress"] = max(0.0, min(1.0, progress))
    return data


def _send_queue_state(sid: str | None, prompt_id: str, workflow: dict, *, seed: bool = False) -> None:
    """Out-of-band phase label for the injected queue row. The row reports the zod-safe `in_progress`
    wire status; `preparing`/`Starting…` live here (civitai.queue_state) so an unknown value never
    empties /api/jobs. `civitai-queue-state.js` relabels the row and hands off to the native progress
    bar on the first real trace frame."""
    if not sid or not prompt_id:
        return
    data = _queue_state_data(workflow, prompt_id)
    if data is None:
        return
    if seed:
        data["seed"] = True
    _emit_event(sid, "civitai.queue_state", data)


def _offload_active() -> dict:
    """Reconnect gap-closer for civitai-queue-state.js: the still-preparing/processing offloads with
    their phase + download fraction. Native /api/jobs prunes these passthrough fields, so the overlay
    reads them here on (re)attach to restore the label instead of the row's native 'Running'."""
    with _running_lock:
        snapshot = [(wid, dict(info)) for wid, info in _active_offloads.items()]
    jobs = []
    for wid, info in snapshot:
        state = info.get("queue_state") or {}
        status = state.get("status")
        if status not in ("preparing", "processing"):
            continue
        jobs.append(
            {
                "id": wid,
                "civitai_orch_status": status,
                "civitai_preparation_progress": state.get("progress"),
            }
        )
    return {"jobs": jobs}


def _publish_failed_job_history(
    prompt: dict, output_nodes: list[str], *, prompt_id: str, workflow_id: str | None, message: str
) -> None:
    """Land a failed offload in history so it shows in the Failed tab instead of vanishing from the queue."""
    if not prompt_id:
        return
    pq = _prompt_queue()
    if pq is None or not hasattr(pq, "history"):
        return
    now_ms = int(time.time() * 1000)
    extra_data = {
        "create_time": now_ms,
        "extra_pnginfo": {"workflow": {"id": workflow_id or prompt_id, "source": "civitai_offload"}},
    }
    history_item = {
        "prompt": (0, prompt_id, prompt, extra_data, list(output_nodes or [])),
        "outputs": {},
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [
                ("execution_start", {"prompt_id": prompt_id, "timestamp": now_ms}),
                ("execution_error", {"prompt_id": prompt_id, "exception_message": message, "timestamp": now_ms}),
            ],
        },
    }
    try:
        with pq.mutex:
            pq.history[prompt_id] = history_item
    except Exception:
        _log.debug("Could not publish failed offload history", exc_info=True)
        return
    _notify_queue_updated()


def _cancel_offload(workflow_id: str) -> None:
    """Cancel the orchestrator workflow and drop its running row now; the poll loop would also converge
    on `canceled`, but this is snappier."""
    from .client import OrchestrationClient
    from .config import resolve_config

    with _running_lock:
        info = dict(_active_offloads.get(workflow_id) or {})
    try:
        OrchestrationClient(resolve_config(interactive=False)).cancel_workflow(workflow_id)
    except Exception:
        _log.warning("offload cancel: upstream cancel failed for %s", workflow_id, exc_info=True)
    if info:
        _emit_event(
            info.get("sid"),
            "execution_interrupted",
            {
                "prompt_id": workflow_id,
                "node_id": "",
                "node_type": "",
                "executed": [],
                "timestamp": int(time.time() * 1000),
            },
        )
        _remove_running_queue(info.get("task_id"))


def _queue_local_prompt(comfy_base_url: str, prompt: dict) -> dict:
    response = requests.post(
        f"{comfy_base_url.rstrip('/')}/prompt",
        json={"prompt": prompt, "client_id": "civitai-offload-hybrid"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise CivitaiNodeError(f"Local Comfy continuation queue failed ({response.status_code}): {response.text}")
    return response.json()


def _offload_inventory() -> dict:
    from . import model_resolve, offload

    return {
        "models": [record.as_dict() for record in model_resolve.scan_local_model_files()],
        "nodepacks": [nodepack.as_dict() for nodepack in offload.scan_installed_nodepacks()],
    }


def _extract_trace_url(workflow: dict) -> str | None:
    for step in workflow.get("steps") or []:
        url = (step.get("output") or {}).get("traceUrl")
        if url:
            return url
    return None


def _push_offload_status(sid: str | None, state: str, **fields) -> None:
    """Push a terminal offload status (`done`/`error`) to the originating tab over the local /ws.
    Best-effort: a no-op outside ComfyUI or if the socket is gone."""
    try:
        from server import PromptServer  # ComfyUI runtime
    except Exception:
        return
    try:
        PromptServer.instance.send_sync("civitai.offload.status", {"state": state, **fields}, sid)
    except Exception:
        _log.debug("Could not push offload status", exc_info=True)


class _TraceTailHandle:
    def __init__(self, thread: threading.Thread, stop_event: threading.Event, box: dict):
        self._thread = thread
        self._stop_event = stop_event
        self._box = box

    def stop(self, grace: float = 10.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=grace)

    def drain(self, grace: float = 10.0) -> None:
        self._thread.join(timeout=grace)
        if self._thread.is_alive():
            self.stop(grace=1.0)

    def summary(self) -> dict | None:
        stats = self._box.get("stats")
        return stats.as_dict() if stats is not None else None


def _start_trace_tail(
    config, workflow: dict, *, sid: str | None, prompt_id: str | None = None
) -> _TraceTailHandle | None:
    """Spawn a daemon thread that waits for the customComfy traceUrl and replays it locally. `prompt_id`
    (the synthetic id) is stamped onto each frame in place of the worker's so the bar binds to our row."""
    from . import trace_tail
    from .client import OrchestrationClient

    workflow_id = workflow.get("id") or workflow.get("workflowId")
    prompt_id = prompt_id or workflow_id
    trace_url = _extract_trace_url(workflow)
    if not trace_url and not workflow_id:
        return None

    stop_event = threading.Event()
    box: dict = {}

    def _run():
        resolved_url = trace_url
        poll_client = OrchestrationClient(config)
        terminal_seen_at = None
        while not resolved_url and workflow_id and not stop_event.is_set():
            try:
                current = poll_client.get_workflow(workflow_id, wait=5)
            except CivitaiNodeError:
                if stop_event.wait(1.0):
                    return
                continue
            resolved_url = _extract_trace_url(current)
            if resolved_url:
                break
            status = str(current.get("status") or "").lower()
            if status in {"succeeded", "failed", "expired", "canceled"}:
                now = time.monotonic()
                terminal_seen_at = terminal_seen_at or now
                if now - terminal_seen_at >= TRACE_URL_TERMINAL_GRACE_SECONDS:
                    return
                if stop_event.wait(TRACE_URL_POLL_DELAY_SECONDS):
                    return
            else:
                terminal_seen_at = None
        if resolved_url and not stop_event.is_set():
            box["stats"] = trace_tail.tail_trace_to_websocket(
                resolved_url, stop_event=stop_event, sid=sid, prompt_id=prompt_id
            )

    thread = threading.Thread(target=_run, name="civitai-trace-tail", daemon=True)
    thread.start()
    return _TraceTailHandle(thread, stop_event, box)


def _offload_submit(
    prompt: dict,
    selected_node_ids: list[str] | None,
    workflow: dict | None,
    *,
    whatif: bool,
    do_tail: bool,
) -> dict:
    """Build the customComfy offload and submit it with wait=0 so the caller gets the workflow id
    back immediately. The long-running poll + local replay happen later in `_offload_finalize`."""
    from . import offload
    from .client import OrchestrationClient
    from .config import resolve_config, stored_buzz_account, stored_min_vram_gb, stored_use_sage_attention

    config = resolve_config(interactive=False)
    config.buzz_account = stored_buzz_account()
    client = OrchestrationClient(config)
    started = time.monotonic()
    _log.info("offload submit: building customComfy payload (whatif=%s)", whatif)
    build = offload.build_custom_comfy_offload(
        prompt,
        selected_node_ids=selected_node_ids,
        workflow=workflow,
        token=config.token,
        trace="binary" if do_tail else None,
        min_vram_gb=stored_min_vram_gb(),
        use_sage_attention=stored_use_sage_attention(),
        session_owner_api_token=config.token,
        upload_blob_file=client.upload_blob_file,
    )
    built = time.monotonic()
    submitted = client.submit_steps(build.steps, wait=0, whatif=whatif)
    _log.info(
        "offload submit: workflow=%s build=%.2fs submit=%.2fs",
        submitted.get("id") or submitted.get("workflowId"),
        built - started,
        time.monotonic() - built,
    )
    return {"config": config, "build": build, "workflow": submitted}


def _offload_finalize(
    prompt: dict,
    build,
    config,
    workflow: dict,
    comfy_base_url: str,
    *,
    sid: str | None,
    do_tail: bool,
) -> None:
    """Background half of an offload run: tail the trace onto the local /ws, poll to completion,
    then download the result and queue the local continuation. Runs in a daemon thread, so it
    reports terminal state via a `civitai.offload.status` ws event instead of an HTTP response."""
    from .client import OrchestrationClient

    workflow_id = workflow.get("id") or workflow.get("workflowId")
    client = OrchestrationClient(config)
    _reap_stale_running_entries()
    running_output_nodes = _offload_output_node_ids({"offload": build.as_dict()})
    running_task_id = _inject_running_queue(
        prompt, running_output_nodes, prompt_id=workflow_id, workflow_id=workflow_id
    )
    if workflow_id:
        with _running_lock:
            _active_offloads[workflow_id] = {
                "task_id": running_task_id,
                "sid": sid,
                "queue_state": _queue_state_data(workflow, workflow_id),
            }
        # Override the injected row's native "Running" with the real phase ASAP.
        _send_queue_state(sid, workflow_id, workflow)
    tail = _start_trace_tail(config, workflow, sid=sid, prompt_id=workflow_id) if do_tail else None
    started = time.monotonic()
    _log.info("offload finalize: polling workflow %s to completion (tail=%s)", workflow_id, tail is not None)
    # The editor extrapolates the per-second Buzz tick from the rate between polls, so coarse polling
    # is fine.
    last_status = {"value": str(workflow.get("status") or "")}

    def on_update(wf):
        new_status = str(wf.get("status") or "")
        if sid:
            _send_buzz(sid, wf)
            _send_queue_state(sid, workflow_id, wf)
            _emit_lifecycle_transition(sid, workflow_id, last_status["value"], new_status)
            _emit_progress(sid, workflow_id, wf)
        if workflow_id:
            with _running_lock:
                info = _active_offloads.get(workflow_id)
                if info is not None:
                    info["queue_state"] = _queue_state_data(wf, workflow_id)
        # Re-assert the queue each poll while still running: the injected row lives in
        # currently_running the whole run, but queue_updated only fires at inject + terminal, so a
        # frontend that dropped the row mid-run would never re-fetch it back. This keeps it visible.
        if new_status.lower() not in _TERMINAL_STATUSES:
            _notify_queue_updated()
        last_status["value"] = new_status

    try:
        try:
            final = _poll_workflow_to_terminal(client, workflow, config.timeout_minutes, on_update=on_update)
        except Exception as exc:
            if tail is not None:
                tail.stop()
            _emit_lifecycle_transition(
                sid,
                workflow_id,
                last_status["value"],
                "failed",
                error=_execution_error_data(workflow_id, "failed", message=str(exc)),
            )
            _remove_running_queue(running_task_id)
            _publish_failed_job_history(
                prompt, running_output_nodes, prompt_id=workflow_id, workflow_id=workflow_id, message=str(exc)
            )
            _push_offload_status(sid, "error", message=str(exc))
            _log.warning("offload finalize: poll failed for %s (%s)", workflow_id, exc, exc_info=True)
            return
        _log.info(
            "offload finalize: workflow %s reached %s in %.2fs",
            workflow_id,
            final.get("status"),
            time.monotonic() - started,
        )
        if tail is not None:
            tail.drain()

        offload_result = {"workflow": final, "offload": build.as_dict()}
        try:
            local = _run_local_tail(
                prompt, offload_result, comfy_base_url, client_id=sid, running_task_id=running_task_id
            )
        except Exception as exc:
            _remove_running_queue(running_task_id)
            _publish_failed_job_history(
                prompt, running_output_nodes, prompt_id=workflow_id, workflow_id=workflow_id, message=str(exc)
            )
            _push_offload_status(sid, "error", message=str(exc))
            _log.warning("offload finalize: local tail failed (%s)", exc, exc_info=True)
            return

        _push_offload_status(
            sid,
            "done",
            workflowId=final.get("id") or final.get("workflowId"),
            promptId=((local or {}).get("queue") or {}).get("prompt_id"),
        )
        _log.info("offload finalize: workflow %s done in %.2fs total", workflow_id, time.monotonic() - started)
    finally:
        _remove_running_queue(running_task_id)
        if workflow_id:
            with _running_lock:
                _active_offloads.pop(workflow_id, None)


def _run_local_tail(
    prompt: dict,
    offload_result: dict,
    comfy_base_url: str,
    *,
    client_id: str | None = None,
    running_task_id: int | None = None,
) -> dict | None:
    from . import offload

    assets = _workflow_asset_items(offload_result["workflow"])
    if not assets:
        raise CivitaiNodeError("Civitai workflow completed but returned no downloadable customComfy assets")
    asset = assets[0]
    kind = asset.get("kind") or "image"
    data = _download_url_bytes(asset["url"])
    local_output = _write_bytes_to_output(data, kind=kind)
    output_nodes = _offload_output_node_ids(offload_result)
    workflow_id = (offload_result.get("workflow") or {}).get("id") or (offload_result.get("workflow") or {}).get(
        "workflowId"
    )
    _publish_local_output_preview(
        output_nodes,
        [local_output],
        prompt_id=workflow_id,
        sid=client_id,
    )
    # Remove the running row without notifying, then publish history with the single notify, so the
    # frontend's one refetch sees running-gone AND history-present together — the job moves straight
    # from Running to Completed with no in-between refetch that would drop it from both lists.
    _remove_running_queue(running_task_id, notify=False)
    started_ms, completed_ms = _workflow_run_times(offload_result.get("workflow") or {})
    _publish_local_job_history(
        prompt,
        output_nodes,
        [local_output],
        prompt_id=workflow_id,
        workflow_id=workflow_id,
        started_ms=started_ms,
        completed_ms=completed_ms,
    )
    continuation = None
    if kind == "image":
        continuation = offload.build_local_continuation_prompt(
            prompt,
            remote_node_ids=offload_result["offload"].get("included_node_ids") or [],
            imported_image_name="civitai_offload_result.png",
        )
    if continuation is None:
        return {
            "imported": None,
            "output": local_output,
            "outputNodeIds": output_nodes,
            "continuation": None,
            "queue": None,
        }
    imported = _write_bytes_to_input(data, kind="image")
    continuation.prompt[continuation.bridge_node_id]["inputs"]["image"] = imported["name"]
    queue = _queue_local_prompt(comfy_base_url, continuation.prompt)
    return {
        "imported": imported,
        "output": local_output,
        "outputNodeIds": output_nodes,
        "continuation": continuation.as_dict(),
        "queue": queue,
    }


def node_ecosystem_map() -> dict:
    """Map each recipe node class -> its expected AIR ecosystem (for the picker's default filter)."""
    from . import NODE_CLASS_MAPPINGS  # noqa: PLC0415 - deferred to call time to avoid an import cycle

    result = {}
    for name, cls in NODE_CLASS_MAPPINGS.items():
        eco = catalog.node_ecosystem(getattr(cls, "DISCRIMINATOR", None) or {})
        if eco:
            result[name] = eco
    return result


def _pack_config_payload() -> dict:
    from . import config as cfg

    stored_url = cfg.stored_orchestrator_url()
    source = "env" if os.environ.get("CIVITAI_ORCHESTRATION_URL") else "stored" if stored_url else "default"
    return {
        "orchestratorUrl": stored_url or "",
        "orchestratorEffective": cfg.base_url(),
        "orchestratorDefault": cfg.DEFAULT_BASE_URL,
        "orchestratorSource": source,
        "minVramGb": cfg.stored_min_vram_gb(),
        "vramTiers": cfg.VRAM_TIERS,
        "allowMatureContent": cfg.stored_mature_content(),
        "useSageAttention": cfg.stored_use_sage_attention(),
        "buzzAccount": cfg.stored_buzz_account(),
        "buzzAccounts": list(cfg.BUZZ_ACCOUNTS),
        "gpuGeneration": cfg.GPU_GENERATION_LABEL,
        "enableOffload": cfg.stored_enable_offload(),
        "enableRecipeNodes": cfg.stored_enable_recipe_nodes(),
        "enableLink": cfg.stored_enable_link(),
        "linkUrl": cfg.stored_link_url() or "",
        "linkUrlEffective": cfg.link_url(),
        "linkUrlSource": link.status()["urlSource"],
    }


def _buzz_accounts_payload() -> dict:
    """Balances for the Pay-with picker. Raises CivitaiAuthError without stored credentials."""
    from . import buzz
    from .config import _NO_CREDS_MESSAGE, auth_state

    token, _source = auth_state()
    if not token:
        raise CivitaiAuthError(_NO_CREDS_MESSAGE)
    return buzz.fetch_buzz_accounts(token)


def _apply_pack_config_update(body: dict) -> None:
    """Validate a settings patch from POST /civitai/config and persist it. Raises ValueError on bad
    input (the route maps it to HTTP 400). `gpuGeneration` is display-only and ignored."""
    from . import config as cfg

    settings = cfg.load_pack_settings()
    if "orchestratorUrl" in body:
        url = (body.get("orchestratorUrl") or "").strip().rstrip("/")
        if url and not url.startswith(("http://", "https://")):
            raise ValueError("Orchestrator URL must start with http:// or https://")
        if url:
            settings["orchestratorUrl"] = url
        else:
            settings.pop("orchestratorUrl", None)
    if "minVramGb" in body:
        vram = body.get("minVramGb")
        if vram in (None, "", 0):
            settings.pop("minVramGb", None)
        elif vram in cfg.VRAM_TIERS:
            settings["minVramGb"] = vram
        else:
            raise ValueError(f"minVramGb must be one of {cfg.VRAM_TIERS}")
    if "allowMatureContent" in body:
        mode = body.get("allowMatureContent")
        if mode not in cfg.MATURE_CONTENT_MODES:
            raise ValueError(f"allowMatureContent must be one of {list(cfg.MATURE_CONTENT_MODES)}")
        settings["allowMatureContent"] = mode
    if "useSageAttention" in body:
        settings["useSageAttention"] = bool(body.get("useSageAttention"))
    if "buzzAccount" in body:
        account = body.get("buzzAccount")
        if account not in cfg.BUZZ_ACCOUNTS:
            raise ValueError(f"buzzAccount must be one of {list(cfg.BUZZ_ACCOUNTS)}")
        settings["buzzAccount"] = account
    if "enableOffload" in body:
        settings["enableOffload"] = bool(body.get("enableOffload"))
    if "enableRecipeNodes" in body:
        settings["enableRecipeNodes"] = bool(body.get("enableRecipeNodes"))
    if "enableLink" in body:
        settings["enableLink"] = bool(body.get("enableLink"))
    if "linkUrl" in body:
        url = (body.get("linkUrl") or "").strip().rstrip("/")
        if url and not url.startswith(("http://", "https://")):
            raise ValueError("Civitai Link URL must start with http:// or https://")
        if url:
            settings["linkUrl"] = url
        else:
            settings.pop("linkUrl", None)
    cfg.save_pack_settings(settings)
    if "enableLink" in body or "linkUrl" in body:
        link.reconfigure()


def _link_pair(body: dict) -> dict:
    return link.pair(str((body or {}).get("code") or ""))


_AIR_IDS_RE = re.compile(r":civitai:(\d+)@(\d+)")


def _known_models() -> dict[str, dict]:
    """Local model files already resolved to a Civitai version (from the hash cache, no hashing),
    keyed by the Model Library's `folder/relative-name` node key."""
    from . import link_protocol, model_cache, model_resolve

    records = model_resolve.scan_local_model_files(model_resolve.model_roots_by_folder(link_protocol.LIST_FOLDERS))
    cache = model_cache.bulk_get([record.path for record in records])
    known: dict[str, dict] = {}
    for record in records:
        air = (cache.get(str(record.path)) or {}).get("air") or ""
        match = _AIR_IDS_RE.search(air)
        if not match:
            continue
        model_id, version_id = match.groups()
        known[f"{record.folder}/{record.name.replace(os.sep, '/')}"] = {
            "air": air,
            "modelId": int(model_id),
            "modelVersionId": int(version_id),
            "url": catalog.CIVITAI_MODEL_URL.format(model_id=model_id, version_id=version_id),
        }
    return known


if _server is not None:

    @_server.routes.get("/civitai/catalog/search")
    async def _civitai_catalog_search(request):
        query = (request.query.get("query") or "").strip()
        type_ = request.query.get("type") or None
        ecosystem = request.query.get("ecosystem") or None
        try:
            limit = max(1, min(int(request.query.get("limit", "60")), 100))
        except ValueError:
            limit = 60
        loop = asyncio.get_event_loop()
        try:
            entries = await loop.run_in_executor(None, lambda: catalog.search(query, type_, ecosystem, limit))
        except Exception as e:  # surface upstream/Civitai failures to the picker
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response({"entries": entries})

    @_server.routes.get("/civitai/catalog/lookup")
    async def _civitai_catalog_lookup(request):
        air = (request.query.get("air") or "").strip()
        if not air:
            return web.json_response({"error": "air is required"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            entry = await loop.run_in_executor(None, lambda: catalog.lookup(air))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response({"entry": entry})

    @_server.routes.post("/civitai/catalog/resolve")
    async def _civitai_catalog_resolve(request):
        body = await request.json()
        text = (body.get("input") or "").strip()
        if not text:
            return web.json_response({"error": "input is required"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            entry = await loop.run_in_executor(None, lambda: catalog.resolve_reference(text))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        if not entry:
            return web.json_response(
                {"error": "Couldn't resolve that to a Civitai model — paste a model/version URL or AIR."},
                status=404,
            )
        return web.json_response({"entry": entry})

    @_server.routes.get("/civitai/catalog/meta")
    async def _civitai_catalog_meta(request):
        ecosystems = [{"key": e["key"], "label": e["label"]} for e in catalog.ECOSYSTEMS]
        return web.json_response(
            {"ecosystems": ecosystems, "nodeEcosystems": node_ecosystem_map(), "types": catalog.CATALOG_TYPES}
        )

    @_server.routes.get("/civitai/auth/status")
    async def _civitai_auth_status(request):
        from .config import auth_state

        token, source = auth_state()
        return web.json_response({"authenticated": bool(token), "source": source})

    @_server.routes.post("/civitai/auth/api-key")
    async def _civitai_auth_api_key(request):
        body = await request.json()
        key = (body.get("apiKey") or "").strip()
        if not key:
            return web.json_response({"error": "API key is empty"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: _validate_and_save_key(key))
        except Exception as e:  # invalid/rejected key
            return web.json_response({"error": f"Key rejected: {e}"}, status=401)
        return web.json_response({"ok": True})

    @_server.routes.post("/civitai/auth/login")
    async def _civitai_auth_login(request):
        from . import oauth

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, oauth.interactive_login)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response({"ok": True})

    @_server.routes.post("/civitai/auth/logout")
    async def _civitai_auth_logout(request):
        from . import oauth

        oauth.clear_credentials()
        return web.json_response({"ok": True})

    @_server.routes.get("/civitai/config")
    async def _civitai_config_get(request):
        return web.json_response(_pack_config_payload())

    @_server.routes.post("/civitai/config")
    async def _civitai_config_post(request):
        body = await request.json()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: _apply_pack_config_update(body))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True})

    @_server.routes.get("/civitai/buzz/accounts")
    async def _civitai_buzz_accounts(request):
        loop = asyncio.get_event_loop()
        try:
            payload = await loop.run_in_executor(None, _buzz_accounts_payload)
        except CivitaiAuthError:
            return web.json_response({"error": "auth_required"}, status=401)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response(payload)

    @_server.routes.get("/civitai/models/known")
    async def _civitai_models_known(request):
        loop = asyncio.get_event_loop()
        return web.json_response(await loop.run_in_executor(None, _known_models))

    @_server.routes.get("/civitai/link/status")
    async def _civitai_link_status(request):
        return web.json_response(link.status())

    @_server.routes.post("/civitai/link/pair")
    async def _civitai_link_pair(request):
        body = await request.json()
        loop = asyncio.get_event_loop()
        try:
            payload = await loop.run_in_executor(None, lambda: _link_pair(body))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response(payload)

    @_server.routes.post("/civitai/link/cancel")
    async def _civitai_link_cancel(request):
        body = await request.json()
        try:
            return web.json_response(link.cancel(str((body or {}).get("id") or "")))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=404)

    @_server.routes.post("/civitai/link/unpair")
    async def _civitai_link_unpair(request):
        loop = asyncio.get_event_loop()
        return web.json_response(await loop.run_in_executor(None, link.unpair))

    @_server.routes.get("/civitai/workflows/list")
    async def _civitai_workflows_list(request):
        cursor = request.query.get("cursor") or None
        kinds = request.query.get("kinds")
        kind_set = set(kinds.split(",")) if kinds else None
        tags = _scope_tags(request.query.get("scope"))
        try:
            take = max(1, min(int(request.query.get("take", "60")), 200))
        except ValueError:
            take = 60
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, lambda: _list_generations(cursor, take, tags))
        except CivitaiAuthError:
            return web.json_response({"error": "auth_required"}, status=401)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        items = flatten_generations(data.get("items") or [], kind_set)
        return web.json_response({"next": data.get("next"), "items": items})

    @_server.routes.post("/civitai/workflows/import")
    async def _civitai_workflows_import(request):
        body = await request.json()
        blob_id = body.get("blobId")
        url = body.get("url")
        kind = body.get("kind") or "image"
        if not (blob_id or url):
            return web.json_response({"error": "blobId or url required"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, lambda: _import_blob(blob_id, url, kind))
        except CivitaiAuthError:
            return web.json_response({"error": "auth_required"}, status=401)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response(result)

    @_server.routes.post("/civitai/models/import")
    async def _civitai_models_import(request):
        body = await request.json()
        air = (body.get("air") or "").strip()
        if not air:
            return web.json_response({"error": "air is required"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, lambda: _import_model(air))
        except Exception as e:  # metadata/download failures -> picker toast
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response(result)

    @_server.routes.get("/civitai/offload/inventory")
    async def _civitai_offload_inventory(request):
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, _offload_inventory)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response(data)

    @_server.routes.get("/civitai/offload/active")
    async def _civitai_offload_active(request):
        return web.json_response(_offload_active())

    @_server.routes.post("/civitai/offload/run")
    async def _civitai_offload_run(request):
        body = await request.json()
        prompt = body.get("prompt") or body.get("output")
        if not isinstance(prompt, dict):
            return web.json_response({"error": "prompt must be a ComfyUI API prompt object"}, status=400)
        selected = body.get("selectedNodeIds") or body.get("selected_node_ids") or None
        if selected is not None and not isinstance(selected, list):
            return web.json_response({"error": "selectedNodeIds must be an array"}, status=400)
        workflow = body.get("workflow")
        if workflow is not None and not isinstance(workflow, dict):
            return web.json_response({"error": "workflow must be a serialized ComfyUI workflow object"}, status=400)
        whatif = bool(body.get("whatif", False))
        run_local_tail = bool(body.get("runLocalTail", False))
        live_progress = bool(body.get("liveProgress", True))
        client_id = body.get("clientId")
        if not isinstance(client_id, str):
            client_id = None
        comfy_base_url = f"{request.scheme}://{request.host}"
        selected_ids = [str(node_id) for node_id in selected] if selected else None
        run_background = run_local_tail and not whatif
        do_tail = run_background and live_progress
        loop = asyncio.get_event_loop()
        try:
            submit = await loop.run_in_executor(
                None,
                lambda: _offload_submit(prompt, selected_ids, workflow, whatif=whatif, do_tail=do_tail),
            )
        except CivitaiAuthError:
            return web.json_response({"error": "auth_required"}, status=401)
        except CivitaiNodeError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

        submitted_workflow = submit["workflow"]
        response = {"workflow": submitted_workflow, "offload": submit["build"].as_dict()}
        trace_url = _extract_trace_url(submitted_workflow)
        if trace_url:
            response["traceUrl"] = trace_url
        if run_background:
            threading.Thread(
                target=_offload_finalize,
                args=(prompt, submit["build"], submit["config"], submitted_workflow, comfy_base_url),
                kwargs={"sid": client_id, "do_tail": do_tail},
                name="civitai-offload-finalize",
                daemon=True,
            ).start()
        return web.json_response(response)

    @_server.routes.post("/civitai/offload/cancel")
    async def _civitai_offload_cancel(request):
        body = await request.json()
        workflow_id = body.get("workflowId") or body.get("promptId")
        if not isinstance(workflow_id, str) or not workflow_id:
            return web.json_response({"error": "workflowId is required"}, status=400)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: _cancel_offload(workflow_id))
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response({"ok": True})


if _server is not None:
    link.register(notify=_broadcast_event)
