"""Registers same-origin proxy routes so the catalog picker JS can search Civitai (no CORS), the
Civitai sidebar can list the user's generations, and each node learns its expected ecosystem.
No-op when imported outside ComfyUI (e.g. pytest)."""

import asyncio
import logging
import os
import threading
import time
import uuid

import requests

from . import catalog
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
    """image | video | audio | model3d — from the polymorphic `type` if present, else the property name."""
    declared = blob.get("type")
    if declared in _BLOB_KINDS:
        return declared
    content_type = blob.get("contentType") or blob.get("content_type") or blob.get("mimeType") or blob.get("mime_type")
    if isinstance(content_type, str):
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("audio/"):
            return "audio"
        if content_type in {"model/gltf-binary", "model/gltf+json"}:
            return "model3d"
        if content_type.startswith("image/"):
            return "image"
    name = (key or "").lower()
    if "video" in name:
        return "video"
    if "audio" in name:
        return "audio"
    if "model" in name or "fbx" in name or "3d" in name:
        return "model3d"
    for field in ("id", "url", "previewUrl", "name", "filename"):
        kind = _kind_from_media_ref(blob.get(field))
        if kind:
            return kind
    return "image"  # images, frames, thumbnails, samples, and the lone `ImageBlob Blob` field


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
    node_id = output_nodes[0]
    output_key = _preview_output_key(outputs)
    prompt_queue = getattr(PromptServer.instance, "prompt_queue", None)
    if prompt_queue is None:
        return

    extra_data = {
        "create_time": now_ms,
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
                ("execution_start", {"prompt_id": prompt_id, "timestamp": now_ms}),
                ("execution_success", {"prompt_id": prompt_id, "timestamp": now_ms}),
            ],
        },
    }
    with prompt_queue.mutex:
        prompt_queue.history[prompt_id] = history_item
    try:
        PromptServer.instance.queue_updated()
    except Exception:
        _log.debug("Could not notify Comfy queue update for offload history", exc_info=True)


def _poll_workflow_to_terminal(client, workflow: dict, timeout_minutes: float) -> dict:
    workflow_id = workflow.get("id") or workflow.get("workflowId")
    if not workflow_id:
        return workflow
    deadline = time.monotonic() + max(1.0, timeout_minutes) * 60
    current = workflow
    while str(current.get("status") or "").lower() not in {"succeeded", "failed", "expired", "canceled"}:
        if time.monotonic() > deadline:
            raise CivitaiNodeError(f"Civitai workflow {workflow_id} timed out")
        current = client.get_workflow(workflow_id, wait=10)
    return current


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
    from . import offload

    return {
        "models": [record.as_dict() for record in offload.scan_local_model_files()],
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


def _start_trace_tail(config, workflow: dict, *, sid: str | None) -> _TraceTailHandle | None:
    """Spawn a daemon thread that waits for the customComfy traceUrl and replays it locally."""
    from . import trace_tail
    from .client import OrchestrationClient

    workflow_id = workflow.get("id") or workflow.get("workflowId")
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
            box["stats"] = trace_tail.tail_trace_to_websocket(resolved_url, stop_event=stop_event, sid=sid)

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
    from .config import resolve_config, stored_min_vram_gb, stored_use_sage_attention

    config = resolve_config(interactive=False)
    client = OrchestrationClient(config)
    build = offload.build_custom_comfy_offload(
        prompt,
        selected_node_ids=selected_node_ids,
        workflow=workflow,
        token=config.token,
        trace="binary" if do_tail else None,
        min_vram_gb=stored_min_vram_gb(),
        use_sage_attention=stored_use_sage_attention(),
        upload_blob_file=client.upload_blob_file,
    )
    submitted = client.submit_steps(build.steps, wait=0, whatif=whatif)
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

    client = OrchestrationClient(config)
    tail = _start_trace_tail(config, workflow, sid=sid) if do_tail else None
    try:
        final = _poll_workflow_to_terminal(client, workflow, config.timeout_minutes)
    except Exception as exc:
        if tail is not None:
            tail.stop()
        _push_offload_status(sid, "error", message=str(exc))
        _log.warning("offload finalize: poll failed (%s)", exc, exc_info=True)
        return
    if tail is not None:
        tail.drain()

    offload_result = {"workflow": final, "offload": build.as_dict()}
    try:
        local = _run_local_tail(prompt, offload_result, comfy_base_url, client_id=sid)
    except Exception as exc:
        _push_offload_status(sid, "error", message=str(exc))
        _log.warning("offload finalize: local tail failed (%s)", exc, exc_info=True)
        return

    _push_offload_status(
        sid,
        "done",
        workflowId=final.get("id") or final.get("workflowId"),
        promptId=((local or {}).get("queue") or {}).get("prompt_id"),
    )


def _run_local_tail(prompt: dict, offload_result: dict, comfy_base_url: str, *, client_id: str | None = None) -> dict | None:
    from . import offload

    assets = _workflow_asset_items(offload_result["workflow"])
    if not assets:
        raise CivitaiNodeError("Civitai workflow completed but returned no downloadable customComfy assets")
    asset = assets[0]
    kind = asset.get("kind") or "image"
    data = _download_url_bytes(asset["url"])
    local_output = _write_bytes_to_output(data, kind=kind)
    output_nodes = _offload_output_node_ids(offload_result)
    workflow_id = (
        (offload_result.get("workflow") or {}).get("id")
        or (offload_result.get("workflow") or {}).get("workflowId")
        or f"civitai-offload-{uuid.uuid4()}"
    )
    _publish_local_output_preview(
        output_nodes,
        [local_output],
        prompt_id=workflow_id,
        sid=client_id,
    )
    _publish_local_job_history(
        prompt,
        output_nodes,
        [local_output],
        prompt_id=workflow_id,
        workflow_id=workflow_id,
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
        "gpuGeneration": cfg.GPU_GENERATION_LABEL,
        "enableOffload": cfg.stored_enable_offload(),
        "enableRecipeNodes": cfg.stored_enable_recipe_nodes(),
    }


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
    if "enableOffload" in body:
        settings["enableOffload"] = bool(body.get("enableOffload"))
    if "enableRecipeNodes" in body:
        settings["enableRecipeNodes"] = bool(body.get("enableRecipeNodes"))
    cfg.save_pack_settings(settings)


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
        try:
            _apply_pack_config_update(body)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True})

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

    @_server.routes.get("/civitai/offload/inventory")
    async def _civitai_offload_inventory(request):
        loop = asyncio.get_event_loop()
        try:
            data = await loop.run_in_executor(None, _offload_inventory)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response(data)

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
