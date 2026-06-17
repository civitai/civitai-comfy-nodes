"""Registers same-origin proxy routes so the catalog picker JS can search Civitai (no CORS), the
Civitai sidebar can list the user's generations, and each node learns its expected ecosystem.
No-op when imported outside ComfyUI (e.g. pytest)."""

import asyncio
import os
import uuid

from . import catalog
from .errors import CivitaiAuthError

try:
    from aiohttp import web
    from server import PromptServer

    _server = PromptServer.instance
except Exception:
    _server = None


# ── Generation gallery: flatten workflows → media items (pure; unit-tested without ComfyUI) ──────

_BLOB_KINDS = {"image", "video", "audio", "model3d"}


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
    name = (key or "").lower()
    if "video" in name:
        return "video"
    if "audio" in name:
        return "audio"
    if "model" in name or "fbx" in name or "3d" in name:
        return "model3d"
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


def node_ecosystem_map() -> dict:
    """Map each recipe node class -> its expected AIR ecosystem (for the picker's default filter)."""
    from . import NODE_CLASS_MAPPINGS  # noqa: PLC0415 - deferred to call time to avoid an import cycle

    result = {}
    for name, cls in NODE_CLASS_MAPPINGS.items():
        eco = catalog.node_ecosystem(getattr(cls, "DISCRIMINATOR", None) or {})
        if eco:
            result[name] = eco
    return result


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
