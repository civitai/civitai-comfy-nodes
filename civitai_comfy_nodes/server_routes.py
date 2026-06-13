"""Registers same-origin proxy routes so the catalog picker JS can search Civitai (no CORS)
and learn each node's expected ecosystem. No-op when imported outside ComfyUI (e.g. pytest)."""

import asyncio

from . import catalog

try:
    from aiohttp import web
    from server import PromptServer

    _server = PromptServer.instance
except Exception:
    _server = None


def node_ecosystem_map() -> dict:
    """Map each recipe node class -> its expected AIR ecosystem (for the picker's default filter)."""
    from . import NODE_CLASS_MAPPINGS  # noqa: PLC0415 - deferred to call time to avoid an import cycle

    result = {}
    for name, cls in NODE_CLASS_MAPPINGS.items():
        discriminator = getattr(cls, "DISCRIMINATOR", None) or {}
        model_air = None
        fields = getattr(cls, "FIELDS", {}) or {}
        if "model" in {f.api for f in fields.values()}:
            try:
                default = cls.INPUT_TYPES().get("optional", {}).get("model", (None, {}))[1].get("default")
                model_air = default if isinstance(default, str) and "air:" in default else None
            except Exception:
                model_air = None
        eco = catalog.node_ecosystem(discriminator, model_air)
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

    @_server.routes.get("/civitai/catalog/meta")
    async def _civitai_catalog_meta(request):
        ecosystems = [{"key": e["key"], "label": e["label"]} for e in catalog.ECOSYSTEMS]
        return web.json_response({"ecosystems": ecosystems, "nodeEcosystems": node_ecosystem_map()})
