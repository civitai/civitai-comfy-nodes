"""Registers a same-origin proxy route so the catalog picker JS can search Civitai without
CORS. No-op when imported outside ComfyUI (e.g. under pytest)."""

import asyncio

from . import catalog

try:
    from aiohttp import web
    from server import PromptServer

    _server = PromptServer.instance
except Exception:
    _server = None


if _server is not None:

    @_server.routes.get("/civitai/catalog/search")
    async def _civitai_catalog_search(request):
        query = (request.query.get("query") or "").strip()
        type_ = request.query.get("type") or None
        try:
            limit = max(1, min(int(request.query.get("limit", "60")), 100))
        except ValueError:
            limit = 60
        loop = asyncio.get_event_loop()
        try:
            entries = await loop.run_in_executor(None, lambda: catalog.search(query, type_, limit))
        except Exception as e:  # surface upstream/Civitai failures to the picker
            return web.json_response({"error": str(e)}, status=502)
        return web.json_response({"entries": entries})
