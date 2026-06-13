from .civitai_comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

# Registers the /civitai/catalog/search proxy route when running inside ComfyUI; no-op otherwise.
from .civitai_comfy_nodes import server_routes  # noqa: E402, F401

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
