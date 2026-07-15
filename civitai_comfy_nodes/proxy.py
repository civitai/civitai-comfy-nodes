"""Module-level HTTP proxy configuration for all civitai-comfy-nodes HTTP clients.

Proxy is set either via the CivitaiProxy node at runtime or the CIVITAI_COMFY_PROXY
environment variable. Once set it persists for the lifetime of the Python process and
affects *only* the requests made by this pack — other ComfyUI custom nodes are unaffected.
"""

import os

_proxy_url: str | None = None


def set_proxy(proxy_url: str | None) -> None:
    global _proxy_url
    _proxy_url = proxy_url


def get_proxy() -> dict[str, str] | None:
    from .config import proxy_url as _config_proxy_url
    global _proxy_url
    proxy = _proxy_url or _config_proxy_url()
    if proxy:
        proxy = proxy.strip()
        if proxy:
            return {"http": proxy, "https": proxy}
    return None
