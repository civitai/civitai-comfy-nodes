"""Civitai catalogue search — proxies civitai.com/api/v1/models and flattens each model's
versions into pickable AIR entries. Ported from comfy-cloud's CivitaiCatalogClient so the
catalog picker works in any ComfyUI, not just hosted comfy-cloud sessions.
"""

import requests

CIVITAI_MODELS_URL = "https://civitai.com/api/v1/models"
USER_AGENT = "civitai-comfy-nodes/0.1 (+https://github.com/civitai/civitai-comfy-nodes)"

# Civitai resource types selectable in the picker (the API's `types=` values).
CATALOG_TYPES = ["Checkpoint", "LORA", "TextualInversion", "VAE", "Controlnet", "Upscaler"]

# Civitai baseModel string -> AIR ecosystem segment. Conservative: only mappings the spine
# resource provider accepts, so we never surface AIRs that would fail downstream.
ECOSYSTEM_MAP = {
    "SD 1.4": "sd1",
    "SD 1.5": "sd1",
    "SD 1.5 LCM": "sd1",
    "SD 1.5 Hyper": "sd1",
    "SDXL 0.9": "sdxl",
    "SDXL 1.0": "sdxl",
    "SDXL Turbo": "sdxl",
    "SDXL Lightning": "sdxl",
    "SDXL Hyper": "sdxl",
    "SDXL Distilled": "sdxl",
    "Pony": "sdxl",
    "Illustrious": "sdxl",
    "NoobAI": "sdxl",
    "SD 3": "sd3",
    "SD 3.5": "sd3",
    "SD 3.5 Large": "sd3",
    "SD 3.5 Large Turbo": "sd3",
    "SD 3.5 Medium": "sd3",
    "Flux.1 D": "flux1",
    "Flux.1 S": "flux1",
    "Flux.1 Krea": "flux1",
    "Flux.1 Kontext": "flux1",
    "Flux.2 D": "flux2",
    "HiDream": "hidream",
    "HiDream o1": "hidream-o1",
    "AuraFlow": "auraflow",
    "Chroma 1 HD": "chroma",
    "Chroma 1 BASE": "chroma",
    "LTXV": "ltx2",
    "LTXV 2": "ltx2",
    "LTXV 2.3": "ltx23",
    "Wan Video 1.3B t2v": "wanvideo1_3b_t2v",
    "Wan Video 14B t2v": "wanvideo14b_t2v",
    "Wan Video 14B i2v 480p": "wanvideo14b_i2v_480p",
    "Wan Video 14B i2v 720p": "wanvideo14b_i2v_720p",
    "Upscaler": "other",
}


def ecosystem_for(base_model: str | None) -> str | None:
    if not base_model:
        return None
    return ECOSYSTEM_MAP.get(base_model)


def flatten_models(items: list, max_versions: int = 6, type_filter: str | None = None) -> list[dict]:
    """One entry per model version, skipping versions with no registered ecosystem."""
    entries = []
    for model in items or []:
        model_type = model.get("type") or ""
        if type_filter and model_type.lower() != type_filter.lower():
            continue
        lower_type = model_type.lower()
        if not lower_type:
            continue
        emitted = 0
        for version in model.get("modelVersions") or []:
            if emitted >= max_versions:
                break
            ecosystem = ecosystem_for(version.get("baseModel"))
            if not ecosystem:
                continue
            images = version.get("images") or []
            entries.append(
                {
                    "air": f"urn:air:{ecosystem}:{lower_type}:civitai:{model['id']}@{version['id']}",
                    "name": model.get("name") or f"model {model['id']}",
                    "versionName": version.get("name") or f"v{version['id']}",
                    "ecosystem": ecosystem,
                    "baseModel": version.get("baseModel") or "",
                    "type": model_type,
                    "downloadCount": (model.get("stats") or {}).get("downloadCount") or 0,
                    "thumbnailUrl": images[0].get("url") if images else None,
                }
            )
            emitted += 1
    return entries


def search(
    query: str = "", type_: str | None = None, limit: int = 60, timeout: int = 15, token: str | None = None
) -> list[dict]:
    """Search Civitai. `query` is a real keyword filter only when `types=` is omitted (an API
    quirk), so a keyword search fetches across types and filters client-side; an empty query
    fetches the top resources for the given type."""
    params: dict = {"limit": limit}
    if query:
        params["query"] = query
    elif type_:
        params["types"] = type_
        params["sort"] = "Most Downloaded"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(CIVITAI_MODELS_URL, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    items = response.json().get("items") or []
    return flatten_models(items, type_filter=type_ if query else None)
