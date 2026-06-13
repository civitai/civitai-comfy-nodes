"""Civitai catalogue search — proxies civitai.com/api/v1/models and flattens each model's
versions into pickable AIR entries, with ecosystem filtering so a node only offers compatible
resources (a zImage Turbo node shouldn't list SDXL LoRAs).

Ported/extended from comfy-cloud's CivitaiCatalogClient. ECOSYSTEMS is the single source of
truth: it maps each AIR ecosystem (the `urn:air:<ecosystem>:...` segment the orchestrator matches
on, lowercase) to the Civitai `baseModel` strings that belong to it.
"""

import requests

CIVITAI_MODELS_URL = "https://civitai.com/api/v1/models"
USER_AGENT = "civitai-comfy-nodes/0.1 (+https://github.com/civitai/civitai-comfy-nodes)"
CATALOG_TYPES = ["Checkpoint", "LORA", "TextualInversion", "VAE", "Controlnet", "Upscaler"]

# AIR ecosystem -> {label, baseModels}. AIR ecosystem strings match the orchestrator's lowercase
# values (see TextToImageHandler: "qwen", "zimage", "zimagebase", "anima", ...).
ECOSYSTEMS = [
    {"key": "sd1", "label": "SD 1.x", "baseModels": ["SD 1.4", "SD 1.5", "SD 1.5 LCM", "SD 1.5 Hyper"]},
    {
        "key": "sdxl",
        "label": "SDXL / Pony / Illustrious",
        "baseModels": [
            "SDXL 0.9",
            "SDXL 1.0",
            "SDXL Turbo",
            "SDXL Lightning",
            "SDXL Hyper",
            "SDXL Distilled",
            "Pony",
            "Illustrious",
            "NoobAI",
        ],
    },
    {
        "key": "sd3",
        "label": "SD 3.x",
        "baseModels": ["SD 3", "SD 3.5", "SD 3.5 Large", "SD 3.5 Large Turbo", "SD 3.5 Medium"],
    },
    {"key": "flux1", "label": "Flux.1", "baseModels": ["Flux.1 D", "Flux.1 S", "Flux.1 Krea", "Flux.1 Kontext"]},
    {
        "key": "flux2",
        "label": "Flux.2",
        "baseModels": [
            "Flux.2 D",
            "Flux.2 Klein 9B",
            "Flux.2 Klein 9B-base",
            "Flux.2 Klein 4B",
            "Flux.2 Klein 4B-base",
        ],
    },
    {"key": "qwen", "label": "Qwen", "baseModels": ["Qwen", "Qwen 2"]},
    {"key": "zimage", "label": "Z Image Turbo", "baseModels": ["ZImageTurbo"]},
    {"key": "zimagebase", "label": "Z Image Base", "baseModels": ["ZImageBase"]},
    {"key": "anima", "label": "Anima", "baseModels": ["Anima"]},
    {"key": "hidream", "label": "HiDream", "baseModels": ["HiDream", "HiDream o1"]},
    {"key": "chroma", "label": "Chroma", "baseModels": ["Chroma 1 HD", "Chroma 1 BASE"]},
    {"key": "ltx2", "label": "LTXV 2", "baseModels": ["LTXV", "LTXV 2"]},
    {"key": "ltx23", "label": "LTXV 2.3", "baseModels": ["LTXV 2.3"]},
    {"key": "wanvideo14b_t2v", "label": "Wan Video 14B t2v", "baseModels": ["Wan Video 14B t2v"]},
    {"key": "wanvideo14b_i2v_480p", "label": "Wan Video 14B i2v 480p", "baseModels": ["Wan Video 14B i2v 480p"]},
    {"key": "wanvideo14b_i2v_720p", "label": "Wan Video 14B i2v 720p", "baseModels": ["Wan Video 14B i2v 720p"]},
    {"key": "wanvideo1_3b_t2v", "label": "Wan Video 1.3B t2v", "baseModels": ["Wan Video 1.3B t2v"]},
    {"key": "other", "label": "Upscaler / Other", "baseModels": ["Upscaler"]},
]

ECO_BY_BASEMODEL = {bm: eco["key"] for eco in ECOSYSTEMS for bm in eco["baseModels"]}
ECO_LABELS = {eco["key"]: eco["label"] for eco in ECOSYSTEMS}

# Recipe-node discriminator `ecosystem` value / engine -> AIR ecosystem key.
_DISC_ECO = {
    "sd1": "sd1",
    "sdxl": "sdxl",
    "flux1": "flux1",
    "qwen": "qwen",
    "anima": "anima",
    "flux2Dev": "flux2",
    "flux2Klein": "flux2",
}
_ENGINE_ECO = {"flux2": "flux2", "flux1-kontext": "flux1"}


def ecosystem_for(base_model: str | None) -> str | None:
    if not base_model:
        return None
    return ECO_BY_BASEMODEL.get(base_model)


def base_models_for(ecosystem: str) -> list[str]:
    return [bm for bm, key in ECO_BY_BASEMODEL.items() if key == ecosystem]


def air_ecosystem(air: str | None) -> str | None:
    """Extract the ecosystem segment from an AIR (urn:air:<ecosystem>:<type>:...)."""
    if not air or "air:" not in air:
        return None
    tail = air.split("air:", 1)[1]
    return tail.split(":", 1)[0] or None


def node_ecosystem(discriminator: dict | None, model_air: str | None = None) -> str | None:
    """The AIR ecosystem a recipe node's resources must belong to, derived from its discriminator
    (or a default model AIR for free-model nodes). None means 'no constraint' (Any)."""
    discriminator = discriminator or {}
    eco = discriminator.get("ecosystem")
    if eco == "zImage":
        return "zimagebase" if discriminator.get("model") == "base" else "zimage"
    if eco in _DISC_ECO:
        return _DISC_ECO[eco]
    engine = discriminator.get("engine")
    if engine in _ENGINE_ECO:
        return _ENGINE_ECO[engine]
    return air_ecosystem(model_air)


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
    query: str = "",
    type_: str | None = None,
    ecosystem: str | None = None,
    limit: int = 60,
    timeout: int = 15,
    token: str | None = None,
) -> list[dict]:
    """Search Civitai, filtered server-side by type and (optionally) the ecosystem's baseModels.
    An empty query returns the most-downloaded resources for the type."""
    params: list[tuple[str, str]] = [("limit", str(limit))]
    if query:
        params.append(("query", query))
    else:
        params.append(("sort", "Most Downloaded"))
    if type_:
        params.append(("types", type_))
    if ecosystem:
        for base_model in base_models_for(ecosystem):
            params.append(("baseModels", base_model))
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(CIVITAI_MODELS_URL, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    items = response.json().get("items") or []
    # `type_filter` is a backstop in case the API returns mixed types for some query combinations.
    return flatten_models(items, type_filter=type_)
