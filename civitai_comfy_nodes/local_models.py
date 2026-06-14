"""Download Civitai models and load them as ComfyUI MODEL/CLIP/VAE, so a Civitai loader can feed
local (non-Civitai) nodes like KSampler. comfy/folder_paths imports are guarded so the package
still imports under pytest without ComfyUI."""

import glob
import os
import re

import requests

from . import comfy_compat
from .errors import CivitaiNodeError

CIVITAI_DOWNLOAD_URL = "https://civitai.com/api/download/models/{version_id}"
USER_AGENT = "civitai-comfy-nodes/0.1 (+https://github.com/civitai/civitai-comfy-nodes)"


def version_id_from_air(air: str) -> str:
    match = re.search(r"@(\d+)", air or "")
    if not match:
        raise CivitaiNodeError(f"Cannot parse a Civitai version id from AIR '{air}'")
    return match.group(1)


def _model_dir(folder: str) -> str:
    import folder_paths

    dirs = folder_paths.get_folder_paths(folder)
    if not dirs:
        raise CivitaiNodeError(f"ComfyUI has no '{folder}' model directory configured")
    os.makedirs(dirs[0], exist_ok=True)
    return dirs[0]


def _filename(response: requests.Response, version_id: str) -> str:
    disposition = response.headers.get("content-disposition") or ""
    match = re.search(r'filename="?([^";]+)"?', disposition)
    name = match.group(1) if match else f"{version_id}.safetensors"
    # Prefix with the version id so the cache lookup is a cheap glob and names never collide.
    return f"civitai_{version_id}_{name}"


def download_model(air: str, folder: str = "checkpoints", token: str | None = None) -> str:
    """Download a Civitai resource into ComfyUI's model directory; returns the local path.
    Cached by version id, so a second use loads from disk without re-downloading."""
    version_id = version_id_from_air(air)
    dest_dir = _model_dir(folder)
    cached = glob.glob(os.path.join(dest_dir, f"civitai_{version_id}_*"))
    cached = [p for p in cached if not p.endswith(".part")]
    if cached:
        return cached[0]

    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = CIVITAI_DOWNLOAD_URL.format(version_id=version_id)
    response = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
    if response.status_code >= 400:
        response.close()
        hint = " (this model may be gated — connect a Civitai Auth node)" if response.status_code in (401, 403) else ""
        raise CivitaiNodeError(f"Civitai download failed ({response.status_code}) for version {version_id}{hint}")

    path = os.path.join(dest_dir, _filename(response, version_id))
    tmp = path + ".part"
    total = int(response.headers.get("content-length") or 0)
    bar = comfy_compat.progress_bar(total or 100)
    written = 0
    try:
        with open(tmp, "wb") as out:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                comfy_compat.check_interrupted()
                out.write(chunk)
                written += len(chunk)
                if total:
                    bar.update_absolute(written, total)
    except BaseException:
        response.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str):
    """Local checkpoint -> (MODEL, CLIP, VAE) via ComfyUI's loader."""
    try:
        import comfy.sd
        import folder_paths
    except ImportError as e:
        raise CivitaiNodeError("Loading a model locally requires the ComfyUI runtime.") from e
    out = comfy.sd.load_checkpoint_guess_config(
        path,
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return out[0], out[1], out[2]


def apply_lora(model, clip, path: str, strength: float):
    """Apply a local LoRA file onto (MODEL, CLIP); returns the patched (MODEL, CLIP)."""
    try:
        import comfy.sd
        import comfy.utils
    except ImportError as e:
        raise CivitaiNodeError("Applying a LoRA locally requires the ComfyUI runtime.") from e
    lora = comfy.utils.load_torch_file(path, safe_load=True)
    return comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
