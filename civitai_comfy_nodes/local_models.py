"""Download Civitai models and load them as ComfyUI MODEL/CLIP/VAE, so a Civitai loader can feed
local (non-Civitai) nodes like KSampler. comfy/folder_paths imports are guarded so the package
still imports under pytest without ComfyUI."""

import glob
import hashlib
import os
import re
import threading
from collections.abc import Callable

import requests

from . import comfy_compat
from .errors import CivitaiNodeError

CIVITAI_DOWNLOAD_URL = "https://civitai.com/api/download/models/{version_id}"
USER_AGENT = "civitai-comfy-nodes/0.1 (+https://github.com/civitai/civitai-comfy-nodes)"

# AIR type segment (urn:air:{eco}:{type}:civitai:...) -> the ComfyUI model folder to download into,
# so the file lands where the matching standard loader (Load Checkpoint, LoraLoader, …) reads from.
AIR_TYPE_FOLDERS = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "lycoris": "loras",
    "dora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "embedding": "embeddings",
    "hypernet": "hypernetworks",
    "upscaler": "upscale_models",
    "unet": "diffusion_models",
    "textencoder": "text_encoders",
    "clipvision": "clip_vision",
    "motion": "animatediff_models",
}


class DownloadCanceledError(CivitaiNodeError):
    pass


class DownloadHttpError(CivitaiNodeError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def version_id_from_air(air: str) -> str:
    match = re.search(r"@(\d+)", air or "")
    if not match:
        raise CivitaiNodeError(f"Cannot parse a Civitai version id from AIR '{air}'")
    return match.group(1)


def folder_for_air(air: str, default: str = "checkpoints") -> str:
    """The ComfyUI model folder for an AIR's type segment (checkpoint->checkpoints, lora->loras, …)."""
    parts = (air or "").split(":")
    air_type = parts[3] if len(parts) > 3 else ""
    return AIR_TYPE_FOLDERS.get(air_type, default)


# "Model"/"Pruned Model" are deliberately absent: every model type's primary file carries them.
FILE_TYPE_FOLDERS = {
    "Diffusion Model": "diffusion_models",
    "UNet": "diffusion_models",
    "VAE": "vae",
    "Text Encoder": "text_encoders",
    "CLIPVision": "clip_vision",
    "ControlNet": "controlnet",
    "Upscaler": "upscale_models",
    "Negative": "embeddings",
}


def folder_for_file_type(file_type: str | None, default: str = "checkpoints") -> str:
    """The ComfyUI model folder for a Civitai file `type` (Diffusion Model -> diffusion_models, …),
    falling back to `default` for non-model types (Config, Archive, …) or unknown ones."""
    return FILE_TYPE_FOLDERS.get((file_type or "").strip(), default)


def _model_dir(folder: str) -> str:
    import folder_paths

    dirs = folder_paths.get_folder_paths(folder)
    if not dirs:
        raise CivitaiNodeError(f"ComfyUI has no '{folder}' model directory configured")
    os.makedirs(dirs[0], exist_ok=True)
    return dirs[0]


def _filename(response: requests.Response, version_id: str, prefix: str) -> str:
    disposition = response.headers.get("content-disposition") or ""
    match = re.search(r'filename="?([^";]+)"?', disposition)
    name = match.group(1) if match else f"{version_id}.safetensors"
    # Prefix so the cache lookup is a cheap glob and names never collide.
    return f"{prefix}{name}"


def _open_download(url: str, token: str | None, *, label: str) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
    if response.status_code >= 400:
        status = response.status_code
        response.close()
        hint = " (this model may be gated — connect a Civitai Auth node)" if status in (401, 403) else ""
        raise DownloadHttpError(f"Civitai download failed ({status}) for {label}{hint}", status)
    return response


def _stream_to_file(
    response: requests.Response,
    path: str,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    in_execution: bool = False,
) -> str:
    """Stream the response body to `path` via a `.part` sibling; returns the body's SHA256 hex
    (hashed as it streams, so no second read). `cancel` is checked per chunk."""
    tmp = path + ".part"
    total = int(response.headers.get("content-length") or 0)
    bar = comfy_compat.progress_bar(total or 100) if in_execution else None
    digest = hashlib.sha256()
    written = 0
    try:
        with open(tmp, "wb") as out:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if in_execution:
                    comfy_compat.check_interrupted()
                if cancel is not None and cancel.is_set():
                    raise DownloadCanceledError("Download canceled")
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if bar and total:
                    bar.update_absolute(written, total)
                if on_progress is not None:
                    on_progress(written, total)
    except BaseException:
        response.close()
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, path)
    return digest.hexdigest()


def stream_download(
    url: str,
    dest_path: str,
    *,
    token: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
    in_execution: bool = False,
) -> str:
    """Download `url` to exactly `dest_path` (no cache lookup, no name prefix); returns the SHA256 hex."""
    response = _open_download(url, token, label=url)
    return _stream_to_file(response, dest_path, on_progress=on_progress, cancel=cancel, in_execution=in_execution)


def download_model(
    air: str,
    folder: str = "checkpoints",
    token: str | None = None,
    *,
    download_url: str | None = None,
    file_id: int | str | None = None,
    in_execution: bool = True,
) -> str:
    """Download a Civitai resource into ComfyUI's model directory; returns the local path. Cached so
    a second use loads from disk. Pass `download_url` + `file_id` to fetch a specific additional file
    of the version (e.g. a VAE or text encoder) rather than the primary file; the file id keeps the
    cache key distinct from the primary and from sibling files that share the same folder.

    Pass `in_execution=False` when calling outside a running prompt (e.g. the Model Library import
    route): the ProgressBar hook dereferences per-prompt server state (PromptServer.last_prompt_id —
    AttributeError before the first prompt ever runs), and the interrupt flag belongs to whichever
    prompt is executing, so checking it would let an unrelated cancel abort this download."""
    version_id = version_id_from_air(air)
    dest_dir = _model_dir(folder)
    # File id keeps sibling component files sharing a folder (several text encoders) from colliding.
    prefix = f"civitai_{version_id}_f{file_id}_" if file_id is not None else f"civitai_{version_id}_"
    cached = glob.glob(os.path.join(dest_dir, f"{prefix}*"))
    cached = [p for p in cached if not p.endswith(".part")]
    if cached:
        return cached[0]

    url = download_url or CIVITAI_DOWNLOAD_URL.format(version_id=version_id)
    response = _open_download(url, token, label=f"version {version_id}")
    path = os.path.join(dest_dir, _filename(response, version_id, prefix))
    _stream_to_file(response, path, in_execution=in_execution)
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
