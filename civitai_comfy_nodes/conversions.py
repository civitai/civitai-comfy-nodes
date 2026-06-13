"""Convert between Comfy types (IMAGE/VIDEO/AUDIO) and the wire formats recipes accept/return.

torch/PIL/av are imported lazily: they always exist inside ComfyUI, but the codegen
and IR tests must be able to import this package without them.
"""

import base64
import io
import os
import uuid

from . import comfy_compat
from .errors import CivitaiNodeError


def image_tensor_to_data_url(tensor, index: int = 0) -> str:
    """ComfyUI IMAGE tensor (B,H,W,C) float32 [0,1] -> PNG data URL of one batch entry."""
    import numpy as np
    from PIL import Image

    if tensor.ndim != 4:
        raise CivitaiNodeError(f"Expected IMAGE tensor with shape (B,H,W,C), got {tensor.ndim}D")
    image_np = (tensor[index].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image_np).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def image_tensor_to_data_urls(tensor) -> list[str]:
    return [image_tensor_to_data_url(tensor, i) for i in range(tensor.shape[0])]


def image_tensor_to_png_bytes(tensor, index: int = 0) -> bytes:
    import numpy as np
    from PIL import Image

    image_np = (tensor[index].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image_np).save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_to_image_tensor(data: bytes):
    """Image file bytes -> ComfyUI IMAGE tensor (1,H,W,C) float32 [0,1]."""
    import numpy as np
    import torch
    from PIL import Image

    pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.asarray(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def stack_image_tensors(tensors: list):
    """Stack (1,H,W,C) tensors into one batch, resizing any stragglers to the first image's size."""
    import numpy as np
    import torch
    from PIL import Image

    if not tensors:
        raise CivitaiNodeError("No images returned")
    height, width = tensors[0].shape[1], tensors[0].shape[2]
    resized = []
    for tensor in tensors:
        if tensor.shape[1] == height and tensor.shape[2] == width:
            resized.append(tensor)
            continue
        image_np = (tensor[0].cpu().numpy() * 255.0).astype(np.uint8)
        pil_image = Image.fromarray(image_np).resize((width, height), Image.LANCZOS)
        resized.append(torch.from_numpy(np.asarray(pil_image).astype(np.float32) / 255.0).unsqueeze(0))
    return torch.cat(resized, dim=0)


def bytes_to_video_output(data: bytes, suffix: str = ".mp4"):
    """Video file bytes -> ComfyUI VIDEO (persisted to the Comfy temp dir for the session)."""
    path = os.path.join(comfy_compat.get_temp_dir(), f"civitai_{uuid.uuid4().hex}{suffix}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return comfy_compat.video_from_file(path)


def bytes_to_audio_output(data: bytes) -> dict:
    """Audio file bytes -> ComfyUI AUDIO {"waveform": (1,C,T) float32, "sample_rate": int}."""
    import av
    import numpy as np
    import torch
    from av.audio.resampler import AudioResampler

    frames = []
    sample_rate = 0
    with av.open(io.BytesIO(data)) as container:
        if not container.streams.audio:
            raise CivitaiNodeError("No audio stream found in returned media")
        stream = container.streams.audio[0]
        sample_rate = int(stream.sample_rate or 44100)
        resampler = AudioResampler(format="fltp")
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame) or []:
                frames.append(resampled.to_ndarray())
    if not frames:
        raise CivitaiNodeError("Returned audio stream contained no frames")
    waveform = torch.from_numpy(np.concatenate(frames, axis=1))
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def audio_to_flac_bytes(audio: dict) -> bytes:
    """ComfyUI AUDIO dict -> FLAC file bytes (for uploading as a recipe input)."""
    import av

    waveform = audio["waveform"][0]
    sample_rate = int(audio["sample_rate"])
    layout = "mono" if waveform.shape[0] == 1 else "stereo"
    buffer = io.BytesIO()
    buffer.name = "audio.flac"
    with av.open(buffer, mode="w", format="flac") as container:
        stream = container.add_stream("flac", rate=sample_rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(
            waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format="flt", layout=layout
        )
        frame.sample_rate = sample_rate
        frame.pts = 0
        container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return buffer.getvalue()


def video_to_bytes(video) -> bytes:
    """ComfyUI VIDEO input -> mp4 bytes (for uploading as a recipe input)."""
    buffer = io.BytesIO()
    buffer.name = "video.mp4"
    video.save_to(buffer)
    return buffer.getvalue()
