"""Tail a customComfy *binary trace* and replay it over the local ComfyUI websocket.

When an offloaded customComfy job runs with ``trace: "binary"``, the orchestrator records the
remote ComfyUI ``/ws`` conversation (execution events, image previews, console logs) to a
streaming blob that any client can tail live over an anonymous GET. This module reads that blob
as it grows and re-emits each frame through the *local* ComfyUI server's websocket, so the
user's canvas shows the remote run's progress bars and previews on the offloaded nodes (which
keep their original ids in the submitted graph).

Frame format (big-endian; produced by spine-controller ``TraceFraming``):

    [timestamp:8 int64 ms][direction:1][opcode:1][payload_length:4 uint32][payload]

``opcode 0x1`` = text: a UTF-8 JSON ``/ws`` frame, e.g. ``{"type": ..., "data": ...}``.
``opcode 0x2`` = binary: raw preview image bytes — the worker already stripped ComfyUI's
8-byte preview header, so we re-prepend it before replaying.

Everything here is best-effort: any network/parse/emit error is logged and swallowed. Tailing
must never break the offload it observes.
"""

import copy
import json
import logging
import struct
import threading
from dataclasses import dataclass

import requests

_log = logging.getLogger("civitai_comfy_nodes.trace_tail")

HEADER_SIZE = 14
OPCODE_TEXT = 0x1
OPCODE_BINARY = 0x2

# ComfyUI binary-websocket framing for image previews.
_PREVIEW_IMAGE_EVENT = 1  # server.BinaryEventTypes.PREVIEW_IMAGE
_IMAGE_FORMAT_PNG = 2  # the worker captures SaveImageWebsocket output as image/png

# ``/ws`` event types worth replaying onto the local canvas. ``status`` is intentionally
# excluded so the remote queue count doesn't overwrite the local queue indicator.
FORWARDED_EVENTS = frozenset(
    {
        "execution_start",
        "execution_cached",
        "executing",
        "executed",
        "progress",
        "progress_state",
        "execution_success",
        "execution_error",
        "execution_interrupted",
        "logs",
    }
)

REMOTE_FILE_TYPES = {"input", "output", "temp"}


@dataclass
class TraceTailStats:
    bytes_in: int = 0
    frames: int = 0
    emitted: int = 0
    errors: int = 0

    def as_dict(self) -> dict:
        return {"bytes_in": self.bytes_in, "frames": self.frames, "emitted": self.emitted, "errors": self.errors}


def parse_frames(buffer: bytearray) -> tuple[list[tuple[int, int, int, bytes]], bytearray]:
    """Pull every complete frame out of ``buffer``.

    Returns ``(frames, remainder)`` where ``frames`` is a list of
    ``(timestamp_ms, direction, opcode, payload)`` and ``remainder`` is the trailing bytes of a
    partial frame to carry into the next read (a frame can be split across stream chunks)."""
    frames: list[tuple[int, int, int, bytes]] = []
    offset = 0
    total = len(buffer)
    while total - offset >= HEADER_SIZE:
        payload_len = int.from_bytes(buffer[offset + 10 : offset + 14], "big")
        frame_end = offset + HEADER_SIZE + payload_len
        if frame_end > total:
            break
        timestamp = int.from_bytes(buffer[offset : offset + 8], "big")
        direction = buffer[offset + 8]
        opcode = buffer[offset + 9]
        payload = bytes(buffer[offset + HEADER_SIZE : frame_end])
        frames.append((timestamp, direction, opcode, payload))
        offset = frame_end
    return frames, buffer[offset:]


def _load_comfy_server():
    """Resolve the running ComfyUI server + preview event id. Raises if ComfyUI isn't present."""
    from server import PromptServer  # noqa: PLC0415  (only available inside ComfyUI)

    preview_event = _PREVIEW_IMAGE_EVENT
    try:
        from server import BinaryEventTypes  # noqa: PLC0415

        preview_event = BinaryEventTypes.PREVIEW_IMAGE
    except Exception:
        pass
    return PromptServer.instance, preview_event


def _strip_remote_file_refs(value):
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str) and value.get("type") in REMOTE_FILE_TYPES:
            return None
        cleaned = {}
        for key, child in value.items():
            stripped = _strip_remote_file_refs(child)
            if stripped is None:
                continue
            if stripped == [] or stripped == {}:
                continue
            cleaned[key] = stripped
        return cleaned
    if isinstance(value, list):
        cleaned = []
        for child in value:
            stripped = _strip_remote_file_refs(child)
            if stripped is None:
                continue
            if stripped == [] or stripped == {}:
                continue
            cleaned.append(stripped)
        return cleaned
    return value


def sanitize_text_message(message: dict) -> dict:
    """Drop remote Comfy file references before replaying events into the local UI.

    The worker may emit `executed` output refs such as `ComfyUI_00010_.png`. Those filenames live
    in the worker's output directory, but the local frontend resolves them through the user's local
    `/view` route. If the user has an old local file with the same name, the canvas shows that stale
    image instead of the offloaded result.
    """
    if message.get("type") != "executed":
        return message
    data = message.get("data")
    if not isinstance(data, dict):
        return message
    output = data.get("output")
    if not isinstance(output, dict):
        return message
    sanitized = copy.deepcopy(message)
    sanitized_data = sanitized.get("data") or {}
    sanitized_data["output"] = _strip_remote_file_refs(sanitized_data.get("output") or {}) or {}
    return sanitized


def emit_frame(server, preview_event: int, opcode: int, payload: bytes, sid: str | None) -> bool:
    """Replay one trace frame onto the local websocket. Returns True if a message was sent."""
    if opcode == OPCODE_TEXT:
        try:
            message = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return False
        if not isinstance(message, dict):
            return False
        event = message.get("type")
        if event not in FORWARDED_EVENTS:
            return False
        message = sanitize_text_message(message)
        server.send_sync(event, message.get("data"), sid)
        return True
    if opcode == OPCODE_BINARY:
        if not payload:
            return False
        # send_sync routes a bytes payload through send_bytes, which prepends the 4-byte event
        # id. We prepend the 4-byte image-format id so the frontend gets ComfyUI's full 8-byte
        # preview header followed by the PNG bytes.
        server.send_sync(preview_event, struct.pack(">I", _IMAGE_FORMAT_PNG) + payload, sid)
        return True
    return False


def tail_trace_to_websocket(
    trace_url: str,
    *,
    stop_event: threading.Event,
    sid: str | None = None,
    session: requests.Session | None = None,
) -> TraceTailStats:
    """Tail ``trace_url`` and replay its frames onto the local ComfyUI websocket until the stream
    ends (the provider finished the upload) or ``stop_event`` is set. Best-effort; never raises."""
    stats = TraceTailStats()
    try:
        server, preview_event = _load_comfy_server()
    except Exception as exc:
        _log.warning("trace tail: ComfyUI server unavailable, skipping (%s)", exc)
        return stats

    get = (session or requests).get
    backoff = 0.5
    buffer = bytearray()
    while not stop_event.is_set():
        try:
            response = get(trace_url, stream=True, timeout=(10, None))
        except requests.RequestException as exc:
            _log.debug("trace tail: connect failed (%s)", exc)
            if stop_event.wait(backoff):
                break
            backoff = min(backoff * 2, 5.0)
            continue

        with response:
            # The blob 404s until the worker produces the first byte; retry until the job starts.
            if response.status_code == 404:
                if stop_event.wait(backoff):
                    break
                backoff = min(backoff * 2, 5.0)
                continue
            if response.status_code >= 400:
                _log.warning("trace tail: GET %s -> HTTP %s", trace_url, response.status_code)
                break
            try:
                for chunk in response.iter_content(chunk_size=65536):
                    if stop_event.is_set():
                        break
                    if not chunk:
                        continue
                    stats.bytes_in += len(chunk)
                    buffer.extend(chunk)
                    frames, buffer = parse_frames(buffer)
                    for _ts, _direction, opcode, payload in frames:
                        stats.frames += 1
                        try:
                            if emit_frame(server, preview_event, opcode, payload, sid):
                                stats.emitted += 1
                        except Exception as exc:
                            stats.errors += 1
                            _log.debug("trace tail: emit failed (%s)", exc)
            except requests.RequestException as exc:
                _log.debug("trace tail: stream interrupted (%s)", exc)
        # Clean EOF means the upload completed — don't reconnect (a fresh GET would replay the
        # whole blob from the start and double-emit). Mid-stream drops also exit here on purpose.
        break

    _log.debug(
        "trace tail finished: %s bytes, %s frames, %s emitted, %s errors",
        stats.bytes_in,
        stats.frames,
        stats.emitted,
        stats.errors,
    )
    return stats
