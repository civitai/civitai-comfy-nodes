import json
import struct
import threading

from civitai_comfy_nodes import trace_tail


# Mirror spine-controller's TraceFraming so the parser is tested against the real wire format:
# [timestamp:8 BE][direction:1][opcode:1][payload_length:4 BE][payload].
def _encode(timestamp, direction, opcode, payload):
    return (
        timestamp.to_bytes(8, "big")
        + bytes([direction, opcode])
        + len(payload).to_bytes(4, "big")
        + bytes(payload)
    )


def _text_frame(obj):
    return _encode(1, 0, trace_tail.OPCODE_TEXT, json.dumps(obj).encode("utf-8"))


def _binary_frame(data):
    return _encode(2, 0, trace_tail.OPCODE_BINARY, data)


class _RecordingServer:
    def __init__(self):
        self.calls = []

    def send_sync(self, event, data, sid=None):
        self.calls.append((event, data, sid))


class _FakeResponse:
    def __init__(self, status_code, chunks=(), on_enter=None):
        self.status_code = status_code
        self._chunks = list(chunks)
        self._on_enter = on_enter
        self.closed = False

    def __enter__(self):
        if self._on_enter:
            self._on_enter()
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def iter_content(self, chunk_size=65536):
        yield from self._chunks

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = 0

    def get(self, url, stream=False, timeout=None):
        self.requests += 1
        return self._responses.pop(0)


def test_parse_frames_reassembles_payload_split_across_chunks():
    blob = _text_frame({"type": "progress", "data": {"value": 1}}) + _binary_frame(b"PNGDATA")
    cut = len(blob) - 3  # slice through the middle of the second frame's payload

    frames, remainder = trace_tail.parse_frames(bytearray(blob[:cut]))
    assert len(frames) == 1
    assert frames[0][2] == trace_tail.OPCODE_TEXT

    frames2, rest = trace_tail.parse_frames(remainder + blob[cut:])
    assert len(frames2) == 1
    assert frames2[0][2] == trace_tail.OPCODE_BINARY
    assert frames2[0][3] == b"PNGDATA"
    assert bytes(rest) == b""


def test_emit_frame_forwards_only_allowlisted_text_events():
    server = _RecordingServer()
    sent = trace_tail.emit_frame(
        server, 1, trace_tail.OPCODE_TEXT,
        json.dumps({"type": "progress", "data": {"value": 3, "max": 10}}).encode("utf-8"), "sid1",
    )
    skipped = trace_tail.emit_frame(
        server, 1, trace_tail.OPCODE_TEXT,
        json.dumps({"type": "status", "data": {"exec_info": {}}}).encode("utf-8"), "sid1",
    )

    assert sent is True
    assert skipped is False
    assert server.calls == [("progress", {"value": 3, "max": 10}, "sid1")]


def test_emit_frame_strips_remote_file_refs_from_executed_events():
    server = _RecordingServer()
    payload = {
        "type": "executed",
        "data": {
            "node": "46",
            "output": {
                "images": [
                    {"filename": "ComfyUI_00010_.png", "subfolder": "", "type": "output"},
                ],
                "text": ["done"],
            },
        },
    }

    assert trace_tail.emit_frame(
        server,
        1,
        trace_tail.OPCODE_TEXT,
        json.dumps(payload).encode("utf-8"),
        "sid1",
    ) is True

    assert server.calls == [
        (
            "executed",
            {"node": "46", "output": {"text": ["done"]}},
            "sid1",
        )
    ]


def test_emit_frame_binary_prepends_png_format_header():
    server = _RecordingServer()
    image = b"\x89PNG\r\n\x1a\n rest of png"

    assert trace_tail.emit_frame(server, 7, trace_tail.OPCODE_BINARY, image, "sid2") is True

    event, data, sid = server.calls[0]
    assert event == 7  # the resolved PREVIEW_IMAGE event id is passed straight through
    assert data[:4] == struct.pack(">I", 2)  # inner image-format id = PNG
    assert data[4:] == image
    assert sid == "sid2"


def test_tail_replays_stream_frames(monkeypatch):
    server = _RecordingServer()
    monkeypatch.setattr(trace_tail, "_load_comfy_server", lambda: (server, 1))
    blob = (
        _text_frame({"type": "execution_start", "data": {"prompt_id": "p"}})
        + _binary_frame(b"IMG")
        + _text_frame({"type": "status", "data": {}})  # filtered out
        + _text_frame({"type": "progress", "data": {"value": 5}})
    )
    mid = len(blob) // 2  # force a chunk boundary mid-frame
    session = _FakeSession([_FakeResponse(200, chunks=[blob[:mid], blob[mid:]])])

    stats = trace_tail.tail_trace_to_websocket(
        "http://x/trace.bin", stop_event=threading.Event(), sid="s", session=session
    )

    assert stats.frames == 4
    assert stats.emitted == 3
    assert [event for event, _data, _sid in server.calls] == ["execution_start", 1, "progress"]
    assert session.requests == 1  # clean EOF -> no reconnect (would replay the whole blob)


def test_tail_noops_without_comfy_server(monkeypatch):
    def _boom():
        raise RuntimeError("server module not importable outside ComfyUI")

    monkeypatch.setattr(trace_tail, "_load_comfy_server", _boom)

    stats = trace_tail.tail_trace_to_websocket("http://x", stop_event=threading.Event())

    assert stats.frames == 0
    assert stats.emitted == 0


def test_tail_retries_on_404_until_stopped(monkeypatch):
    server = _RecordingServer()
    monkeypatch.setattr(trace_tail, "_load_comfy_server", lambda: (server, 1))
    stop = threading.Event()
    # Setting the stop flag as the 404 response is entered makes the post-404 backoff return
    # immediately, so the tailer exits without sleeping or emitting.
    session = _FakeSession([_FakeResponse(404, on_enter=stop.set)])

    stats = trace_tail.tail_trace_to_websocket("http://x", stop_event=stop, session=session)

    assert stats.frames == 0
    assert server.calls == []
