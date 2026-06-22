from civitai_comfy_nodes import server_routes as sr
from civitai_comfy_nodes import trace_tail


def _wf(steps, **extra):
    return {"id": "6-1", "status": "Succeeded", "createdAt": "2026-06-14T00:00:00Z", "steps": steps, **extra}


def test_flatten_detects_concrete_blobs_without_type_field():
    # The real shape: concrete ImageBlob[]/VideoBlob carry NO `type` discriminator — kind must come
    # from the property name. (This is the bug where only base-`Blob` audio showed up.)
    workflows = [
        _wf(
            [
                {
                    "$type": "imageGen",
                    "output": {
                        "images": [
                            {"id": "b1", "available": True, "url": "u1", "previewUrl": "p1", "width": 512},
                            {"id": "b2", "available": False, "url": "u2"},  # unavailable -> dropped
                            {"id": "b3", "available": True, "blockedReason": "nsfw", "url": "u3"},  # blocked -> dropped
                        ]
                    },
                }
            ],
            cost={"total": 16},
        ),
        _wf([{"$type": "vid", "output": {"video": {"id": "v1", "available": True, "url": "vu"}}}]),
        _wf([{"$type": "echo", "output": {"message": "hi"}}]),  # no blobs -> dropped
    ]
    items = sr.flatten_generations(workflows)
    assert len(items) == 2
    img = items[0]
    assert img["workflowId"] == "6-1" and img["cost"] == 16
    assert [m["blobId"] for m in img["media"]] == ["b1"]
    assert img["media"][0]["kind"] == "image"  # inferred from the "images" property name
    assert items[1]["media"][0]["kind"] == "video"  # inferred from "video"
    assert items[1]["media"][0]["previewUrl"] == "vu"  # falls back to url when no previewUrl


def test_list_generations_requests_mature(monkeypatch):
    # The list API hides R+ blobs by default, which stripped whole mature workflows and partial
    # batches from the user's own history — the gallery must opt out of that.
    captured = {}

    class _FakeClient:
        def query_workflows(self, **kwargs):
            captured.update(kwargs)
            return {"next": None, "items": []}

    monkeypatch.setattr(sr, "_new_client", lambda *a, **k: _FakeClient())
    sr._list_generations(cursor="c1", take=60)
    assert captured["hide_mature"] is False
    assert captured["cursor"] == "c1" and captured["take"] == 60
    assert captured["tags"] is None


def test_scope_tags_mapping(monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_SESSION_ID", "sess-1")
    from civitai_comfy_nodes.config import SOURCE_TAG

    assert sr._scope_tags("session") == [SOURCE_TAG, f"{SOURCE_TAG}:session:sess-1"]
    assert sr._scope_tags("source") == [SOURCE_TAG]
    assert sr._scope_tags("all") is None
    assert sr._scope_tags(None) is None


def test_list_generations_forwards_tags(monkeypatch):
    captured = {}

    class _FakeClient:
        def query_workflows(self, **kwargs):
            captured.update(kwargs)
            return {"next": None, "items": []}

    monkeypatch.setattr(sr, "_new_client", lambda *a, **k: _FakeClient())
    sr._list_generations(cursor=None, take=60, tags=["civitai-comfy-nodes"])
    assert captured["tags"] == ["civitai-comfy-nodes"]


def test_flatten_uses_type_field_when_present():
    # Base-`Blob` outputs (e.g. aceStepAudio) DO carry a polymorphic `type`; trust it over the name.
    workflows = [_wf([{"output": {"blob": {"type": "audio", "id": "au", "available": True, "url": "auu"}}}])]
    assert sr.flatten_generations(workflows)[0]["media"][0]["kind"] == "audio"


def test_flatten_kind_inference_and_filter():
    workflows = [
        _wf(
            [
                {
                    "output": {
                        "images": [{"id": "a", "available": True, "url": "u"}],
                        "video": {"id": "v", "available": True, "url": "vu"},
                        "model": {"id": "m", "available": True, "url": "mu"},
                    }
                }
            ]
        )
    ]
    kinds = {m["kind"] for m in sr.flatten_generations(workflows)[0]["media"]}
    assert kinds == {"image", "video", "model3d"}
    only_video = sr.flatten_generations(workflows, kinds={"video"})
    assert [m["kind"] for m in only_video[0]["media"]] == ["video"]


def test_guess_ext_sniffs_magic_bytes():
    assert sr._guess_ext("image", b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00") == ".png"
    assert sr._guess_ext("image", b"\xff\xd8\xff\xe0\x00\x10JFIF") == ".jpg"
    assert sr._guess_ext("video", b"\x00\x00\x00\x18ftypmp42") == ".mp4"
    assert sr._guess_ext("audio", b"not a known header") == ".flac"  # falls back by kind


def test_workflow_asset_urls_reads_custom_comfy_assets_and_blobs():
    workflow = _wf(
        [
            {
                "$type": "customComfy",
                "output": {
                    "assets": ["http://asset/a.png", {"url": "http://asset/b.png"}],
                    "images": [{"id": "blob1", "available": True, "url": "http://blob/c.png"}],
                },
            }
        ]
    )

    assert sr._workflow_asset_urls(workflow) == [
        "http://asset/a.png",
        "http://asset/b.png",
        "http://blob/c.png",
    ]


def test_workflow_asset_items_infers_media_kind_from_custom_comfy_blob_id():
    workflow = _wf(
        [
            {
                "$type": "customComfy",
                "output": {
                    "blobs": [
                        {"id": "customcomfy-asset-audio.mp3", "available": True, "url": "http://blob/a"},
                        {"id": "customcomfy-asset-video.mp4", "available": True, "url": "http://blob/v"},
                    ]
                },
            }
        ]
    )

    assert sr._workflow_asset_items(workflow) == [
        {"url": "http://blob/a", "kind": "audio"},
        {"url": "http://blob/v", "kind": "video"},
    ]


def test_offload_output_node_ids_detects_save_image_node():
    result = {
        "offload": {
            "workflow": {
                "8": {"class_type": "VAEDecode", "inputs": {}},
                "46": {"class_type": "SaveImage", "inputs": {}},
            }
        }
    }

    assert sr._offload_output_node_ids(result) == ["46"]


def test_publish_local_output_preview_sends_local_executed_event(monkeypatch):
    import sys
    import types

    class FakeServer:
        def __init__(self):
            self.calls = []

        def send_sync(self, event, data, sid=None):
            self.calls.append((event, data, sid))

    fake_server = FakeServer()
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake_server)),
    )

    sr._publish_local_output_preview(
        ["46"],
        [
            {
                "filename": "civitai_offload_abc.png",
                "subfolder": "",
                "type": "output",
                "asset": {"id": "asset-1"},
            }
        ],
        prompt_id="wf-1",
        sid="browser-1",
    )

    assert fake_server.calls == [
        (
            "executed",
            {
                "node": "46",
                "display_node": "46",
                "output": {
                    "images": [{"filename": "civitai_offload_abc.png", "subfolder": "", "type": "output"}]
                },
                "prompt_id": "wf-1",
            },
            "browser-1",
        )
    ]


def test_publish_local_output_preview_uses_audio_key(monkeypatch):
    import sys
    import types

    class FakeServer:
        def __init__(self):
            self.calls = []

        def send_sync(self, event, data, sid=None):
            self.calls.append((event, data, sid))

    fake_server = FakeServer()
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake_server)),
    )

    sr._publish_local_output_preview(
        ["46"],
        [{"filename": "civitai_offload_abc.mp3", "subfolder": "", "type": "output", "kind": "audio"}],
        prompt_id="wf-1",
        sid="browser-1",
    )

    assert fake_server.calls[0][1]["output"] == {
        "audio": [{"filename": "civitai_offload_abc.mp3", "subfolder": "", "type": "output"}]
    }


def test_publish_local_job_history_adds_completed_output_job(monkeypatch):
    import sys
    import threading
    import types

    class FakeQueue:
        def __init__(self):
            self.mutex = threading.RLock()
            self.history = {}

    class FakeServer:
        def __init__(self):
            self.prompt_queue = FakeQueue()
            self.queue_updates = 0

        def queue_updated(self):
            self.queue_updates += 1

    fake_server = FakeServer()
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake_server)),
    )

    sr._publish_local_job_history(
        {"46": {"class_type": "SaveImage", "inputs": {}}},
        ["46"],
        [
            {
                "filename": "civitai_offload_abc.png",
                "subfolder": "",
                "type": "output",
                "asset": {"id": "asset-1"},
            }
        ],
        prompt_id="wf-1",
        workflow_id="wf-1",
    )

    assert fake_server.queue_updates == 1
    history = fake_server.prompt_queue.history["wf-1"]
    assert history["prompt"][1] == "wf-1"
    assert history["prompt"][4] == ["46"]
    assert history["status"]["status_str"] == "success"
    assert history["outputs"] == {
        "46": {"images": [{"filename": "civitai_offload_abc.png", "subfolder": "", "type": "output"}]}
    }


def test_start_trace_tail_returns_none_without_trace_target():
    assert sr._start_trace_tail(object(), {"steps": []}, sid=None) is None


def test_start_trace_tail_waits_for_delayed_trace_url(monkeypatch):
    from civitai_comfy_nodes import client as client_mod

    calls = []

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def get_workflow(self, workflow_id, wait=0):
            calls.append((workflow_id, wait))
            if len(calls) == 1:
                return {"id": workflow_id, "status": "processing", "steps": [{"output": {}}]}
            return {"id": workflow_id, "status": "processing", "steps": [{"output": {"traceUrl": "http://trace"}}]}

    seen = {}

    def fake_tail_trace_to_websocket(url, *, stop_event, sid=None, session=None):
        seen["url"] = url
        seen["sid"] = sid
        seen["stopped"] = stop_event.is_set()
        return trace_tail.TraceTailStats(bytes_in=10, frames=2, emitted=2)

    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(trace_tail, "tail_trace_to_websocket", fake_tail_trace_to_websocket)

    handle = sr._start_trace_tail(object(), {"id": "wf-1", "steps": [{"output": {}}]}, sid="browser-1")
    assert handle is not None
    handle.drain()

    assert calls == [("wf-1", 5), ("wf-1", 5)]
    assert seen == {"url": "http://trace", "sid": "browser-1", "stopped": False}
    assert handle.summary() == {"bytes_in": 10, "frames": 2, "emitted": 2, "errors": 0}


def test_start_trace_tail_keeps_polling_briefly_after_terminal_without_trace(monkeypatch):
    from civitai_comfy_nodes import client as client_mod

    calls = []

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def get_workflow(self, workflow_id, wait=0):
            calls.append((workflow_id, wait))
            if len(calls) == 1:
                return {"id": workflow_id, "status": "succeeded", "steps": [{"output": {}}]}
            return {"id": workflow_id, "status": "succeeded", "steps": [{"output": {"traceUrl": "http://trace"}}]}

    seen = {}

    def fake_tail_trace_to_websocket(url, *, stop_event, sid=None, session=None):
        seen["url"] = url
        seen["sid"] = sid
        return trace_tail.TraceTailStats(bytes_in=5, frames=1, emitted=1)

    monkeypatch.setattr(sr, "TRACE_URL_POLL_DELAY_SECONDS", 0.0)
    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(trace_tail, "tail_trace_to_websocket", fake_tail_trace_to_websocket)

    handle = sr._start_trace_tail(object(), {"id": "wf-1", "steps": [{"output": {}}]}, sid="browser-1")
    assert handle is not None
    handle.drain()

    assert calls == [("wf-1", 5), ("wf-1", 5)]
    assert seen == {"url": "http://trace", "sid": "browser-1"}
    assert handle.summary() == {"bytes_in": 5, "frames": 1, "emitted": 1, "errors": 0}


def _usage_step(usage):
    return {"$type": "customComfy", "output": {"usage": usage}}


def test_extract_usage_reads_first_step_usage():
    wf = _wf([{"$type": "customComfy", "output": {}}, _usage_step({"buzzPerSecond": 1})])
    assert sr._extract_usage(wf) == {"buzzPerSecond": 1}
    assert sr._extract_usage(_wf([{"$type": "x", "output": {}}])) is None


def test_buzz_message_running_carries_rate_and_anchor():
    wf = {
        "id": "6-9",
        "status": "Processing",
        "steps": [
            _usage_step(
                {
                    "buzzPerSecond": 1,
                    "runtimeSeconds": 3.48,
                    "estimatedCost": 4,
                    "startedAt": "2026-06-22T02:14:00Z",
                    "computedAt": "2026-06-22T02:14:03Z",
                }
            )
        ],
    }
    msg = sr._buzz_message(wf)
    assert msg["prompt_id"] == "6-9"
    assert msg["status"] == "processing"
    assert msg["terminal"] is False
    assert msg["buzz_per_second"] == 1
    assert msg["runtime_seconds"] == 3.48
    assert msg["estimated_cost"] == 4
    assert msg["started_at"] == 1782094440000
    assert msg["computed_at"] == 1782094443000


def test_buzz_message_terminal_prefers_settled_cost_total():
    wf = {
        "id": "6-9",
        "status": "Succeeded",
        "cost": {"total": 12},
        "steps": [_usage_step({"buzzPerSecond": 1, "estimatedCost": 11})],
    }
    msg = sr._buzz_message(wf)
    assert msg["terminal"] is True
    assert msg["estimated_cost"] == 12  # settled charge wins over the last live estimate


def test_buzz_message_none_without_usage_or_cost():
    assert sr._buzz_message({"status": "processing", "steps": [{"output": {}}]}) is None


def test_send_buzz_pushes_frame_to_sid(monkeypatch):
    import sys
    import types

    class FakeServer:
        def __init__(self):
            self.calls = []

        def send_sync(self, event, data, sid=None):
            self.calls.append((event, data, sid))

    fake_server = FakeServer()
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake_server)),
    )

    wf = {"id": "6-9", "status": "Processing", "steps": [_usage_step({"buzzPerSecond": 1})]}
    sr._send_buzz("browser-1", wf)
    assert len(fake_server.calls) == 1
    event, data, sid = fake_server.calls[0]
    assert event == "civitai.buzz" and sid == "browser-1" and data["buzz_per_second"] == 1

    # No sid, or no usage/cost → no push.
    sr._send_buzz(None, wf)
    sr._send_buzz("browser-1", {"status": "processing", "steps": [{"output": {}}]})
    assert len(fake_server.calls) == 1


def test_buzz_message_terminal_includes_per_wallet_transactions():
    wf = {
        "id": "6-9",
        "status": "Succeeded",
        "cost": {"total": 8},
        "steps": [_usage_step({"buzzPerSecond": 1})],
        "transactions": {
            "list": [
                {"amount": 11, "accountType": "blue", "type": "debit"},
                {"amount": 5, "accountType": "green", "type": "credit"},
                {"amount": 2, "accountType": "weird"},
            ]
        },
    }
    msg = sr._buzz_message(wf)
    assert msg["cost_total"] == 8
    assert msg["transactions"] == [
        {"amount": 11, "currency": "Blue", "refund": False},
        {"amount": 5, "currency": "Green", "refund": True},
        {"amount": 2, "currency": "weird", "refund": False},
    ]


def test_buzz_message_running_has_no_transactions():
    wf = {"id": "6-9", "status": "Processing", "steps": [_usage_step({"buzzPerSecond": 1})]}
    msg = sr._buzz_message(wf)
    assert "transactions" not in msg and "cost_total" not in msg
