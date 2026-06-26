import pytest

from civitai_comfy_nodes import server_routes as sr
from civitai_comfy_nodes import trace_tail
from civitai_comfy_nodes.errors import CivitaiNodeError


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
                "output": {"images": [{"filename": "civitai_offload_abc.png", "subfolder": "", "type": "output"}]},
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


@pytest.fixture()
def settings_store(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_ORCHESTRATION_URL", raising=False)
    monkeypatch.setenv("CIVITAI_COMFY_SETTINGS_STORE", str(tmp_path / "settings.json"))
    return tmp_path


def test_pack_config_payload_defaults(settings_store):
    payload = sr._pack_config_payload()
    assert payload["orchestratorUrl"] == ""
    assert payload["orchestratorSource"] == "default"
    assert payload["minVramGb"] is None
    assert payload["allowMatureContent"] == "auto"
    assert payload["useSageAttention"] is True  # Sage Attention defaults on
    assert payload["gpuGeneration"] == "Ada"
    assert payload["vramTiers"] == [24]
    assert payload["enableOffload"] is True
    assert payload["enableRecipeNodes"] is True


def test_apply_pack_config_update_feature_toggles(settings_store):
    sr._apply_pack_config_update({"enableOffload": False, "enableRecipeNodes": False})
    payload = sr._pack_config_payload()
    assert payload["enableOffload"] is False
    assert payload["enableRecipeNodes"] is False
    sr._apply_pack_config_update({"enableOffload": True})
    assert sr._pack_config_payload()["enableOffload"] is True
    assert sr._pack_config_payload()["enableRecipeNodes"] is False  # untouched by the second patch


def test_apply_pack_config_update_round_trip(settings_store):
    sr._apply_pack_config_update(
        {"orchestratorUrl": "http://dev/", "minVramGb": 24, "allowMatureContent": "true", "useSageAttention": True}
    )
    payload = sr._pack_config_payload()
    assert payload["orchestratorUrl"] == "http://dev"  # trailing slash stripped
    assert payload["orchestratorSource"] == "stored"
    assert payload["minVramGb"] == 24
    assert payload["allowMatureContent"] == "true"
    assert payload["useSageAttention"] is True


def test_apply_pack_config_update_blank_url_clears_override(settings_store):
    sr._apply_pack_config_update({"orchestratorUrl": "http://dev"})
    sr._apply_pack_config_update({"orchestratorUrl": "  "})
    assert sr._pack_config_payload()["orchestratorSource"] == "default"


def test_apply_pack_config_update_rejects_bad_input(settings_store):
    with pytest.raises(ValueError):
        sr._apply_pack_config_update({"orchestratorUrl": "ftp://nope"})
    with pytest.raises(ValueError):
        sr._apply_pack_config_update({"minVramGb": 999})
    with pytest.raises(ValueError):
        sr._apply_pack_config_update({"allowMatureContent": "maybe"})


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

    def fake_tail_trace_to_websocket(url, *, stop_event, sid=None, session=None, prompt_id=None):
        seen["url"] = url
        seen["sid"] = sid
        seen["prompt_id"] = prompt_id
        seen["stopped"] = stop_event.is_set()
        return trace_tail.TraceTailStats(bytes_in=10, frames=2, emitted=2)

    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(trace_tail, "tail_trace_to_websocket", fake_tail_trace_to_websocket)

    handle = sr._start_trace_tail(object(), {"id": "wf-1", "steps": [{"output": {}}]}, sid="browser-1")
    assert handle is not None
    handle.drain()

    assert calls == [("wf-1", 5), ("wf-1", 5)]
    assert seen == {"url": "http://trace", "sid": "browser-1", "prompt_id": "wf-1", "stopped": False}
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

    def fake_tail_trace_to_websocket(url, *, stop_event, sid=None, session=None, prompt_id=None):
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


def test_push_offload_status_sends_custom_ws_event(monkeypatch):
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

    sr._push_offload_status("browser-1", "done", workflowId="wf-1", promptId="p-9")

    assert fake_server.calls == [
        (
            "civitai.offload.status",
            {"state": "done", "workflowId": "wf-1", "promptId": "p-9"},
            "browser-1",
        )
    ]


def test_push_offload_status_is_noop_without_comfy_server():
    # No `server` module installed in the test env -> import fails -> silent no-op.
    sr._push_offload_status("browser-1", "error", message="boom")


def test_offload_submit_uses_wait_zero_and_requests_trace(monkeypatch):
    import types

    from civitai_comfy_nodes import client as client_mod
    from civitai_comfy_nodes import config as config_mod
    from civitai_comfy_nodes import offload as offload_mod

    captured = {}

    class FakeClient:
        def __init__(self, config):
            self.config = config
            self.upload_blob_file = lambda *a, **k: None

        def submit_steps(self, steps, *, wait, whatif=False):
            captured["wait"] = wait
            captured["whatif"] = whatif
            captured["steps"] = steps
            return {"id": "wf-1", "status": "queued"}

    fake_build = types.SimpleNamespace(steps=[{"$type": "customComfy", "input": {}}], as_dict=lambda: {"ok": True})

    def fake_build_offload(prompt, **kwargs):
        captured["trace"] = kwargs.get("trace")
        return fake_build

    monkeypatch.setattr(
        config_mod, "resolve_config", lambda interactive=False: types.SimpleNamespace(token="t", timeout_minutes=5)
    )
    monkeypatch.setattr(config_mod, "stored_min_vram_gb", lambda: 24)
    monkeypatch.setattr(config_mod, "stored_use_sage_attention", lambda: False)
    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(offload_mod, "build_custom_comfy_offload", fake_build_offload)

    result = sr._offload_submit({"3": {}}, None, None, whatif=False, do_tail=True)

    assert captured["wait"] == 0
    assert captured["trace"] == "binary"
    assert result["workflow"] == {"id": "wf-1", "status": "queued"}
    assert result["build"] is fake_build
    assert result["config"].token == "t"


def test_offload_submit_omits_trace_when_not_tailing(monkeypatch):
    import types

    from civitai_comfy_nodes import client as client_mod
    from civitai_comfy_nodes import config as config_mod
    from civitai_comfy_nodes import offload as offload_mod

    captured = {}

    class FakeClient:
        def __init__(self, config):
            self.upload_blob_file = lambda *a, **k: None

        def submit_steps(self, steps, *, wait, whatif=False):
            captured["whatif"] = whatif
            return {"id": "wf-2"}

    def fake_build_offload(prompt, **kwargs):
        captured["trace"] = kwargs.get("trace")
        return types.SimpleNamespace(steps=[], as_dict=lambda: {})

    monkeypatch.setattr(
        config_mod, "resolve_config", lambda interactive=False: types.SimpleNamespace(token="t", timeout_minutes=5)
    )
    monkeypatch.setattr(config_mod, "stored_min_vram_gb", lambda: 24)
    monkeypatch.setattr(config_mod, "stored_use_sage_attention", lambda: False)
    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(offload_mod, "build_custom_comfy_offload", fake_build_offload)

    sr._offload_submit({"3": {}}, None, None, whatif=True, do_tail=False)

    assert captured["trace"] is None
    assert captured["whatif"] is True


def _finalize_env(monkeypatch, events, *, poll, local):
    import types

    from civitai_comfy_nodes import client as client_mod

    monkeypatch.setattr(client_mod, "OrchestrationClient", lambda config: object())

    fake_tail = types.SimpleNamespace(
        drain=lambda: events.append("drain"),
        stop=lambda: events.append("stop"),
        summary=lambda: None,
    )
    monkeypatch.setattr(
        sr, "_start_trace_tail", lambda config, wf, sid=None, prompt_id=None: (events.append("tail"), fake_tail)[1]
    )
    monkeypatch.setattr(sr, "_poll_workflow_to_terminal", poll)
    monkeypatch.setattr(sr, "_run_local_tail", local)
    monkeypatch.setattr(sr, "_push_offload_status", lambda sid, state, **f: events.append(("status", state, sid, f)))


def test_offload_finalize_pushes_done_on_success(monkeypatch):
    import types

    events = []
    _finalize_env(
        monkeypatch,
        events,
        poll=lambda client, wf, timeout, on_update=None: {"id": "wf-1", "status": "succeeded"},
        local=lambda prompt, result, base, client_id=None, running_task_id=None: {"queue": {"prompt_id": "p-9"}},
    )
    build = types.SimpleNamespace(as_dict=lambda: {"k": "v"})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize(
        {"p": 1}, build, config, {"id": "wf-1"}, "http://localhost:8188", sid="browser-1", do_tail=True
    )

    assert ("status", "done", "browser-1", {"workflowId": "wf-1", "promptId": "p-9"}) in events
    assert "drain" in events
    assert "stop" not in events


def test_offload_finalize_done_includes_cost_when_present(monkeypatch):
    import types

    events = []
    final = {
        "id": "wf-1",
        "status": "succeeded",
        "cost": {"total": 1234},
        "transactions": {"list": [{"amount": 1234, "accountType": "yellow", "type": "debit"}]},
    }
    _finalize_env(
        monkeypatch,
        events,
        poll=lambda client, wf, timeout, on_update=None: final,
        local=lambda prompt, result, base, client_id=None, running_task_id=None: {"queue": {"prompt_id": "p-9"}},
    )
    build = types.SimpleNamespace(as_dict=lambda: {"k": "v"})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize({"p": 1}, build, config, {"id": "wf-1"}, "http://x", sid="browser-1", do_tail=True)

    done = next(e[3] for e in events if isinstance(e, tuple) and e[0] == "status" and e[1] == "done")
    assert done["costTotal"] == 1234
    assert done["transactions"] == [{"amount": 1234, "currency": "Yellow", "refund": False}]


def test_offload_finalize_pushes_error_and_stops_tail_when_poll_fails(monkeypatch):
    import types

    events = []

    def boom(client, wf, timeout, on_update=None):
        raise CivitaiNodeError("poll boom")

    _finalize_env(
        monkeypatch,
        events,
        poll=boom,
        local=lambda *a, **k: {"queue": {"prompt_id": "p-9"}},
    )
    # (local isn't reached on a poll failure)
    build = types.SimpleNamespace(as_dict=lambda: {})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize({"p": 1}, build, config, {"id": "wf-1"}, "http://x", sid="browser-1", do_tail=True)

    assert ("status", "error", "browser-1", {"message": "poll boom"}) in events
    assert "stop" in events
    assert "drain" not in events


def test_offload_finalize_pushes_error_when_local_tail_fails(monkeypatch):
    import types

    events = []

    def boom_local(prompt, result, base, client_id=None, running_task_id=None):
        raise CivitaiNodeError("no assets")

    _finalize_env(
        monkeypatch,
        events,
        poll=lambda client, wf, timeout, on_update=None: {"id": "wf-1", "status": "succeeded"},
        local=boom_local,
    )
    build = types.SimpleNamespace(as_dict=lambda: {})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize({"p": 1}, build, config, {"id": "wf-1"}, "http://x", sid="browser-1", do_tail=True)

    assert ("status", "error", "browser-1", {"message": "no assets"}) in events
    assert "drain" in events  # poll succeeded, tail drained, then local failed


# ── Native queue / lifecycle bridge ──────────────────────────────────────────────────────────────


def _install_fake_server(monkeypatch, *, with_queue=False):
    import sys
    import threading
    import types

    class FakeQueue:
        def __init__(self):
            self.mutex = threading.RLock()
            self.history = {}
            self.currently_running = {}

    class FakeServer:
        def __init__(self):
            self.calls = []
            self.queue_updates = 0
            if with_queue:
                self.prompt_queue = FakeQueue()

        def send_sync(self, event, data, sid=None):
            self.calls.append((event, data, sid))

        def queue_updated(self):
            self.queue_updates += 1

    fake = FakeServer()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake)))
    return fake


def test_inject_running_queue_adds_entry_then_removes_it(monkeypatch):
    fake = _install_fake_server(monkeypatch, with_queue=True)

    task_id = sr._inject_running_queue(
        {"46": {"class_type": "SaveImage", "inputs": {}}}, ["46"], prompt_id="wf-1", workflow_id="wf-1"
    )
    assert task_id is not None
    item = fake.prompt_queue.currently_running[task_id]
    assert item[1] == "wf-1"  # prompt_id, what /api/jobs reports as the running job id
    assert item[4] == ["46"]  # output node ids
    assert item[3]["extra_pnginfo"]["workflow"] == {"id": "wf-1", "source": "civitai_offload"}
    assert fake.queue_updates == 1

    sr._remove_running_queue(task_id)
    assert task_id not in fake.prompt_queue.currently_running
    assert fake.queue_updates == 2


def test_inject_running_queue_noop_without_prompt_id(monkeypatch):
    fake = _install_fake_server(monkeypatch, with_queue=True)
    assert sr._inject_running_queue({}, [], prompt_id="", workflow_id=None) is None
    assert fake.prompt_queue.currently_running == {}


def test_emit_lifecycle_transition_emits_native_start_and_success(monkeypatch):
    fake = _install_fake_server(monkeypatch)
    sr._emit_lifecycle_transition("sid", "wf-1", "scheduled", "processing")
    sr._emit_lifecycle_transition("sid", "wf-1", "processing", "succeeded")
    assert [c[0] for c in fake.calls] == ["execution_start", "executing", "executing", "execution_success"]
    assert all(c[1]["prompt_id"] == "wf-1" and c[2] == "sid" for c in fake.calls)


def test_emit_lifecycle_transition_failed_and_canceled(monkeypatch):
    fake = _install_fake_server(monkeypatch)
    sr._emit_lifecycle_transition("sid", "wf-1", "processing", "failed")
    sr._emit_lifecycle_transition("sid", "wf-2", "processing", "canceled")
    assert fake.calls[0][0] == "execution_error" and fake.calls[0][1]["prompt_id"] == "wf-1"
    assert fake.calls[1][0] == "execution_interrupted" and fake.calls[1][1]["prompt_id"] == "wf-2"


def test_emit_lifecycle_transition_noop_on_same_status_or_no_sid(monkeypatch):
    fake = _install_fake_server(monkeypatch)
    sr._emit_lifecycle_transition("sid", "wf-1", "processing", "processing")
    sr._emit_lifecycle_transition(None, "wf-1", "scheduled", "processing")
    assert fake.calls == []


def test_emit_progress_maps_max_estimated_rate(monkeypatch):
    fake = _install_fake_server(monkeypatch)
    wf = {"steps": [{"jobs": [{"estimatedProgressRate": 0.25}, {"estimatedProgressRate": 0.6}]}]}
    sr._emit_progress("sid", "wf-1", wf)
    assert fake.calls == [("progress", {"value": 600, "max": 1000, "prompt_id": "wf-1", "node": None}, "sid")]
    fake.calls.clear()
    sr._emit_progress("sid", "wf-1", {"steps": [{"jobs": []}]})  # no rate -> no frame
    assert fake.calls == []


def test_publish_failed_job_history_writes_error_entry(monkeypatch):
    fake = _install_fake_server(monkeypatch, with_queue=True)
    sr._publish_failed_job_history({"46": {}}, ["46"], prompt_id="wf-1", workflow_id="wf-1", message="boom")
    entry = fake.prompt_queue.history["wf-1"]
    assert entry["status"]["status_str"] == "error"
    assert entry["status"]["completed"] is False
    assert entry["outputs"] == {}
    assert entry["prompt"][1] == "wf-1"
    assert fake.queue_updates == 1


def test_cancel_offload_cancels_workflow_and_removes_running_row(monkeypatch):
    from civitai_comfy_nodes import client as client_mod
    from civitai_comfy_nodes import config as config_mod

    fake = _install_fake_server(monkeypatch, with_queue=True)
    canceled = []

    class FakeClient:
        def __init__(self, config):
            pass

        def cancel_workflow(self, workflow_id):
            canceled.append(workflow_id)

    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(config_mod, "resolve_config", lambda interactive=False: object())

    task_id = sr._inject_running_queue({"3": {}}, [], prompt_id="wf-9", workflow_id="wf-9")
    with sr._running_lock:
        sr._active_offloads["wf-9"] = {"task_id": task_id, "sid": "browser-1"}
    try:
        sr._cancel_offload("wf-9")
    finally:
        sr._active_offloads.pop("wf-9", None)

    assert canceled == ["wf-9"]
    assert task_id not in fake.prompt_queue.currently_running
    interrupts = [c for c in fake.calls if c[0] == "execution_interrupted"]
    assert interrupts and interrupts[0][1]["prompt_id"] == "wf-9" and interrupts[0][2] == "browser-1"


def test_offload_finalize_shows_running_row_then_clears_it(monkeypatch):
    import types

    fake = _install_fake_server(monkeypatch, with_queue=True)
    events = []

    def local(prompt, result, base, client_id=None, running_task_id=None):
        # The row is live while the offload runs, and the local tail swaps it for completed history.
        assert running_task_id in fake.prompt_queue.currently_running
        sr._remove_running_queue(running_task_id)
        return {"queue": {"prompt_id": "p-9"}}

    _finalize_env(
        monkeypatch,
        events,
        poll=lambda client, wf, timeout, on_update=None: {"id": "wf-1", "status": "succeeded"},
        local=local,
    )
    build = types.SimpleNamespace(as_dict=lambda: {"k": "v"})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize({"p": 1}, build, config, {"id": "wf-1"}, "http://x", sid="browser-1", do_tail=True)

    assert fake.prompt_queue.currently_running == {}  # cleaned up, no phantom running job
    assert ("status", "done", "browser-1", {"workflowId": "wf-1", "promptId": "p-9"}) in events
    assert "wf-1" not in sr._active_offloads
