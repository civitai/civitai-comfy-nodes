from civitai_comfy_nodes import server_routes as sr


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


def test_flatten_infers_kind_from_blob_id_extension():
    # customComfy outputs land under generic `blobs`/`tempBlobs` keys with no `type` field; the
    # asset blob id keeps the original output filename, so its extension is the only kind signal.
    workflows = [
        _wf(
            [
                {
                    "$type": "customComfy",
                    "output": {
                        "blobs": [
                            {
                                "id": "customcomfy-X-asset-audio-stable_audio_3_00063.mp3",
                                "available": True,
                                "url": "u1",
                            },
                            {"id": "customcomfy-X-asset-anim_00001.webm", "available": True, "url": "u2"},
                            {"id": "customcomfy-X-asset-ComfyUI_00001.png", "available": True, "url": "u3"},
                        ],
                        "tempBlobs": [
                            {"id": "customcomfy-X-asset-ComfyUI_temp_nxjwl_00001.flac", "available": True, "url": "u4"},
                        ],
                    },
                }
            ]
        )
    ]
    assert [m["kind"] for m in sr.flatten_generations(workflows)[0]["media"]] == ["audio", "video", "image", "audio"]


def test_flatten_unclassifiable_blobs_become_other():
    # No type, no known extension, no telling key -> "other" (nodepack snapshot layers, extensionless
    # customComfy assets). The singular ImageBlob-typed `blob` fields (convertImage etc.) stay image.
    workflows = [
        _wf(
            [
                {
                    "output": {
                        "blob": {"id": "converted-img", "available": True, "url": "cu"},
                        "layer": {"id": "snapshot-layer", "available": True, "url": "lu"},
                        "blobs": [{"id": "customcomfy-X-asset-noext", "available": True, "url": "bu"}],
                    }
                }
            ]
        )
    ]
    assert [m["kind"] for m in sr.flatten_generations(workflows)[0]["media"]] == ["image", "other", "other"]


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


def test_import_model_downloads_primary_and_required_components(monkeypatch):
    # The Model Library import mirrors the Model Selector's folder rules: the primary file's folder
    # follows the *file's* type (Diffusion Model -> diffusion_models, not the AIR's checkpoints),
    # and only isRequired components ride along, keyed by their own downloadUrl + file id.
    air = "urn:air:sdxl:checkpoint:civitai:101@202"
    monkeypatch.setattr(
        sr.catalog,
        "components",
        lambda a, token=None: {
            "primary": {"id": 1, "name": "model.safetensors", "type": "Diffusion Model", "downloadUrl": "du1"},
            "clip": [
                {"id": 2, "name": "te.safetensors", "type": "Text Encoder", "downloadUrl": "du2", "isRequired": True},
            ],
            "vae": [
                {"id": 3, "name": "opt.safetensors", "type": "VAE", "downloadUrl": "du3", "isRequired": False},
            ],
        },
    )
    from civitai_comfy_nodes import config, local_models

    monkeypatch.setattr(config, "auth_state", lambda: (None, None))
    calls = []

    def fake_download(a, folder="checkpoints", token=None, *, download_url=None, file_id=None, in_execution=True):
        assert in_execution is False  # route path must not touch execution-scoped progress/interrupts
        calls.append({"folder": folder, "download_url": download_url, "file_id": file_id})
        return f"/models/{folder}/file{len(calls)}.safetensors"

    monkeypatch.setattr(local_models, "download_model", fake_download)
    result = sr._import_model(air)
    assert result == {
        "files": [
            {"folder": "diffusion_models", "name": "file1.safetensors"},
            {"folder": "text_encoders", "name": "file2.safetensors"},
        ]
    }
    assert calls[0] == {"folder": "diffusion_models", "download_url": None, "file_id": None}
    assert calls[1] == {"folder": "text_encoders", "download_url": "du2", "file_id": 2}


def test_guess_ext_sniffs_magic_bytes():
    assert sr._guess_ext("image", b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00") == ".png"
    assert sr._guess_ext("image", b"\xff\xd8\xff\xe0\x00\x10JFIF") == ".jpg"
    assert sr._guess_ext("video", b"\x00\x00\x00\x18ftypmp42") == ".mp4"
    assert sr._guess_ext("audio", b"not a known header") == ".flac"  # falls back by kind


def _prep_step(job_status, rate, *, step_status="preparing"):
    return {
        "$type": "customComfy",
        "status": step_status,
        "jobs": [{"status": job_status, "estimatedProgressRate": rate}],
    }


def test_queue_phase_maps_orchestration_status():
    assert sr._queue_phase("Preparing") == "preparing"
    assert sr._queue_phase("Scheduled") == "preparing"
    assert sr._queue_phase("Unassigned") == "preparing"
    assert sr._queue_phase("Processing") == "processing"
    assert sr._queue_phase("Succeeded") == "succeeded"
    assert sr._queue_phase("Canceled") == "canceled"
    assert sr._queue_phase("Cancelled") == "cancelled"


def test_queue_state_data_carries_preparation_progress():
    wf = {"id": "6-1", "status": "Preparing", "steps": [_prep_step("preparing", 0.42)]}
    data = sr._queue_state_data(wf, "6-1")
    assert data == {"prompt_id": "6-1", "status": "preparing", "progress": 0.42}


def test_queue_state_data_processing_has_no_progress():
    wf = {"id": "6-1", "status": "Processing", "steps": [_prep_step("processing", 0.9, step_status="processing")]}
    data = sr._queue_state_data(wf, "6-1")
    assert data == {"prompt_id": "6-1", "status": "processing"}


def test_preparation_progress_ignores_non_preparing_jobs():
    wf = {"steps": [_prep_step("processing", 0.9, step_status="processing")]}
    assert sr._preparation_progress(wf) is None


def test_offload_active_lists_only_preparing_and_processing():
    with sr._running_lock:
        sr._active_offloads.clear()
        sr._active_offloads["6-1"] = {
            "task_id": -1,
            "sid": "s",
            "queue_state": {"prompt_id": "6-1", "status": "preparing", "progress": 0.5},
        }
        sr._active_offloads["6-2"] = {
            "task_id": -2,
            "sid": "s",
            "queue_state": {"prompt_id": "6-2", "status": "processing"},
        }
        sr._active_offloads["6-3"] = {
            "task_id": -3,
            "sid": "s",
            "queue_state": {"prompt_id": "6-3", "status": "succeeded"},
        }
    try:
        jobs = {job["id"]: job for job in sr._offload_active()["jobs"]}
        assert set(jobs) == {"6-1", "6-2"}  # terminal 6-3 excluded
        assert jobs["6-1"] == {"id": "6-1", "civitai_orch_status": "preparing", "civitai_preparation_progress": 0.5}
        assert jobs["6-2"] == {"id": "6-2", "civitai_orch_status": "processing", "civitai_preparation_progress": None}
    finally:
        with sr._running_lock:
            sr._active_offloads.clear()
