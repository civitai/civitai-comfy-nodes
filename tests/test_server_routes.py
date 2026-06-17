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
