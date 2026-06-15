from civitai_comfy_nodes import server_routes as sr


def _wf(steps, **extra):
    return {"id": "6-1", "status": "Succeeded", "createdAt": "2026-06-14T00:00:00Z", "steps": steps, **extra}


def test_flatten_keeps_available_media_drops_rest():
    workflows = [
        _wf(
            [
                {
                    "$type": "imageGen",
                    "output": {
                        "images": [
                            {"type": "image", "id": "b1", "available": True, "url": "u1", "previewUrl": "p1"},
                            {"type": "image", "id": "b2", "available": False, "url": "u2"},  # unavailable
                            {"type": "image", "id": "b3", "blockedReason": "nsfw", "url": "u3"},  # blocked
                        ]
                    },
                }
            ],
            cost={"total": 16},
        ),
        _wf([{"$type": "vid", "output": {"video": {"type": "video", "id": "v1", "available": True, "url": "vu"}}}]),
        _wf([{"$type": "echo", "output": {"message": "hi"}}]),  # no media -> dropped entirely
    ]
    items = sr.flatten_generations(workflows)
    assert len(items) == 2
    img = items[0]
    assert img["workflowId"] == "6-1" and img["cost"] == 16
    assert [m["blobId"] for m in img["media"]] == ["b1"]
    assert img["media"][0]["kind"] == "image" and img["media"][0]["previewUrl"] == "p1"
    assert items[1]["media"][0]["kind"] == "video"
    assert items[1]["media"][0]["previewUrl"] == "vu"  # falls back to url when no previewUrl


def test_flatten_handles_audio_and_3d_and_kind_filter():
    workflows = [
        _wf(
            [
                {
                    "output": {
                        "images": [{"type": "image", "id": "a", "available": True, "url": "u"}],
                        "audio": {"type": "audio", "id": "au", "available": True, "url": "auu"},
                        "model": {"type": "model3d", "id": "m", "available": True, "url": "mu"},
                    }
                }
            ]
        )
    ]
    everything = sr.flatten_generations(workflows)[0]["media"]
    assert {m["kind"] for m in everything} == {"image", "audio", "model3d"}
    only_audio = sr.flatten_generations(workflows, kinds={"audio"})
    assert [m["kind"] for m in only_audio[0]["media"]] == ["audio"]


def test_guess_ext_sniffs_magic_bytes():
    assert sr._guess_ext("image", b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00") == ".png"
    assert sr._guess_ext("image", b"\xff\xd8\xff\xe0\x00\x10JFIF") == ".jpg"
    assert sr._guess_ext("video", b"\x00\x00\x00\x18ftypmp42") == ".mp4"
    assert sr._guess_ext("audio", b"not a known header") == ".flac"  # falls back by kind
