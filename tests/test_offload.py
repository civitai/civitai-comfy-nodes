import hashlib
import json
import sys
import types

from civitai_comfy_nodes import offload


def _write_safetensors(path, header, payload=b"tensor-bytes"):
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + payload)


def test_reads_safetensors_metadata_hash_before_computing(tmp_path):
    model = tmp_path / "model.safetensors"
    embedded = "a" * 64
    _write_safetensors(model, {"__metadata__": {"sshs_model_hash": embedded}})

    hashes, source = offload.get_model_hashes(model)

    assert source == "metadata"
    assert hashes == {"AutoV3": embedded.upper()}


def test_compute_model_hashes_matches_scanner_shape(tmp_path):
    model = tmp_path / "tiny.bin"
    model.write_bytes(b"hello world")

    hashes = offload.compute_model_hashes(model)

    sha = hashlib.sha256(b"hello world").hexdigest().upper()
    assert hashes["SHA256"] == sha
    assert hashes["AutoV2"] == sha[:10]
    assert hashes["CRC32"] == "0D4A1185"
    assert "AutoV1" not in hashes


def test_compute_autov3_for_safetensors_payload(tmp_path):
    model = tmp_path / "model.safetensors"
    payload = b"payload-only"
    _write_safetensors(model, {"__metadata__": {"format": "pt"}}, payload=payload)

    hashes = offload.compute_model_hashes(model)

    assert hashes["AutoV3"] == hashlib.sha256(payload).hexdigest().upper()


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.responses.pop(0)


def test_resolve_model_air_uses_metadata_then_computed_fallback(tmp_path):
    model = tmp_path / "model.safetensors"
    _write_safetensors(model, {"__metadata__": {"sshs_model_hash": "b" * 64}}, payload=b"payload")
    session = _Session(
        [
            _Resp(404, {"error": "not found"}),
            _Resp(200, {"id": 12, "air": "urn:air:sdxl:checkpoint:civitai:1@12"}),
        ]
    )

    resolved = offload.resolve_model_air(model, session=session, civitai_base_url="http://civitai.test")

    assert resolved.air == "urn:air:sdxl:checkpoint:civitai:1@12"
    assert resolved.hash_source == "computed"
    assert session.urls[0].endswith("/" + "B" * 64)
    assert session.urls[1].endswith("/" + hashlib.sha256(model.read_bytes()).hexdigest().upper())


def test_scan_installed_nodepacks_infers_air_only_with_version(tmp_path):
    rgthree = tmp_path / "rgthree-comfy"
    rgthree.mkdir()
    (rgthree / "package.json").write_text(
        json.dumps(
            {
                "name": "rgthree-comfy",
                "version": "1.0.2605082257",
                "repository": {"url": "https://github.com/rgthree/rgthree-comfy.git"},
            }
        )
    )
    commit_only = tmp_path / "commit-only"
    commit_only.mkdir()
    (commit_only / "package.json").write_text(
        json.dumps({"name": "commit-only", "repository": "https://github.com/acme/commit-only"})
    )

    nodepacks = {nodepack.folder: nodepack for nodepack in offload.scan_installed_nodepacks(tmp_path)}

    assert nodepacks["rgthree-comfy"].air == (
        "urn:air:comfy:nodepack:comfyregistry:rgthree/rgthree-comfy@1.0.2605082257"
    )
    assert nodepacks["commit-only"].version is None
    assert nodepacks["commit-only"].air is None


def test_build_custom_comfy_offload_rewrites_local_models_and_adds_nodepacks(tmp_path):
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "dream.safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    model = offload.LocalModelRecord(
        folder="checkpoints",
        name="dream.safetensors",
        path=str(tmp_path / "dream.safetensors"),
        air="urn:air:sdxl:checkpoint:civitai:11@22",
    )
    nodepack = offload.InstalledNodepack(
        folder="rgthree-comfy",
        registry_id="rgthree/rgthree-comfy",
        version="1.0.0",
        air="urn:air:comfy:nodepack:comfyregistry:rgthree/rgthree-comfy@1.0.0",
    )

    built = offload.build_custom_comfy_offload(prompt, model_records=[model], nodepacks=[nodepack])

    custom_input = built.steps[0]["input"]
    assert built.steps[0]["$type"] == "customComfy"
    assert custom_input["workflow"]["1"]["inputs"]["ckpt_name"] == "urn:air:sdxl:checkpoint:civitai:11@22"
    assert custom_input["resources"] == [
        "urn:air:comfy:nodepack:comfyregistry:rgthree/rgthree-comfy@1.0.0",
        "urn:air:sdxl:checkpoint:civitai:11@22",
    ]


def test_build_custom_comfy_offload_includes_vram_and_sage_when_set():
    prompt = {"1": {"class_type": "SaveImage", "inputs": {}}}
    built = offload.build_custom_comfy_offload(
        prompt, model_records=[], nodepacks=[], min_vram_gb=24, use_sage_attention=True
    )
    custom_input = built.steps[0]["input"]
    assert custom_input["minVramGb"] == 24
    assert custom_input["useSageAttention"] is True


def test_build_custom_comfy_offload_omits_vram_sage_and_gpu_by_default():
    prompt = {"1": {"class_type": "SaveImage", "inputs": {}}}
    custom_input = offload.build_custom_comfy_offload(prompt, model_records=[], nodepacks=[]).steps[0]["input"]
    assert "minVramGb" not in custom_input
    assert "useSageAttention" not in custom_input
    assert "gpuGeneration" not in custom_input  # GPU generation is display-only, never submitted


def test_build_custom_comfy_offload_uploads_load_image_inputs_as_blob_airs(tmp_path):
    image = tmp_path / "fried-duck.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake-png")
    uploads = []

    def upload(path, content_type):
        uploads.append((path, content_type))
        return {"id": "abc123.png", "url": "http://orch/v2/consumer/blobs/abc123.png"}

    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "fried-duck.png"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=upload,
        input_path_resolver=lambda name: tmp_path / name,
    )

    air = "urn:air:other:other:orchestrator:blob@abc123.png"
    assert uploads == [(image.resolve(), "image/png")]
    assert built.workflow["1"]["inputs"]["image"] == air
    assert built.steps[0]["input"]["workflow"]["1"]["inputs"]["image"] == air
    assert built.steps[0]["input"]["resources"] == [air]
    assert built.input_blobs[0]["original_name"] == "fried-duck.png"
    assert built.input_blobs[0]["air"] == air


def test_build_custom_comfy_offload_uploads_duplicate_load_images_once(tmp_path):
    image = tmp_path / "source.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake-png")
    calls = []

    def upload(path, content_type):
        calls.append((path, content_type))
        return {"id": "dup.png"}

    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
        "2": {"class_type": "LoadImageMask", "inputs": {"image": "source.png", "channel": "alpha"}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=upload,
        input_path_resolver=lambda name: tmp_path / name,
    )

    air = "urn:air:other:other:orchestrator:blob@dup.png"
    assert calls == [(image.resolve(), "image/png")]
    assert built.workflow["1"]["inputs"]["image"] == air
    assert built.workflow["2"]["inputs"]["image"] == air
    assert built.resources == [air]


def test_build_custom_comfy_offload_keeps_existing_load_image_air():
    air = "urn:air:other:other:orchestrator:blob@already.png"
    prompt = {
        "1": {"class_type": "LoadImage", "inputs": {"image": air}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=lambda _path, _content_type: (_ for _ in ()).throw(AssertionError("should not upload")),
    )

    assert built.workflow["1"]["inputs"]["image"] == air
    assert built.resources == [air]
    assert built.input_blobs == []


def test_build_custom_comfy_offload_uploads_audio_inputs_as_blob_airs(tmp_path):
    audio = tmp_path / "voice.mp3"
    audio.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00fake-audio")
    uploads = []

    def upload(path, content_type):
        uploads.append((path, content_type))
        return {"id": "voice.mp3"}

    prompt = {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": "voice.mp3"}},
        "2": {"class_type": "SaveAudioMP3", "inputs": {"filename_prefix": "audio-test", "audio": ["1", 0]}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=upload,
        input_path_resolver=lambda name: tmp_path / name,
    )

    air = "urn:air:other:other:orchestrator:blob@voice.mp3"
    assert uploads == [(audio.resolve(), "audio/mpeg")]
    assert built.workflow["1"]["inputs"]["audio"] == air
    assert built.resources == [air]
    assert built.input_blobs[0]["content_type"] == "audio/mpeg"


def test_build_custom_comfy_offload_uploads_video_inputs_as_blob_airs(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video")
    uploads = []

    def upload(path, content_type):
        uploads.append((path, content_type))
        return {"id": "clip.mp4"}

    prompt = {
        "1": {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}},
        "2": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video-test", "video": ["1", 0]}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=upload,
        input_path_resolver=lambda name: tmp_path / name,
    )

    air = "urn:air:other:other:orchestrator:blob@clip.mp4"
    assert uploads == [(video.resolve(), "video/mp4")]
    assert built.workflow["1"]["inputs"]["file"] == air
    assert built.steps[0]["input"]["resources"] == [air]
    assert built.input_blobs[0]["input_name"] == "file"


def test_build_custom_comfy_offload_prefers_audio_webm_for_audio_loader(tmp_path):
    audio = tmp_path / "voice.webm"
    audio.write_bytes(b"\x1a\x45\xdf\xa3fake-webm")

    prompt = {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": "voice.webm"}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=lambda _path, _content_type: {"id": "voice.webm"},
        input_path_resolver=lambda name: tmp_path / name,
    )

    assert built.input_blobs[0]["content_type"] == "audio/webm"


def test_build_custom_comfy_offload_uploads_video_helper_suite_upload_inputs(tmp_path):
    video = tmp_path / "clip.webm"
    video.write_bytes(b"\x1a\x45\xdf\xa3fake-webm")

    prompt = {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "clip.webm", "frame_load_cap": 1}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=lambda _path, _content_type: {"id": "clip.webm"},
        input_path_resolver=lambda name: tmp_path / name,
    )

    assert built.workflow["1"]["inputs"]["video"] == "urn:air:other:other:orchestrator:blob@clip.webm"
    assert built.input_blobs[0]["content_type"] == "video/webm"


def test_build_custom_comfy_offload_keeps_existing_media_url():
    prompt = {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": "https://example.test/voice.mp3"}},
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        model_records=[],
        nodepacks=[],
        upload_blob_file=lambda _path, _content_type: (_ for _ in ()).throw(AssertionError("should not upload")),
    )

    assert built.workflow["1"]["inputs"]["audio"] == "https://example.test/voice.mp3"
    assert built.resources == []
    assert built.input_blobs == []


def test_build_custom_comfy_offload_only_adds_used_nodepacks(monkeypatch):
    class UsedNode:
        RELATIVE_PYTHON_MODULE = "custom_nodes.rgthree-comfy.nodes"

    class UnusedNode:
        RELATIVE_PYTHON_MODULE = "custom_nodes.other-pack.nodes"

    monkeypatch.setitem(
        sys.modules,
        "nodes",
        types.SimpleNamespace(
            NODE_CLASS_MAPPINGS={
                "UsedCustomNode": UsedNode,
                "UnusedCustomNode": UnusedNode,
                "SaveImage": type("SaveImage", (), {"RELATIVE_PYTHON_MODULE": "nodes"}),
            }
        ),
    )
    prompt = {
        "1": {"class_type": "UsedCustomNode", "inputs": {}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    nodepacks = [
        offload.InstalledNodepack(
            folder="rgthree-comfy",
            registry_id="rgthree/rgthree-comfy",
            version="1.0.0",
            air="urn:air:comfy:nodepack:comfyregistry:rgthree/rgthree-comfy@1.0.0",
        ),
        offload.InstalledNodepack(
            folder="other-pack",
            registry_id="acme/other-pack",
            version="2.0.0",
            air="urn:air:comfy:nodepack:comfyregistry:acme/other-pack@2.0.0",
        ),
    ]

    built = offload.build_custom_comfy_offload(prompt, model_records=[], nodepacks=nodepacks)

    assert built.steps[0]["input"]["resources"] == [
        "urn:air:comfy:nodepack:comfyregistry:rgthree/rgthree-comfy@1.0.0"
    ]
    assert [nodepack["folder"] for nodepack in built.nodepack_resources] == ["rgthree-comfy"]


def test_offload_markers_select_region_and_are_stripped():
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "2": {"class_type": "CivitaiOffloadStart", "inputs": {"region_id": "r", "value": ["1", 0]}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
        "4": {"class_type": "CivitaiOffloadEnd", "inputs": {"region_id": "r", "value": ["3", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
    }

    built = offload.build_custom_comfy_offload(prompt, model_records=[], nodepacks=[])

    assert "2" not in built.workflow
    assert "4" not in built.workflow
    assert built.workflow["3"]["inputs"]["images"] == ["1", 0]
    assert built.selected_node_ids == ["2", "3", "4"]
    assert built.included_node_ids == ["1", "2", "3", "4"]


def test_offload_markers_preserve_user_save_inside_region_and_exclude_local_tail():
    prompt = {
        "1": {"class_type": "VAEDecode", "inputs": {"samples": ["100", 0], "vae": ["101", 0]}},
        "2": {"class_type": "CivitaiOffloadStart", "inputs": {"region_id": "anima", "value": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "civitai-offload-anima", "images": ["2", 0]}},
        "4": {"class_type": "CivitaiOffloadEnd", "inputs": {"region_id": "anima", "value": ["2", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": "civitai_offload_marker_remote.png"}},
        "6": {"class_type": "ImageInvert", "inputs": {"image": ["5", 0]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "local-after-offload", "images": ["6", 0]}},
        "100": {"class_type": "KSampler", "inputs": {}},
        "101": {"class_type": "VAELoader", "inputs": {}},
    }

    built = offload.build_custom_comfy_offload(prompt, model_records=[], nodepacks=[])

    assert "2" not in built.workflow
    assert "4" not in built.workflow
    assert "5" not in built.workflow
    assert "6" not in built.workflow
    assert "7" not in built.workflow
    assert built.workflow["3"] == {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "civitai-offload-anima", "images": ["1", 0]},
    }


def test_visual_offload_markers_select_nodes_between_start_and_end_with_save_image():
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "2": {"class_type": "ImageInvert", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "remote", "images": ["2", 0]}},
        "4": {"class_type": "ImageBlur", "inputs": {"image": ["2", 0], "blur_radius": 2, "sigma": 1.0}},
        "5": {"class_type": "SaveImage", "inputs": {"filename_prefix": "local", "images": ["4", 0]}},
    }
    workflow = {
        "nodes": [
            {"id": 100, "type": "CivitaiOffloadStart", "pos": [0, 0], "widgets_values": ["anima"]},
            {"id": 1, "type": "EmptyImage", "pos": [120, -120]},
            {"id": 2, "type": "ImageInvert", "pos": [280, -120]},
            {"id": 3, "type": "SaveImage", "pos": [440, -120]},
            {"id": 101, "type": "CivitaiOffloadEnd", "pos": [620, 0], "widgets_values": ["anima"]},
            {"id": 4, "type": "ImageBlur", "pos": [760, -120]},
            {"id": 5, "type": "SaveImage", "pos": [920, -120]},
        ]
    }

    built = offload.build_custom_comfy_offload(prompt, workflow=workflow, model_records=[], nodepacks=[])

    assert built.selected_node_ids == ["1", "2", "3"]
    assert built.included_node_ids == ["1", "2", "3"]
    assert set(built.workflow) == {"1", "2", "3"}
    assert built.workflow["3"] == {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "remote", "images": ["2", 0]},
    }


def test_explicit_selection_overrides_visual_offload_markers():
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "2": {"class_type": "SaveImage", "inputs": {"filename_prefix": "remote", "images": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"filename_prefix": "manual", "images": ["1", 0]}},
    }
    workflow = {
        "nodes": [
            {"id": 100, "type": "CivitaiOffloadStart", "pos": [0, 0], "widgets_values": ["default"]},
            {"id": 1, "type": "EmptyImage", "pos": [120, 0]},
            {"id": 2, "type": "SaveImage", "pos": [280, 0]},
            {"id": 101, "type": "CivitaiOffloadEnd", "pos": [440, 0], "widgets_values": ["default"]},
            {"id": 3, "type": "SaveImage", "pos": [600, 0]},
        ]
    }

    built = offload.build_custom_comfy_offload(
        prompt,
        selected_node_ids=["3"],
        workflow=workflow,
        model_records=[],
        nodepacks=[],
    )

    assert built.selected_node_ids == ["3"]
    assert built.included_node_ids == ["1", "3"]
    assert set(built.workflow) == {"1", "3"}


def test_build_local_continuation_replaces_remote_region_with_load_image_bridge():
    prompt = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "clip.safetensors"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima.safetensors"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": "prompt"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "6": {
            "class_type": "KSampler",
            "inputs": {"model": ["2", 0], "positive": ["4", 0], "negative": ["4", 0], "latent_image": ["5", 0]},
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "remote", "images": ["7", 0]}},
        "9": {"class_type": "ImageInvert", "inputs": {"image": ["7", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": "local", "images": ["9", 0]}},
    }

    built = offload.build_local_continuation_prompt(
        prompt,
        remote_node_ids=["1", "2", "3", "4", "5", "6", "7", "8"],
        imported_image_name="civitai_offload_remote.png",
    )

    assert built is not None
    assert built.bridge_node_id == "civitai_remote_asset"
    assert built.tail_node_ids == ["9", "10"]
    assert built.output_node_ids == ["10"]
    assert built.remote_source_node_ids == ["7"]
    assert built.prompt == {
        "civitai_remote_asset": {"class_type": "LoadImage", "inputs": {"image": "civitai_offload_remote.png"}},
        "9": {"class_type": "ImageInvert", "inputs": {"image": ["civitai_remote_asset", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"filename_prefix": "local", "images": ["9", 0]}},
    }


def test_build_local_continuation_returns_none_when_no_local_tail():
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }

    assert (
        offload.build_local_continuation_prompt(
            prompt,
            remote_node_ids=["1", "2"],
            imported_image_name="civitai_offload_remote.png",
        )
        is None
    )


def test_build_custom_comfy_offload_includes_trace_when_requested():
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }

    build = offload.build_custom_comfy_offload(prompt, model_records=[], nodepacks=[], trace="binary")

    assert build.steps[0]["$type"] == "customComfy"
    assert build.steps[0]["input"]["trace"] == "binary"


def test_build_custom_comfy_offload_omits_trace_by_default():
    prompt = {
        "1": {"class_type": "EmptyImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }

    build = offload.build_custom_comfy_offload(prompt, model_records=[], nodepacks=[])

    assert "trace" not in build.steps[0]["input"]
