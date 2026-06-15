import pytest

from civitai_comfy_nodes import local_models, oauth
from civitai_comfy_nodes.errors import CivitaiNodeError
from civitai_comfy_nodes.nodes_manual import CivitaiModelSelector


def test_version_id_parse():
    assert local_models.version_id_from_air("urn:air:sdxl:checkpoint:civitai:101055@128078") == "128078"
    with pytest.raises(CivitaiNodeError):
        local_models.version_id_from_air("not-an-air")


def test_folder_for_air_maps_type_to_comfy_folder():
    assert local_models.folder_for_air("urn:air:sdxl:checkpoint:civitai:1@2") == "checkpoints"
    assert local_models.folder_for_air("urn:air:sdxl:lora:civitai:1@2") == "loras"
    assert local_models.folder_for_air("urn:air:sdxl:lycoris:civitai:1@2") == "loras"
    assert local_models.folder_for_air("urn:air:flux1:vae:civitai:1@2") == "vae"
    assert local_models.folder_for_air("urn:air:x:controlnet:civitai:1@2") == "controlnet"
    assert local_models.folder_for_air("garbage") == "checkpoints"  # default


def test_download_uses_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(local_models, "_model_dir", lambda folder: str(tmp_path))
    cached = tmp_path / "civitai_128078_dreamshaper.safetensors"
    cached.write_text("weights")

    def boom(*a, **k):
        raise AssertionError("hit the network despite a cached file")

    monkeypatch.setattr(local_models.requests, "get", boom)
    assert local_models.download_model("urn:air:x:checkpoint:civitai:1@128078") == str(cached)


def test_path_consumed_detection():
    consumed = CivitaiModelSelector._path_consumed
    # node "5" takes its ckpt_name input from node "3" output slot 1 (the `path` output)
    prompt = {"5": {"inputs": {"ckpt_name": ["3", 1]}}, "6": {"inputs": {"text": "hi"}}}
    assert consumed(prompt, "3") is True
    assert consumed(prompt, "9") is False  # nothing references node 9
    # only the AIR (slot 0) is consumed -> no download
    assert consumed({"5": {"inputs": {"model_air": ["3", 0]}}}, "3") is False
    assert consumed(None, "3") is False  # no prompt available -> don't download


def test_is_changed_tracks_path_connection():
    wired = CivitaiModelSelector.IS_CHANGED("urn@1", prompt={"5": {"inputs": {"m": ["3", 1]}}}, unique_id="3")
    unwired = CivitaiModelSelector.IS_CHANGED("urn@1", prompt={}, unique_id="3")
    assert wired != unwired


def test_select_air_only_does_not_download(monkeypatch):
    # No `path` wired -> returns (air, "") and never downloads.
    def boom(*a, **k):
        raise AssertionError("downloaded despite AIR-only wiring")

    monkeypatch.setattr(local_models, "download_model", boom)
    air, path = CivitaiModelSelector().select("  urn:air:x:checkpoint:civitai:1@2  ", prompt={}, unique_id="3")
    assert air == "urn:air:x:checkpoint:civitai:1@2"
    assert path == ""


def test_select_downloads_to_air_folder_when_path_wired(monkeypatch):
    monkeypatch.setattr(oauth, "get_valid_access_token", lambda: None)
    captured = {}

    def fake_download(air, folder, token):
        captured["folder"] = folder
        return f"/models/{folder}/civitai_2_model.safetensors"

    monkeypatch.setattr(local_models, "download_model", fake_download)
    # ckpt_name on node "9" is wired to this node ("3") `path` output (slot 1)
    prompt = {"9": {"inputs": {"ckpt_name": ["3", 1]}}}
    air, path = CivitaiModelSelector().select("urn:air:sdxl:lora:civitai:1@2", prompt=prompt, unique_id="3")
    assert captured["folder"] == "loras"  # folder derived from the AIR type
    assert path == "civitai_2_model.safetensors"  # folder-relative name for the loader combo
    assert air == "urn:air:sdxl:lora:civitai:1@2"


def _rows(*entries):
    import json

    return json.dumps([{"air": air, "strength": strength, "on": True} for air, strength in entries])


def test_lora_loader_cloud_mode_does_not_download(monkeypatch):
    from civitai_comfy_nodes.nodes_manual import CivitaiLoraLoader

    monkeypatch.setattr(local_models, "download_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    stack, model, clip = CivitaiLoraLoader().load(loras_json=_rows(("urn:air:sdxl:lora:civitai:1@2", 0.7)))
    assert stack == [{"air": "urn:air:sdxl:lora:civitai:1@2", "strength": 0.7}]
    assert model is None and clip is None


def test_lora_loader_local_mode_applies_whole_stack(monkeypatch):
    from civitai_comfy_nodes import nodes_manual
    from civitai_comfy_nodes.nodes_manual import CivitaiLoraLoader

    downloaded = []
    applied = []
    monkeypatch.setattr(nodes_manual, "local_models", local_models, raising=False)

    def fake_download(air, folder, token):
        downloaded.append((air, folder))
        return f"/{air}"

    monkeypatch.setattr(local_models, "download_model", fake_download)

    def fake_apply(model, clip, path, strength):
        applied.append((path, strength))
        return f"{model}+{path}", f"{clip}+{path}"

    monkeypatch.setattr(local_models, "apply_lora", fake_apply)
    # two loras in one node's rows, applied locally
    stack, model, clip = CivitaiLoraLoader().load(
        loras_json=_rows(("urn:air:x:lora:civitai:1@2", 0.5), ("urn:air:x:lora:civitai:3@4", 0.8)),
        model="M",
        clip="C",
    )
    assert [a for a, _ in downloaded] == ["urn:air:x:lora:civitai:1@2", "urn:air:x:lora:civitai:3@4"]
    assert applied == [("/urn:air:x:lora:civitai:1@2", 0.5), ("/urn:air:x:lora:civitai:3@4", 0.8)]
    assert model.startswith("M+") and clip.startswith("C+")
