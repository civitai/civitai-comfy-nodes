import pytest

from civitai_comfy_nodes import catalog, config, local_models
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


def test_folder_for_file_type_maps_civitai_file_type():
    assert local_models.folder_for_file_type("Diffusion Model") == "diffusion_models"
    assert local_models.folder_for_file_type("UNet") == "diffusion_models"
    assert local_models.folder_for_file_type("Model") == "checkpoints"
    assert local_models.folder_for_file_type("VAE") == "vae"
    assert local_models.folder_for_file_type("Text Encoder") == "text_encoders"
    assert local_models.folder_for_file_type("CLIPVision") == "clip_vision"
    assert local_models.folder_for_file_type("Config", "checkpoints") == "checkpoints"  # non-model -> default
    assert local_models.folder_for_file_type(None) == "checkpoints"


def test_download_uses_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(local_models, "_model_dir", lambda folder: str(tmp_path))
    cached = tmp_path / "civitai_128078_dreamshaper.safetensors"
    cached.write_text("weights")

    def boom(*a, **k):
        raise AssertionError("hit the network despite a cached file")

    monkeypatch.setattr(local_models.requests, "get", boom)
    assert local_models.download_model("urn:air:x:checkpoint:civitai:1@128078") == str(cached)


def test_download_specific_file_keyed_by_file_id(monkeypatch, tmp_path):
    monkeypatch.setattr(local_models, "_model_dir", lambda folder: str(tmp_path))
    # a sibling primary file of the same version is already cached; the file-id glob must not match it
    (tmp_path / "civitai_999_model.safetensors").write_text("primary")
    captured = {}

    class _Resp:
        status_code = 200
        headers = {"content-disposition": 'filename="clip_l.safetensors"', "content-length": "5"}

        def iter_content(self, chunk_size=0):
            yield b"abcde"

        def close(self):
            pass

    def fake_get(url, headers=None, stream=None, timeout=None, allow_redirects=None):
        captured["url"] = url
        return _Resp()

    monkeypatch.setattr(local_models.requests, "get", fake_get)
    out = local_models.download_model(
        "urn:air:zimage:checkpoint:civitai:1@999",
        folder="text_encoders",
        download_url="https://civitai.com/api/download/models/999?type=Text%20Encoder",
        file_id=4,
    )
    assert captured["url"].endswith("type=Text%20Encoder")  # the file's own downloadUrl, not the version URL
    assert out == str(tmp_path / "civitai_999_f4_clip_l.safetensors")  # cache name scoped by file id


def test_download_specific_file_uses_its_own_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(local_models, "_model_dir", lambda folder: str(tmp_path))
    (tmp_path / "civitai_999_f4_clip_l.safetensors").write_text("weights")

    def boom(*a, **k):
        raise AssertionError("hit the network despite a cached component file")

    monkeypatch.setattr(local_models.requests, "get", boom)
    out = local_models.download_model(
        "urn:air:x:checkpoint:civitai:1@999", folder="text_encoders", download_url="u", file_id=4
    )
    assert out == str(tmp_path / "civitai_999_f4_clip_l.safetensors")


def test_consumed_slots_detection():
    consumed = CivitaiModelSelector._consumed_slots
    # node "5" takes its ckpt_name input from node "3" output slot 1 (the `path` output)
    prompt = {"5": {"inputs": {"ckpt_name": ["3", 1]}}, "6": {"inputs": {"text": "hi"}}}
    assert consumed(prompt, "3") == {1}
    assert consumed(prompt, "9") == set()  # nothing references node 9
    # only the AIR (slot 0) is consumed -> no download slots
    assert consumed({"5": {"inputs": {"model_air": ["3", 0]}}}, "3") == {0}
    assert consumed(None, "3") == set()  # no prompt available -> don't download
    # unet (path, slot 1), clip (slot 2) and vae (slot 3) all wired from one downstream node
    multi = {"7": {"inputs": {"unet_name": ["3", 1], "clip_name": ["3", 2], "vae_name": ["3", 3]}}}
    assert consumed(multi, "3") == {1, 2, 3}


def test_is_changed_tracks_path_connection():
    wired = CivitaiModelSelector.IS_CHANGED("urn@1", prompt={"5": {"inputs": {"m": ["3", 1]}}}, unique_id="3")
    unwired = CivitaiModelSelector.IS_CHANGED("urn@1", prompt={}, unique_id="3")
    assert wired != unwired


def test_select_air_only_does_not_download(monkeypatch):
    # No download output wired -> returns the air plus empty paths and never downloads.
    def boom(*a, **k):
        raise AssertionError("downloaded despite AIR-only wiring")

    monkeypatch.setattr(local_models, "download_model", boom)
    result = CivitaiModelSelector().select("  urn:air:x:checkpoint:civitai:1@2  ", prompt={}, unique_id="3")
    assert result[0] == "urn:air:x:checkpoint:civitai:1@2"
    assert result[1:] == ("", "", "", "", "")  # path + vae + clip + clip 2 + clip 3 all empty


def test_select_path_falls_back_to_air_folder_without_primary_type(monkeypatch):
    monkeypatch.setattr(config, "auth_state", lambda: (None, "none"))
    monkeypatch.setattr(catalog, "components", lambda air, token=None: {"primary": None, "vae": [], "clip": []})
    captured = {}

    def fake_download(air, folder, token, **kw):
        captured["folder"] = folder
        return f"/models/{folder}/civitai_2_model.safetensors"

    monkeypatch.setattr(local_models, "download_model", fake_download)
    # ckpt_name on node "9" is wired to this node ("3") `path` output (slot 1)
    prompt = {"9": {"inputs": {"ckpt_name": ["3", 1]}}}
    result = CivitaiModelSelector().select("urn:air:sdxl:lora:civitai:1@2", prompt=prompt, unique_id="3")
    assert captured["folder"] == "loras"  # no primary file type -> fall back to the AIR type
    assert result[1] == "civitai_2_model.safetensors"  # folder-relative name for the loader combo
    assert result[0] == "urn:air:sdxl:lora:civitai:1@2"


def test_select_path_folder_follows_primary_file_type(monkeypatch):
    monkeypatch.setattr(config, "auth_state", lambda: (None, "none"))
    # AIR type says checkpoint, but the primary file is a Diffusion Model -> diffusion_models/
    monkeypatch.setattr(
        catalog,
        "components",
        lambda air, token=None: {
            "primary": {"id": 1, "name": "m.safetensors", "type": "Diffusion Model", "downloadUrl": "u"},
            "vae": [],
            "clip": [],
        },
    )
    captured = {}

    def fake_download(air, folder, token, **kw):
        captured["folder"] = folder
        return f"/models/{folder}/civitai_9_m.safetensors"

    monkeypatch.setattr(local_models, "download_model", fake_download)
    prompt = {"9": {"inputs": {"unet_name": ["3", 1]}}}
    result = CivitaiModelSelector().select("urn:air:zimage:checkpoint:civitai:1@9", prompt=prompt, unique_id="3")
    assert captured["folder"] == "diffusion_models"  # from the FILE type, not the AIR's "checkpoint"
    assert result[1] == "civitai_9_m.safetensors"


def test_select_downloads_components_when_wired(monkeypatch):
    monkeypatch.setattr(config, "auth_state", lambda: (None, "none"))
    monkeypatch.setattr(
        catalog,
        "components",
        lambda air, token=None: {
            "primary": None,
            "vae": [{"id": 22, "name": "ae.safetensors", "type": "VAE", "downloadUrl": "u-vae"}],
            "clip": [{"id": 33, "name": "clip_l.safetensors", "type": "Text Encoder", "downloadUrl": "u-clip"}],
        },
    )
    calls = []

    def fake_download(air, folder, token, download_url=None, file_id=None):
        calls.append((folder, download_url, file_id))
        return f"/models/{folder}/civitai_9_f{file_id}_{folder}.safetensors"

    monkeypatch.setattr(local_models, "download_model", fake_download)
    # node "7" wires the first clip (slot 2) and the vae (slot 3) of node "3"
    prompt = {"7": {"inputs": {"clip_name": ["3", 2], "vae_name": ["3", 3]}}}
    result = CivitaiModelSelector().select("urn:air:zimage:checkpoint:civitai:1@9", prompt=prompt, unique_id="3")
    assert result[2] == "civitai_9_f33_text_encoders.safetensors"  # clip output -> text_encoders/
    assert result[3] == "civitai_9_f22_vae.safetensors"  # vae output -> vae/ folder, keyed by file id
    assert result[1] == ""  # path not wired -> primary not downloaded
    assert ("vae", "u-vae", 22) in calls
    assert ("text_encoders", "u-clip", 33) in calls


def test_select_errors_when_a_wired_component_is_missing(monkeypatch):
    monkeypatch.setattr(config, "auth_state", lambda: (None, "none"))
    monkeypatch.setattr(catalog, "components", lambda air, token=None: {"vae": [], "clip": []})
    monkeypatch.setattr(local_models, "download_model", lambda *a, **k: "/x")
    prompt = {"7": {"inputs": {"clip_name": ["3", 2]}}}  # clip wired but the model has none
    with pytest.raises(CivitaiNodeError):
        CivitaiModelSelector().select("urn:air:x:checkpoint:civitai:1@9", prompt=prompt, unique_id="3")


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
