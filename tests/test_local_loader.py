import pytest

from civitai_comfy_nodes import local_models
from civitai_comfy_nodes.errors import CivitaiNodeError
from civitai_comfy_nodes.nodes_manual import CivitaiCheckpointLoader


def test_version_id_parse():
    assert local_models.version_id_from_air("urn:air:sdxl:checkpoint:civitai:101055@128078") == "128078"
    with pytest.raises(CivitaiNodeError):
        local_models.version_id_from_air("not-an-air")


def test_download_uses_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(local_models, "_model_dir", lambda folder: str(tmp_path))
    cached = tmp_path / "civitai_128078_dreamshaper.safetensors"
    cached.write_text("weights")

    def boom(*a, **k):
        raise AssertionError("hit the network despite a cached file")

    monkeypatch.setattr(local_models.requests, "get", boom)
    assert local_models.download_model("urn:air:x:checkpoint:civitai:1@128078") == str(cached)


def test_local_outputs_consumed_detection():
    consumed = CivitaiCheckpointLoader._local_outputs_consumed
    # node "5" takes its model input from node "3" output slot 1 (MODEL)
    prompt = {"5": {"inputs": {"model": ["3", 1]}}, "6": {"inputs": {"text": "hi"}}}
    assert consumed(prompt, "3") is True
    assert consumed(prompt, "9") is False  # nothing references node 9
    # only the AIR (slot 0) is consumed -> no local download
    assert consumed({"5": {"inputs": {"model_air": ["3", 0]}}}, "3") is False
    assert consumed(None, "3") is False  # no prompt available -> don't download


def test_is_changed_tracks_local_connection():
    wired = CivitaiCheckpointLoader.IS_CHANGED("urn@1", prompt={"5": {"inputs": {"m": ["3", 1]}}}, unique_id="3")
    unwired = CivitaiCheckpointLoader.IS_CHANGED("urn@1", prompt={}, unique_id="3")
    assert wired != unwired


def test_cloud_only_load_returns_air_without_downloading(monkeypatch):
    # No local outputs wired -> returns the AIR and Nones, never touching local_models.
    def boom(*a, **k):
        raise AssertionError("downloaded despite cloud-only wiring")

    monkeypatch.setattr(local_models, "download_model", boom)
    result = CivitaiCheckpointLoader().load("urn:air:x:checkpoint:civitai:1@2", prompt={}, unique_id="3")
    assert result == ("urn:air:x:checkpoint:civitai:1@2", None, None, None)
