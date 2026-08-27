import json

import pytest

from civitai_comfy_nodes import link_protocol as proto
from civitai_comfy_nodes.offload import LocalModelRecord

HASH = "a" * 64


def test_folder_for_resource_maps_model_types_and_prefers_file_type():
    assert proto.folder_for_resource("Checkpoint") == "checkpoints"
    assert proto.folder_for_resource("Checkpoint", "Diffusion Model") == "diffusion_models"
    assert proto.folder_for_resource("LoCon") == "loras"
    assert proto.folder_for_resource("DoRA", "Model") == "loras"  # generic file type never overrides
    assert proto.folder_for_resource("TextualInversion") == "embeddings"
    assert proto.folder_for_resource("Poses") is None
    assert proto.folder_for_resource(None) is None


def test_safe_filename_strips_directories_and_rejects_traversal():
    assert proto.safe_filename("  model.safetensors ") == "model.safetensors"
    assert proto.safe_filename("sub/dir/model.safetensors") == "model.safetensors"
    assert proto.safe_filename("C:\\models\\x.pt") == "x.pt"
    for bad in ("", "  ", "..", "../", "a\x00b"):
        with pytest.raises(ValueError):
            proto.safe_filename(bad)


def test_normalize_key_and_sha256():
    assert proto.normalize_key(" A1B2C3 ") == "a1b2c3"
    assert proto.normalize_key("f" * 128) == "f" * 128
    for bad in ("", "abc", "xyz123", "a" * 64):
        with pytest.raises(ValueError):
            proto.normalize_key(bad)
    assert proto.normalize_sha256(HASH.upper()) == HASH
    with pytest.raises(ValueError):
        proto.normalize_sha256("abc")


def test_make_response_echoes_command_and_stamps_times():
    command = {"id": "1", "type": "resources:add", "createdAt": "2026-01-01T00:00:00Z", "resource": {"name": "x"}}
    response = proto.make_response(command, "processing", progress=12.5, error=None)
    assert response["id"] == "1" and response["type"] == "resources:add"
    assert response["createdAt"] == "2026-01-01T00:00:00Z"
    assert response["resource"] == {"name": "x"}
    assert response["status"] == "processing" and response["progress"] == 12.5
    assert "error" not in response  # None fields are dropped
    assert response["updatedAt"]
    assert proto.make_response({"id": "2", "type": "resources:list"}, "success")["createdAt"]


def test_progress_fields_math():
    fields = proto.progress_fields(written=50, total=200, started=0.0, now=10.0)
    assert fields == {"speed": 5.0, "progress": 25.0, "remainingTime": 30.0}
    assert proto.progress_fields(written=300, total=200, started=0.0, now=1.0)["progress"] == 100.0
    assert "progress" not in proto.progress_fields(written=5, total=0, started=0.0, now=1.0)


def test_resource_entry_lowercases_hash_and_uses_folder_relative_path():
    entry = proto.resource_entry("loras", "sub/x.safetensors", HASH.upper())
    assert entry == {
        "type": "LORA",
        "hash": HASH,
        "name": "x.safetensors",
        "path": "loras/sub/x.safetensors",
        "hasPreview": "",
    }
    assert proto.resource_entry("loras", "x", HASH, downloading=True)["downloading"] is True


def test_parse_resource_path_rejects_unknown_folders_and_traversal():
    assert proto.parse_resource_path("loras/sub/x.safetensors") == ("loras", "sub/x.safetensors")
    assert proto.parse_resource_path("loras\\x.safetensors") == ("loras", "x.safetensors")
    for bad in (None, "", "loras", "loras/", "nope/x", "loras/../x", "loras/sub//x", "loras/./x"):
        assert proto.parse_resource_path(bad) is None


def test_build_resource_list_splits_cached_and_pending():
    records = [
        LocalModelRecord(folder="loras", name="a.safetensors", path="/m/loras/a.safetensors"),
        LocalModelRecord(folder="checkpoints", name="b.safetensors", path="/m/checkpoints/b.safetensors"),
        LocalModelRecord(folder="vae", name="c.safetensors", path="/m/vae/c.safetensors"),
    ]
    cache = {
        "/m/loras/a.safetensors": {"hashes": {"SHA256": HASH.upper()}},
        "/m/vae/c.safetensors": {"hashes": {"AutoV3": "x"}},  # no SHA256 -> still pending
    }
    entries, pending = proto.build_resource_list(records, cache)
    assert [e["path"] for e in entries] == ["loras/a.safetensors"]
    assert entries[0]["hash"] == HASH
    assert [r.name for r in pending] == ["b.safetensors", "c.safetensors"]


def test_trim_resource_list_drops_path_then_entries():
    entries = [proto.resource_entry("loras", f"model-{i}.safetensors", HASH) for i in range(50)]
    same = proto.trim_resource_list(entries, max_bytes=10**6)
    assert same is entries
    slim = proto.trim_resource_list(entries, max_bytes=len(json.dumps(entries)) - 1)
    assert slim and all("path" not in e for e in slim)
    tiny = proto.trim_resource_list(entries, max_bytes=400)
    assert 0 < len(tiny) < 50


def test_unique_destination_never_clobbers():
    taken = {"/m/x.safetensors", "/m/x_aaaaaaaa.safetensors"}
    assert proto.unique_destination("/m", "y.safetensors", HASH, exists=taken.__contains__) == "/m/y.safetensors"
    assert (
        proto.unique_destination("/m", "x.safetensors", HASH, exists=taken.__contains__)
        == "/m/x_aaaaaaaa_2.safetensors"
    )


def test_activities_ring_upserts_in_place_and_caps():
    ring = proto.Activities(limit=3)
    for i in range(4):
        ring.upsert({"id": str(i), "status": "pending"})
    assert [a["id"] for a in ring.list()] == ["1", "2", "3"]
    ring.upsert({"id": "2", "status": "success"})
    assert [a["id"] for a in ring.list()] == ["1", "2", "3"]  # position kept
    assert ring.get("2")["status"] == "success"
    ring.upsert({"status": "x"})  # no id -> ignored
    assert len(ring.list()) == 3
    ring.clear()
    assert ring.list() == []
