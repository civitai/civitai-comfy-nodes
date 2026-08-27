import os

from civitai_comfy_nodes import model_cache


def _model(tmp_path, data=b"weights"):
    path = tmp_path / "model.safetensors"
    path.write_bytes(data)
    return path


def test_put_then_get_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = _model(tmp_path)

    model_cache.put(model, hashes={"SHA256": "ABC"}, air="urn:air:x@1", model_version_id=7)
    entry = model_cache.get(model)

    assert entry is not None
    assert entry["hashes"] == {"SHA256": "ABC"}
    assert entry["air"] == "urn:air:x@1"
    assert entry["model_version_id"] == 7


def test_get_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    assert model_cache.get(_model(tmp_path)) is None


def test_get_invalidates_when_file_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = _model(tmp_path, b"weights")
    model_cache.put(model, hashes={"SHA256": "ABC"}, air="urn:air:x@1")

    model.write_bytes(b"weights-but-larger-now")  # size changes -> identity mismatch

    assert model_cache.get(model) is None


def test_get_invalidates_when_mtime_changes(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = _model(tmp_path, b"weights")
    model_cache.put(model, hashes={"SHA256": "ABC"}, air="urn:air:x@1")
    assert model_cache.get(model) is not None

    stat = os.stat(model)
    os.utime(model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))  # same size, newer mtime

    assert model_cache.get(model) is None


def test_put_without_air_stores_hashes_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = _model(tmp_path)

    model_cache.put(model, hashes={"SHA256": "ABC"})
    entry = model_cache.get(model)

    assert entry is not None
    assert entry["hashes"] == {"SHA256": "ABC"}
    assert entry.get("air") is None


def test_get_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    assert model_cache.get(tmp_path / "does-not-exist.safetensors") is None


def test_bulk_get_returns_only_fresh_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    fresh = _model(tmp_path)
    stale = tmp_path / "stale.safetensors"
    stale.write_bytes(b"v1")
    model_cache.put(fresh, hashes={"SHA256": "ABC"})
    model_cache.put(stale, hashes={"SHA256": "DEF"})
    stale.write_bytes(b"v2-longer")

    found = model_cache.bulk_get([fresh, stale, tmp_path / "missing.safetensors"])

    assert set(found) == {str(fresh)}
    assert found[str(fresh)]["hashes"] == {"SHA256": "ABC"}


def test_find_by_hash_is_case_insensitive_and_ignores_changed_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = _model(tmp_path)
    model_cache.put(model, hashes={"SHA256": "ABC123"})

    assert model_cache.find_by_hash("abc123") == str(model)
    assert model_cache.find_by_hash("zzz") is None
    model.write_bytes(b"changed-contents")
    assert model_cache.find_by_hash("ABC123") is None


def test_remove_drops_the_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = _model(tmp_path)
    model_cache.put(model, hashes={"SHA256": "ABC"})
    model_cache.remove(model)
    assert model_cache.get(model) is None
    model_cache.remove(model)  # idempotent
