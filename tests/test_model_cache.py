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
