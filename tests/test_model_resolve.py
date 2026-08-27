import hashlib
import json

from civitai_comfy_nodes import model_resolve


def _write_safetensors(path, header, payload=b"tensor-bytes"):
    header_bytes = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + payload)


def test_reads_safetensors_metadata_hash_before_computing(tmp_path):
    model = tmp_path / "model.safetensors"
    embedded = "a" * 64
    _write_safetensors(model, {"__metadata__": {"sshs_model_hash": embedded}})

    hashes, source = model_resolve.get_model_hashes(model)

    assert source == "metadata"
    assert hashes == {"AutoV3": embedded.upper()}


def test_compute_model_hashes_matches_scanner_shape(tmp_path):
    model = tmp_path / "tiny.bin"
    model.write_bytes(b"hello world")

    hashes = model_resolve.compute_model_hashes(model)

    sha = hashlib.sha256(b"hello world").hexdigest().upper()
    assert hashes["SHA256"] == sha
    assert hashes["AutoV2"] == sha[:10]
    assert hashes["CRC32"] == "0D4A1185"
    assert "AutoV1" not in hashes


def test_compute_autov3_for_safetensors_payload(tmp_path):
    model = tmp_path / "model.safetensors"
    payload = b"payload-only"
    _write_safetensors(model, {"__metadata__": {"format": "pt"}}, payload=payload)

    hashes = model_resolve.compute_model_hashes(model)

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

    resolved = model_resolve.resolve_model_air(model, session=session, civitai_base_url="http://civitai.test")

    assert resolved.air == "urn:air:sdxl:checkpoint:civitai:1@12"
    assert resolved.hash_source == "computed"
    assert session.urls[0].endswith("/" + "B" * 64)
    assert session.urls[1].endswith("/" + hashlib.sha256(model.read_bytes()).hexdigest().upper())


def test_resolve_model_air_caches_and_skips_network_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = tmp_path / "model.safetensors"
    _write_safetensors(model, {"__metadata__": {"format": "pt"}}, payload=b"payload")
    first = _Session([_Resp(200, {"id": 9, "air": "urn:air:sdxl:checkpoint:civitai:5@9"})])

    resolved = model_resolve.resolve_model_air(model, session=first, civitai_base_url="http://civitai.test")
    assert resolved.air == "urn:air:sdxl:checkpoint:civitai:5@9"
    assert len(first.urls) == 1  # SHA256 lookup hit on the first candidate

    second = _Session([])  # no responses queued -> any network call would IndexError
    cached = model_resolve.resolve_model_air(model, session=second, civitai_base_url="http://civitai.test")
    assert cached.air == "urn:air:sdxl:checkpoint:civitai:5@9"
    assert cached.hash_source == "cache"
    assert second.urls == []  # served from cache, no network


def test_resolve_model_air_skips_autov3_when_sha256_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = tmp_path / "model.safetensors"
    _write_safetensors(model, {"__metadata__": {"format": "pt"}}, payload=b"payload")
    session = _Session([_Resp(200, {"id": 9, "air": "urn:air:x@9"})])

    resolved = model_resolve.resolve_model_air(model, session=session, civitai_base_url="http://civitai.test")

    assert resolved.air == "urn:air:x@9"
    assert "AutoV3" not in resolved.hashes  # the second full-file pass is skipped when SHA256 hits


def test_resolve_model_air_reuses_cached_hashes_without_rehashing(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = tmp_path / "model.safetensors"
    _write_safetensors(model, {"__metadata__": {"format": "pt"}}, payload=b"payload")

    miss = _Session([_Resp(404, {}) for _ in range(6)])  # not found -> hashes cached, no air
    assert model_resolve.resolve_model_air(model, session=miss, civitai_base_url="http://civitai.test") is None

    def _boom(*args, **kwargs):
        raise AssertionError("compute_model_hashes must not run when hashes are cached")

    monkeypatch.setattr(model_resolve, "compute_model_hashes", _boom)
    hit = _Session([_Resp(200, {"id": 3, "air": "urn:air:x@3"})])
    resolved = model_resolve.resolve_model_air(model, session=hit, civitai_base_url="http://civitai.test")

    assert resolved.air == "urn:air:x@3"
    assert resolved.hash_source == "computed"


def test_resolve_model_air_relooks_up_autov3_after_negative_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    model = tmp_path / "model.safetensors"
    _write_safetensors(model, {"__metadata__": {"format": "pt"}}, payload=b"payload")

    # First call: nothing matches -> AutoV3 gets computed and cached (no air).
    miss = _Session([_Resp(404, {}) for _ in range(6)])
    assert model_resolve.resolve_model_air(model, session=miss, civitai_base_url="http://civitai.test") is None

    # Second call: the model is now on Civitai, matchable only by AutoV3. Must not re-hash, and must
    # re-run the AutoV3 lookup even though AutoV3 was cached from the prior miss.
    def _no_rehash(*args, **kwargs):
        raise AssertionError("must not re-hash when hashes are cached")

    monkeypatch.setattr(model_resolve, "compute_model_hashes", _no_rehash)
    monkeypatch.setattr(model_resolve, "_autov3_safetensors_payload", _no_rehash)
    autov3 = hashlib.sha256(b"payload").hexdigest().upper()
    second = _Session([_Resp(404, {}), _Resp(200, {"id": 5, "air": "urn:air:x@5"})])
    resolved = model_resolve.resolve_model_air(model, session=second, civitai_base_url="http://civitai.test")

    assert resolved is not None and resolved.air == "urn:air:x@5"
    assert any(url.endswith("/" + autov3) for url in second.urls)


def test_version_file_for_hash_matches_any_hash_type_case_insensitively():
    version = {
        "files": [
            {"type": "Model", "hashes": {"SHA256": "AAA", "AutoV3": "aaa3"}},
            {"type": "VAE", "hashes": {"SHA256": "BBB", "AutoV3": "BBB3"}},
        ]
    }
    assert model_resolve.version_file_for_hash(version, "bbb3")["type"] == "VAE"
    assert model_resolve.version_file_for_hash(version, "aaa")["type"] == "Model"
    assert model_resolve.version_file_for_hash(version, "nope") is None
    assert model_resolve.version_file_for_hash(None, "aaa") is None


def test_lookup_record_replaces_metadata_sha256_with_the_canonical_file_hash(tmp_path, monkeypatch):
    # A header can't hold its own whole-file digest, so a metadata "SHA256" is a tensor hash; the
    # version file record from the by-hash lookup carries the real one.
    model = tmp_path / "m.safetensors"
    model.write_bytes(b"x")
    version = {
        "id": 5,
        "air": "urn:air:sd1:lora:civitai:1@5",
        "files": [{"type": "Model", "hashes": {"SHA256": "REAL", "AutoV3": "TENSORHASH"}}],
    }
    monkeypatch.setattr(
        model_resolve, "lookup_model_version_by_hash", lambda value, **kw: version if value == "TENSORHASH" else None
    )

    record = model_resolve._lookup_record(
        model, {"SHA256": "BOGUS", "AutoV3": "TENSORHASH"}, "metadata", token=None, session=None, civitai_base_url=None
    )

    assert record.air == version["air"] and record.lookup_hash_type == "AutoV3"
    assert record.hashes == {"AutoV3": "TENSORHASH", "SHA256": "REAL"}
    computed = model_resolve._lookup_record(
        model, {"SHA256": "REAL2", "AutoV3": "TENSORHASH"}, "computed", token=None, session=None, civitai_base_url=None
    )
    assert computed.hashes["SHA256"] == "REAL"  # the API's value wins even over a computed one
