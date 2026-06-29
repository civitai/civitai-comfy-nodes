from civitai_comfy_nodes import catalog


def test_flatten_builds_airs_and_skips_unknown_ecosystems():
    items = [
        {
            "id": 100,
            "name": "Cool LoRA",
            "type": "LORA",
            "stats": {"downloadCount": 42},
            "modelVersions": [
                {"id": 200, "name": "v1", "baseModel": "SDXL 1.0", "images": [{"url": "http://img/1.png"}],
                 "trainedWords": ["brush stroke", "traditional media"]},
                {"id": 201, "name": "v2", "baseModel": "Some Future Model"},  # no ecosystem -> skipped
                {"id": 202, "name": "v3", "baseModel": "Pony"},
            ],
        }
    ]
    entries = catalog.flatten_models(items)
    # One entry per model; its versions ride along (the unknown-ecosystem one skipped).
    assert len(entries) == 1
    version_airs = [v["air"] for v in entries[0]["versions"]]
    assert version_airs == [
        "urn:air:sdxl:lora:civitai:100@200",
        "urn:air:sdxl:lora:civitai:100@202",  # Pony -> sdxl
    ]
    # Top-level fields mirror the representative (first) version.
    assert entries[0]["air"] == "urn:air:sdxl:lora:civitai:100@200"
    assert entries[0]["thumbnailUrl"] == "http://img/1.png"
    assert entries[0]["downloadCount"] == 42
    assert entries[0]["name"] == "Cool LoRA"
    assert entries[0]["modelId"] == 100
    assert entries[0]["versionId"] == 200
    assert entries[0]["modelUrl"] == "https://civitai.com/models/100?modelVersionId=200"
    assert entries[0]["trainedWords"] == ["brush stroke", "traditional media"]
    assert entries[0]["versions"][1]["trainedWords"] == []  # absent -> empty list
    # Components are attached per version (no files here -> empty buckets).
    assert entries[0]["versions"][0]["components"] == {"primary": None, "vae": [], "clip": []}


def test_air_type_uses_civitai_type_map_not_lowercase():
    def air_of(model_type):
        items = [{"id": 1, "type": model_type, "modelVersions": [{"id": 2, "baseModel": "SD 1.5"}]}]
        return catalog.flatten_models(items)[0]["air"]

    assert air_of("TextualInversion") == "urn:air:sd1:embedding:civitai:1@2"  # not :textualinversion:
    assert air_of("LoCon") == "urn:air:sd1:lycoris:civitai:1@2"
    assert air_of("Hypernetwork") == "urn:air:sd1:hypernet:civitai:1@2"
    assert air_of("UNet") == "urn:air:sd1:unet:civitai:1@2"


def test_flatten_skips_non_resource_types():
    items = [{"id": 1, "type": "Poses", "modelVersions": [{"id": 2, "baseModel": "SD 1.5"}]}]
    assert catalog.flatten_models(items) == []


def test_flatten_caps_versions_and_filters_type():
    items = [
        {"id": 1, "type": "Checkpoint", "modelVersions": [{"id": i, "baseModel": "SD 1.5"} for i in range(10)]},
        {"id": 2, "type": "LORA", "modelVersions": [{"id": 99, "baseModel": "SD 1.5"}]},
    ]
    checkpoints = catalog.flatten_models(items, max_versions=3, type_filter="Checkpoint")
    # max_versions caps versions within a model; the LORA model is filtered out by type.
    assert len(checkpoints) == 1
    assert checkpoints[0]["type"] == "Checkpoint"
    assert len(checkpoints[0]["versions"]) == 3


def test_ecosystem_map():
    assert catalog.ecosystem_for("Flux.1 D") == "flux1"
    assert catalog.ecosystem_for("Illustrious") == "sdxl"
    assert catalog.ecosystem_for("ZImageTurbo") == "zimage"
    assert catalog.ecosystem_for("ZImageBase") == "zimagebase"
    assert catalog.ecosystem_for("Qwen") == "qwen"
    assert catalog.ecosystem_for(None) is None
    assert catalog.ecosystem_for("Nonexistent") is None


def test_base_models_for_roundtrips():
    assert catalog.base_models_for("zimage") == ["ZImageTurbo"]
    assert "Pony" in catalog.base_models_for("sdxl")


def test_air_ecosystem_parse():
    assert catalog.air_ecosystem("urn:air:sd1:checkpoint:civitai:4384@128713") == "sd1"
    assert catalog.air_ecosystem("urn:air:flux1:lora:civitai:1@2") == "flux1"
    assert catalog.air_ecosystem("") is None


def test_node_ecosystem_from_discriminator():
    assert catalog.node_ecosystem({"engine": "sdcpp", "ecosystem": "zImage", "model": "turbo"}) == "zimage"
    assert catalog.node_ecosystem({"engine": "sdcpp", "ecosystem": "zImage", "model": "base"}) == "zimagebase"
    assert catalog.node_ecosystem({"engine": "sdcpp", "ecosystem": "sdxl"}) == "sdxl"
    assert catalog.node_ecosystem({"engine": "flux2", "model": "dev"}) == "flux2"
    assert catalog.node_ecosystem({}, model_air="urn:air:sd1:checkpoint:civitai:4384@1") == "sd1"
    assert catalog.node_ecosystem({"engine": "seedream"}) is None


def test_node_ecosystem_map_covers_zimage_turbo():
    from civitai_comfy_nodes import server_routes

    mapping = server_routes.node_ecosystem_map()
    assert mapping["CivitaiImageGenSdcppZImageTurboCreateImage"] == "zimage"
    assert mapping["CivitaiImageGenSdcppZImageBaseCreateImage"] == "zimagebase"


def test_server_routes_imports_without_comfyui():
    # Must import cleanly under pytest (no `server` module) and register nothing.
    from civitai_comfy_nodes import server_routes

    assert server_routes._server is None


def test_version_id_from_air():
    assert catalog.version_id_from_air("urn:air:sd1:checkpoint:civitai:4384@128713") == "128713"
    assert catalog.version_id_from_air("urn:air:sd1:checkpoint:civitai:4384") is None
    assert catalog.version_id_from_air("") is None


def test_lookup_maps_model_version_to_preview(monkeypatch):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "id": 128713,
                "modelId": 4384,
                "name": "v8",
                "baseModel": "SD 1.5",
                "model": {"name": "DreamShaper", "type": "Checkpoint"},
                "images": [{"url": "http://img/cover.jpg"}],
                "trainedWords": ["dreamshaper"],
            }

    monkeypatch.setattr(catalog.requests, "get", lambda *a, **k: _Resp())
    entry = catalog.lookup("urn:air:sd1:checkpoint:civitai:4384@128713")
    assert entry["name"] == "DreamShaper"
    assert entry["versionName"] == "v8"
    assert entry["baseModel"] == "SD 1.5"
    assert entry["thumbnailUrl"] == "http://img/cover.jpg"
    assert entry["ecosystem"] == "sd1"
    assert entry["trainedWords"] == ["dreamshaper"]
    assert entry["modelUrl"] == "https://civitai.com/models/4384?modelVersionId=128713"
    assert entry["components"] == {"primary": None, "vae": [], "clip": []}  # no files in this version


def _version_resp(files):
    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"id": 999, "modelId": 1, "files": files}

    return _Resp()


def test_components_groups_vae_and_clip_in_api_order(monkeypatch):
    files = [
        {"id": 1, "name": "model.safetensors", "type": "Model", "primary": True,
         "downloadUrl": "https://civitai.com/api/download/models/999"},
        {"id": 2, "name": "ae.safetensors", "type": "VAE", "primary": False,
         "downloadUrl": "https://civitai.com/api/download/models/999?type=VAE",
         "metadata": {"isRequired": True}},
        {"id": 3, "name": "clip_l.safetensors", "type": "Text Encoder", "primary": False,
         "downloadUrl": "https://civitai.com/api/download/models/999?type=Text%20Encoder&part=1"},
        {"id": 4, "name": "t5.safetensors", "type": "Text Encoder", "primary": False,
         "downloadUrl": "https://civitai.com/api/download/models/999?type=Text%20Encoder&part=2"},
        {"id": 5, "name": "config.json", "type": "Config", "primary": False,
         "downloadUrl": "https://civitai.com/api/download/models/999?type=Config"},
    ]
    monkeypatch.setattr(catalog.requests, "get", lambda *a, **k: _version_resp(files))
    comps = catalog.components("urn:air:zimage:checkpoint:civitai:1@999")
    assert comps["primary"]["id"] == 1 and comps["primary"]["type"] == "Model"  # captured whatever its type
    assert [f["id"] for f in comps["vae"]] == [2]
    assert comps["vae"][0]["isRequired"] is True
    assert [f["name"] for f in comps["clip"]] == ["clip_l.safetensors", "t5.safetensors"]  # API order preserved
    assert comps["clip"][0]["downloadUrl"].endswith("part=1")  # the file's own URL drives the download
    # primary = the plain version AIR (shared cache identity); non-primary files pinned by +fileId
    assert comps["primary"]["air"] == "urn:air:zimage:checkpoint:civitai:1@999"
    assert comps["vae"][0]["air"] == "urn:air:zimage:vae:civitai:1@999+2"
    assert [f["air"] for f in comps["clip"]] == [
        "urn:air:zimage:text_encoders:civitai:1@999+3",
        "urn:air:zimage:text_encoders:civitai:1@999+4",
    ]


def test_components_captures_primary_regardless_of_type(monkeypatch):
    files = [
        {"id": 1, "name": "diff.safetensors", "type": "Diffusion Model", "primary": True, "downloadUrl": "u1"},
        {"id": 2, "name": "te_no_url", "type": "Text Encoder", "primary": False},  # no downloadUrl -> skipped
    ]
    monkeypatch.setattr(catalog.requests, "get", lambda *a, **k: _version_resp(files))
    comps = catalog.components("urn:air:x:checkpoint:civitai:1@999")
    assert comps["primary"]["type"] == "Diffusion Model"  # AIR says checkpoint, primary file says otherwise
    assert comps["primary"]["air"] == "urn:air:x:checkpoint:civitai:1@999"  # the plain version AIR, no +fileId
    assert comps["vae"] == [] and comps["clip"] == []


def test_components_empty_for_unparseable_air():
    assert catalog.components("not-an-air") == {"primary": None, "vae": [], "clip": []}


def test_components_empty_on_404(monkeypatch):
    class _Resp:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("should not raise on 404")

        def json(self):
            return {}

    monkeypatch.setattr(catalog.requests, "get", lambda *a, **k: _Resp())
    assert catalog.components("urn:air:x:checkpoint:civitai:1@999") == {"primary": None, "vae": [], "clip": []}


def test_lookup_returns_none_for_unparseable_air():
    assert catalog.lookup("not-an-air") is None


def test_lookup_returns_none_on_404(monkeypatch):
    class _Resp:
        status_code = 404

        def raise_for_status(self):
            raise AssertionError("should not raise on 404")

        def json(self):
            return {}

    monkeypatch.setattr(catalog.requests, "get", lambda *a, **k: _Resp())
    assert catalog.lookup("urn:air:sd1:checkpoint:civitai:4384@999") is None


def test_search_does_not_filter_to_generation_models(monkeypatch):
    captured = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr(catalog.requests, "get", fake_get)
    catalog.search(query="dreamshaper", type_="Checkpoint", ecosystem="sdxl")
    assert not any(k == "supportsGeneration" for k, _ in captured["params"])
