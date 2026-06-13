from civitai_comfy_nodes import catalog


def test_flatten_builds_airs_and_skips_unknown_ecosystems():
    items = [
        {
            "id": 100,
            "name": "Cool LoRA",
            "type": "LORA",
            "stats": {"downloadCount": 42},
            "modelVersions": [
                {"id": 200, "name": "v1", "baseModel": "SDXL 1.0", "images": [{"url": "http://img/1.png"}]},
                {"id": 201, "name": "v2", "baseModel": "Some Future Model"},  # no ecosystem -> skipped
                {"id": 202, "name": "v3", "baseModel": "Pony"},
            ],
        }
    ]
    entries = catalog.flatten_models(items)
    airs = [e["air"] for e in entries]
    assert airs == [
        "urn:air:sdxl:lora:civitai:100@200",
        "urn:air:sdxl:lora:civitai:100@202",  # Pony -> sdxl
    ]
    assert entries[0]["thumbnailUrl"] == "http://img/1.png"
    assert entries[0]["downloadCount"] == 42
    assert entries[0]["name"] == "Cool LoRA"


def test_flatten_caps_versions_and_filters_type():
    items = [
        {"id": 1, "type": "Checkpoint", "modelVersions": [{"id": i, "baseModel": "SD 1.5"} for i in range(10)]},
        {"id": 2, "type": "LORA", "modelVersions": [{"id": 99, "baseModel": "SD 1.5"}]},
    ]
    checkpoints = catalog.flatten_models(items, max_versions=3, type_filter="Checkpoint")
    assert len(checkpoints) == 3
    assert all(e["type"] == "Checkpoint" for e in checkpoints)


def test_ecosystem_map():
    assert catalog.ecosystem_for("Flux.1 D") == "flux1"
    assert catalog.ecosystem_for("Illustrious") == "sdxl"
    assert catalog.ecosystem_for(None) is None
    assert catalog.ecosystem_for("Nonexistent") is None


def test_server_routes_imports_without_comfyui():
    # Must import cleanly under pytest (no `server` module) and register nothing.
    from civitai_comfy_nodes import server_routes

    assert server_routes._server is None
