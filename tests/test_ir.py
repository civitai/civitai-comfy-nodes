import json
from pathlib import Path

import pytest

from codegen import generate

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def spec():
    return json.loads((REPO_ROOT / "spec" / "v2-consumers.json").read_text())


@pytest.fixture(scope="session")
def overrides():
    return json.loads((REPO_ROOT / "codegen" / "overrides.json").read_text())


@pytest.fixture(scope="session")
def nodes(spec, overrides):
    return generate.build_nodes(spec, overrides)


def node_by_name(nodes, class_name):
    matches = [n for n in nodes if n.class_name == class_name]
    assert matches, f"node {class_name} not generated"
    return matches[0]


def test_videogen_expands_to_spec_engines(spec, nodes):
    spec_engines = set(spec["components"]["schemas"]["VideoGenInput"]["discriminator"]["mapping"])
    generated_engines = {n.discriminator["engine"] for n in nodes if n.recipe == "videoGen"}
    assert generated_engines == spec_engines
    assert len(spec_engines) >= 17


def test_skipped_recipes_produce_no_nodes(nodes, overrides):
    generated_recipes = {n.recipe for n in nodes}
    assert generated_recipes.isdisjoint(set(overrides["_skip"]))


def test_wan_node_flattens_nested_discriminators(nodes):
    wan = node_by_name(nodes, "CivitaiVideoGenWan")
    fields = {f.api: f for f in wan.fields}
    assert wan.discriminator == {"engine": "wan"}
    assert "engine" not in fields

    version = fields["version"]
    assert isinstance(version.comfy_type, list)
    assert {"v2.1", "v2.2", "v2.5", "v2.6", "v2.7"} <= set(version.comfy_type)
    assert version.options.get("default") == "v2.1"

    # provider/operation come from deeper nested discriminators
    assert isinstance(fields["provider"].comfy_type, list)
    assert {"civitai", "fal"} <= set(fields["provider"].comfy_type)

    assert fields["sourceImage"].kind == "image_inline"
    assert fields["prompt"].required


def test_texttoimage_widget_bounds(nodes):
    t2i = node_by_name(nodes, "CivitaiTextToImage")
    fields = {f.api: f for f in t2i.fields}
    steps = fields["steps"]
    assert steps.comfy_type == "INT"
    assert (steps.options["min"], steps.options["max"], steps.options["default"]) == (1, 150, 30)
    seed = fields["seed"]
    assert seed.options.get("control_after_generate") is True
    assert fields["sourceImage"].kind == "image_inline"
    assert fields["additionalNetworks"].kind == "json"
    assert [o.kind for o in t2i.outputs] == ["image_list"]


def test_image_upscaler_image_input_and_output(nodes):
    upscaler = node_by_name(nodes, "CivitaiImageUpscaler")
    fields = {f.api: f for f in upscaler.fields}
    assert fields["image"].kind == "image_inline"
    assert fields["image"].required is True
    assert [o.kind for o in upscaler.outputs] == ["image"]


def test_url_media_fields_upload(nodes):
    transcription = node_by_name(nodes, "CivitaiTranscription")
    fields = {f.api: f for f in transcription.fields}
    assert fields["mediaUrl"].kind == "audio_url"

    video_upscaler = node_by_name(nodes, "CivitaiVideoUpscaler")
    assert {f.api: f for f in video_upscaler.fields}["video"].kind == "video_url"


def test_acestep_blob_union_occupies_two_slots(nodes):
    ace = node_by_name(nodes, "CivitaiAceStepAudio")
    assert [o.kind for o in ace.outputs] == ["audio_or_video"]


def test_optional_enums_offer_omit_choice(nodes):
    t2i = node_by_name(nodes, "CivitaiTextToImage")
    scheduler = {f.api: f for f in t2i.fields}["scheduler"]
    assert scheduler.comfy_type[0] == ""
    assert scheduler.options["default"] == ""


def test_enum_refs_resolve_to_combos(nodes):
    kling = node_by_name(nodes, "CivitaiVideoGenKling")
    fields = {f.api: f for f in kling.fields}
    assert isinstance(fields["duration"].comfy_type, list)
    assert isinstance(fields["model"].comfy_type, list)


def test_all_recipes_have_module_assignment(spec, overrides):
    recipes = generate.list_recipes(spec, overrides)
    for name, _, _ in recipes:
        assert name in generate.MODULES
