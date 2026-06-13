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


def test_wan_nested_discriminators_become_separate_nodes(nodes):
    # engine/version/provider/operation are all expanded into distinct nodes (fixed discriminators),
    # not collapsed into dropdowns on one node.
    wan = node_by_name(nodes, "CivitaiVideoGenWanV21Fal")
    assert wan.discriminator == {"engine": "wan", "version": "v2.1", "provider": "fal"}
    fields = {f.api: f for f in wan.fields}
    assert "version" not in fields and "provider" not in fields and "engine" not in fields
    assert fields["prompt"].required
    assert fields["sourceImage"].kind == "image_inline"
    # the operation axis also splits into its own nodes
    assert any(n.discriminator.get("operation") == "editVideo" for n in nodes if n.recipe == "videoGen")


def test_identical_siblings_collapse_to_dropdown(nodes):
    # flux1-kontext pro/max/dev share an identical field set -> one node with a `model` dropdown.
    kontext = node_by_name(nodes, "CivitaiImageGenFlux1Kontext")
    model = {f.api: f for f in kontext.fields}["model"]
    assert model.detected_as == "discriminator-combo"
    assert {"pro", "max", "dev"} <= set(model.comfy_type)
    assert "model" not in kontext.discriminator  # collapsed, not fixed


def test_operation_split_removes_image_overlap(nodes):
    # createImage and editImage are separate nodes; only the edit node carries a source-image input.
    create = node_by_name(nodes, "CivitaiImageGenSdcppFlux1CreateImage")
    edit = node_by_name(nodes, "CivitaiImageGenSdcppFlux1EditImage")
    create_imgs = {f.api for f in create.fields if f.kind in ("image_inline", "image_list", "image_url")}
    edit_imgs = {f.api for f in edit.fields if f.kind in ("image_inline", "image_list", "image_url")}
    assert create_imgs == set()
    assert edit_imgs  # edit has at least one image input
    assert create.discriminator["operation"] == "createImage"
    assert edit.discriminator["operation"] == "editImage"


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


def test_node_names_and_displays_unique(nodes):
    assert len({n.class_name for n in nodes}) == len(nodes)
    assert len({n.display_name for n in nodes}) == len(nodes)


def test_all_recipes_have_module_assignment(spec, overrides):
    recipes = generate.list_recipes(spec, overrides)
    for name, _, _ in recipes:
        assert name in generate.MODULES
