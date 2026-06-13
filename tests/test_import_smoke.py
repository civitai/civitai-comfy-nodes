"""The package must import and describe every node without the ComfyUI runtime present."""

from civitai_comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def test_node_count():
    assert len(NODE_CLASS_MAPPINGS) >= 57


def test_display_names_unique_and_complete():
    assert set(NODE_DISPLAY_NAME_MAPPINGS) == set(NODE_CLASS_MAPPINGS)
    displays = list(NODE_DISPLAY_NAME_MAPPINGS.values())
    assert len(displays) == len(set(displays))


def test_every_node_describes_itself():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        input_types = cls.INPUT_TYPES()
        assert "required" in input_types, name
        assert isinstance(cls.RETURN_TYPES, tuple), name
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES), name
        assert callable(getattr(cls, cls.FUNCTION)), name
        assert cls.CATEGORY.startswith("Civitai"), name


def test_generated_nodes_end_with_bookkeeping_outputs():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        if not hasattr(cls, "RECIPE") or not cls.RECIPE:
            continue
        assert cls.RETURN_NAMES[-2:] == ("workflow_id", "raw_json"), name


def _is_recipe_node(cls):
    return bool(getattr(cls, "RECIPE", "")) or cls.__name__ == "CivitaiChatSimple"


def test_recipe_nodes_are_output_nodes():
    # Recipe nodes must be runnable standalone (many return only STRING/JSON);
    # config/loader helper nodes are pure data builders and must NOT be outputs.
    for name, cls in NODE_CLASS_MAPPINGS.items():
        if _is_recipe_node(cls):
            assert cls.OUTPUT_NODE is True, name
        else:
            assert getattr(cls, "OUTPUT_NODE", False) is False, name


def test_api_config_input_present_on_recipe_nodes():
    for name, cls in NODE_CLASS_MAPPINGS.items():
        if not _is_recipe_node(cls):
            continue
        optional = cls.INPUT_TYPES().get("optional", {})
        assert "api_config" in optional, name
        assert optional["api_config"][0] == "CIVITAI_CONFIG", name
