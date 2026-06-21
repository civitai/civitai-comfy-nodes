from civitai_comfy_nodes.node_registry import OFFLOAD_NODE_KEYS, enabled_node_keys

KEYS = ["SomeRecipe", "CivitaiAuth", "CivitaiModelSelector", "CivitaiOffloadStart", "CivitaiOffloadEnd"]
RECIPE_KEYS = {"SomeRecipe", "CivitaiAuth", "CivitaiModelSelector"}


def test_both_enabled_by_default():
    assert enabled_node_keys(KEYS, {}) == set(KEYS)


def test_disable_offload_drops_only_markers():
    enabled = enabled_node_keys(KEYS, {"enableOffload": False})
    assert enabled == RECIPE_KEYS
    assert not (enabled & OFFLOAD_NODE_KEYS)


def test_disable_recipe_nodes_keeps_only_markers():
    assert enabled_node_keys(KEYS, {"enableRecipeNodes": False}) == set(OFFLOAD_NODE_KEYS)


def test_disable_both_registers_nothing():
    assert enabled_node_keys(KEYS, {"enableOffload": False, "enableRecipeNodes": False}) == set()
