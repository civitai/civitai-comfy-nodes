"""Filter the pack's node mappings by the user's feature toggles (Settings -> Civitai).

The two toggles partition every node: the offload marker nodes belong to "Run on Civitai", and
everything else (generated recipe nodes + the manual Auth/Chat/selector nodes) belongs to the
"Civitai recipe nodes" group. Toggles are read at import time, so a change needs a ComfyUI restart.
"""

# Mirror offload.OFFLOAD_START_CLASS / OFFLOAD_END_CLASS without importing the heavy module here.
OFFLOAD_NODE_KEYS = frozenset({"CivitaiOffloadStart", "CivitaiOffloadEnd"})


def enabled_node_keys(all_keys, settings) -> set:
    """Return the subset of node keys to register given the feature toggles in `settings`."""
    enable_recipe = bool(settings.get("enableRecipeNodes", True))
    enable_offload = bool(settings.get("enableOffload", True))
    enabled = set()
    for key in all_keys:
        if key in OFFLOAD_NODE_KEYS:
            if enable_offload:
                enabled.add(key)
        elif enable_recipe:
            enabled.add(key)
    return enabled
