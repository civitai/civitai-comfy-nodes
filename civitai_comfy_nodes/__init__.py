from . import generated, nodes_manual, prompt_context
from .config import load_pack_settings
from .node_registry import enabled_node_keys

prompt_context.register()

_all_classes = {**generated.NODE_CLASS_MAPPINGS, **nodes_manual.NODE_CLASS_MAPPINGS}
_all_names = {**generated.NODE_DISPLAY_NAME_MAPPINGS, **nodes_manual.NODE_DISPLAY_NAME_MAPPINGS}
_enabled = enabled_node_keys(_all_classes, load_pack_settings())

NODE_CLASS_MAPPINGS = {key: cls for key, cls in _all_classes.items() if key in _enabled}
NODE_DISPLAY_NAME_MAPPINGS = {key: name for key, name in _all_names.items() if key in _enabled}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
