from . import generated, nodes_manual

NODE_CLASS_MAPPINGS = {
    **generated.NODE_CLASS_MAPPINGS,
    **nodes_manual.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **generated.NODE_DISPLAY_NAME_MAPPINGS,
    **nodes_manual.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
