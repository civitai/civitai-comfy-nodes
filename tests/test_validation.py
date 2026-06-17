import pytest

from civitai_comfy_nodes.base import CivitaiRecipeNodeBase, F
from civitai_comfy_nodes.errors import CivitaiNodeError


class _Node(CivitaiRecipeNodeBase):
    STEP_TYPE = "imageGen"
    FIELDS = {
        "prompt": F("prompt"),
        "neg": F("negativePrompt"),
        "messages": F("messages", "json"),
        "model": F("model", "air"),  # socket, guarded by ComfyUI — never flagged here
        "seed": F("seed"),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {}),
                "messages": ("STRING", {"multiline": True}),
                "model": ("CIVITAI_AIR", {}),
                "seed": ("INT", {}),
            },
            "optional": {"neg": ("STRING", {})},
        }


def test_blank_required_text_is_flagged():
    n = _Node()
    assert n._missing_required({"prompt": "", "messages": "[]", "model": "", "seed": 0}) == ["prompt"]
    # whitespace counts as blank
    assert n._missing_required({"prompt": "   ", "messages": "[]", "seed": 0}) == ["prompt"]


def test_required_json_blank_is_flagged():
    n = _Node()
    assert n._missing_required({"prompt": "hi", "messages": "   ", "seed": 0}) == ["messages"]


def test_air_socket_and_optional_not_flagged():
    n = _Node()
    # an empty `air` socket is ComfyUI's job to catch (unconnected required socket), not ours
    assert "model" not in n._missing_required({"prompt": "hi", "messages": "[]", "model": "", "seed": 0})
    # optional blanks are fine
    assert n._missing_required({"prompt": "hi", "messages": "[]", "neg": "", "seed": 0}) == []


def test_zero_number_is_not_blank():
    n = _Node()
    assert n._missing_required({"prompt": "hi", "messages": "[]", "model": "x", "seed": 0}) == []


def test_run_raises_before_any_network_on_missing():
    # validation happens before auth/submit, so this needs no creds or network
    with pytest.raises(CivitaiNodeError) as exc:
        _Node().run(prompt="", messages="[]", model="urn:air:sd1:checkpoint:civitai:1@2", seed=0)
    assert "prompt" in str(exc.value)


def test_missing_message_pluralizes():
    assert _Node._missing_message(["prompt"]) == "Missing required input: 'prompt'. Fill it in before running."
    msg = _Node._missing_message(["prompt", "messages"])
    assert "inputs: 'prompt', 'messages'" in msg and "Fill them in" in msg
