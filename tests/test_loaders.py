from civitai_comfy_nodes.base import CivitaiRecipeNodeBase, F
from civitai_comfy_nodes.client import OrchestrationClient
from civitai_comfy_nodes.config import ClientConfig
from civitai_comfy_nodes.nodes_manual import CivitaiCheckpointLoader, CivitaiLoraLoader


def test_lora_loader_chains():
    (first,) = CivitaiLoraLoader().append("urn:air:sdxl:lora:civitai:1@2", 0.8, trigger_word="foo")
    (second,) = CivitaiLoraLoader().append("urn:air:sdxl:lora:civitai:3@4", 1.0, loras=first)
    assert second == [
        {"air": "urn:air:sdxl:lora:civitai:1@2", "strength": 0.8, "triggerWord": "foo"},
        {"air": "urn:air:sdxl:lora:civitai:3@4", "strength": 1.0},
    ]


def test_lora_loader_skips_blank_air():
    (stack,) = CivitaiLoraLoader().append("   ", 1.0)
    assert stack == []


def test_checkpoint_loader_trims():
    assert CivitaiCheckpointLoader().load("  urn:air:x  ") == ("urn:air:x",)


class _ArrayNode(CivitaiRecipeNodeBase):
    FIELDS = {"loras": F("loras", "lora_array")}


class _MapNode(CivitaiRecipeNodeBase):
    FIELDS = {"additional_networks": F("additionalNetworks", "network_map")}


class _CnNode(CivitaiRecipeNodeBase):
    FIELDS = {"control_nets": F("controlNets", "controlnet_array")}


def _client():
    return OrchestrationClient(ClientConfig(base_url="http://x", token="t"))


def test_lora_array_serialization():
    loras = [{"air": "a", "strength": 0.5, "triggerWord": "w"}]
    payload = _ArrayNode()._build_payload(_client(), {"loras": loras})
    assert payload == {"loras": [{"air": "a", "strength": 0.5}]}  # array shape drops triggerWord


def test_network_map_serialization():
    loras = [{"air": "a", "strength": 0.5, "triggerWord": "w"}]
    payload = _MapNode()._build_payload(_client(), {"additional_networks": loras})
    assert payload == {"additionalNetworks": {"a": {"strength": 0.5, "triggerWord": "w"}}}


def test_controlnet_passthrough_and_empty_omitted():
    cn = [{"preprocessor": "canny", "weight": 1.0, "startStep": 0.0, "endStep": 1.0}]
    assert _CnNode()._build_payload(_client(), {"control_nets": cn}) == {"controlNets": cn}
    assert _CnNode()._build_payload(_client(), {"control_nets": []}) == {}


class _StrengthMapNode(CivitaiRecipeNodeBase):
    FIELDS = {"loras": F("loras", "lora_strength_map")}


def test_lora_strength_map_serialization():
    loras = [{"air": "a", "strength": 0.7, "triggerWord": "x"}, {"air": "b", "strength": 1.0}]
    payload = _StrengthMapNode()._build_payload(_client(), {"loras": loras})
    assert payload == {"loras": {"a": 0.7, "b": 1.0}}  # dict-of-strength, triggerWord dropped
