import json

from civitai_comfy_nodes.base import CivitaiRecipeNodeBase, F
from civitai_comfy_nodes.client import OrchestrationClient
from civitai_comfy_nodes.config import ClientConfig
from civitai_comfy_nodes.nodes_manual import CivitaiLoraLoader


def test_lora_loader_builds_stack_from_rows():
    # Cloud mode (no model/clip wired): builds the stack from the rows JSON, model/clip pass as None.
    rows = json.dumps(
        [
            {"air": "urn:air:sdxl:lora:civitai:1@2", "strength": 0.8, "triggerWord": "foo", "on": True},
            {"air": "urn:air:sdxl:lora:civitai:3@4", "strength": 1.0},  # `on` defaults to enabled
            {"air": "urn:air:sdxl:lora:civitai:5@6", "strength": 1.0, "on": False},  # disabled -> skipped
            {"air": "   ", "on": True},  # blank -> skipped
        ]
    )
    stack, model, clip = CivitaiLoraLoader().load(loras_json=rows)
    assert (model, clip) == (None, None)
    assert stack == [
        {"air": "urn:air:sdxl:lora:civitai:1@2", "strength": 0.8, "triggerWord": "foo"},
        {"air": "urn:air:sdxl:lora:civitai:3@4", "strength": 1.0},
    ]


def test_lora_loader_chains_from_input():
    first = [{"air": "a", "strength": 0.5}]
    rows = json.dumps([{"air": "b", "strength": 1.0, "on": True}])
    stack, _m, _c = CivitaiLoraLoader().load(loras_json=rows, loras=first)
    assert stack == [{"air": "a", "strength": 0.5}, {"air": "b", "strength": 1.0}]


def test_lora_loader_tolerates_empty_and_bad_json():
    assert CivitaiLoraLoader().load(loras_json="[]")[0] == []
    assert CivitaiLoraLoader().load(loras_json="not json")[0] == []
    assert CivitaiLoraLoader().load()[0] == []


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


class _AirNode(CivitaiRecipeNodeBase):
    FIELDS = {"model": F("model", "air"), "embeddings": F("embeddings", "air_list")}


def test_air_and_air_list_serialization():
    node = _AirNode()
    air = "urn:air:sd1:checkpoint:civitai:1@2"
    payload = node._build_payload(_client(), {"model": air, "embeddings": ["e1", "e2"]})
    assert payload == {"model": air, "embeddings": ["e1", "e2"]}
    # empty AIR / empty list are omitted, not sent
    assert node._build_payload(_client(), {"model": "", "embeddings": ["", None]}) == {}


def test_embedding_selector_chains():
    from civitai_comfy_nodes.nodes_manual import CivitaiEmbeddingSelector

    first = CivitaiEmbeddingSelector().append("urn:air:sd1:embedding:civitai:1@2")[0]
    second = CivitaiEmbeddingSelector().append("  urn:air:sd1:embedding:civitai:3@4  ", embeddings=first)[0]
    assert second == ["urn:air:sd1:embedding:civitai:1@2", "urn:air:sd1:embedding:civitai:3@4"]
    assert CivitaiEmbeddingSelector().append("   ")[0] == []  # blank AIR ignored


class _StrengthMapNode(CivitaiRecipeNodeBase):
    FIELDS = {"loras": F("loras", "lora_strength_map")}


def test_lora_strength_map_serialization():
    loras = [{"air": "a", "strength": 0.7, "triggerWord": "x"}, {"air": "b", "strength": 1.0}]
    payload = _StrengthMapNode()._build_payload(_client(), {"loras": loras})
    assert payload == {"loras": {"a": 0.7, "b": 1.0}}  # dict-of-strength, triggerWord dropped
