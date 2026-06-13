"""Parse the consumer OpenAPI spec into node definitions (intermediate representation).

Spec realities this parser is built around:
- Recipe endpoints live at /v2/consumer/recipes/{name}; request/response schemas are $refs.
- Discriminated inputs are recursive: a variant schema is allOf[$ref parent, inline props],
  and inline parts may carry their own discriminator (engine -> version -> provider -> ...).
  The top-level discriminator is expanded into one node per variant; nested ones are
  flattened into COMBO widgets plus the union of all subtree fields.
- Nullability is JSON Schema 2020-12 style: "type": ["null", "string"].
"""

import re
from dataclasses import dataclass, field

PROMPT_LIKE_NAMES = {
    "prompt",
    "negativeprompt",
    "lyrics",
    "text",
    "message",
    "musicdescription",
    "custominstructions",
    "context",
    "systemprompt",
}
IMAGE_FIELD_NAMES = {
    "image",
    "sourceimage",
    "endimage",
    "endsourceimage",
    "startimage",
    "mask",
    "maskimage",
    "frontalimage",
    "cover",
}
SEED_MAX_DEFAULT = 4294967295


@dataclass
class FieldIR:
    widget: str
    api: str
    kind: str  # value | json | image_inline | image_list | image_url | video_url | audio_url
    comfy_type: str | list  # "STRING"/"INT"/... or list of combo options
    options: dict
    required: bool
    detected_as: str = ""  # audit-table annotation


@dataclass
class OutputIR:
    api: str
    kind: str  # image | image_list | video | audio | audio_or_video | string | json


@dataclass
class NodeIR:
    class_name: str
    display_name: str
    recipe: str
    step_type: str
    discriminator: dict
    category: str
    module: str
    description: str
    fields: list[FieldIR] = field(default_factory=list)
    outputs: list[OutputIR] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def snake_case(name: str) -> str:
    name = re.sub(r"[.\-]", "_", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"__+", "_", name).lower()


def pascal_case(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[.\-_ ]+", name) if part)


def title_case(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return spaced[:1].upper() + spaced[1:]


def resolve_ref(spec: dict, ref: str) -> dict:
    assert ref.startswith("#/"), f"unsupported $ref: {ref}"
    node = spec
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def deref_property(spec: dict, schema: dict, depth: int = 0) -> dict:
    """Inline $ref / single-allOf / nullable-anyOf property wrappers so enums and
    scalar refs classify correctly; outer description/default win over the target's."""
    if depth > 5 or not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        inner = deref_property(spec, resolve_ref(spec, schema["$ref"]), depth + 1)
        outer = {k: v for k, v in schema.items() if k != "$ref"}
        return {**inner, **outer}
    if "allOf" in schema and len(schema["allOf"]) == 1:
        inner = deref_property(spec, schema["allOf"][0], depth + 1)
        outer = {k: v for k, v in schema.items() if k != "allOf"}
        return {**inner, **outer}
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            inner = deref_property(spec, non_null[0], depth + 1)
            outer = {k: v for k, v in schema.items() if k != "anyOf"}
            return {**inner, **outer}
    return schema


def unwrap_nullable(schema: dict) -> tuple[dict, bool]:
    """Normalize "type": ["null", X] to "type": X; returns (schema, was_nullable)."""
    type_value = schema.get("type")
    if isinstance(type_value, list):
        non_null = [t for t in type_value if t != "null"]
        nullable = "null" in type_value
        schema = {**schema, "type": non_null[0] if len(non_null) == 1 else non_null}
        return schema, nullable
    return schema, False


def merge_allof_chain(spec: dict, schema: dict) -> tuple[dict, set, list]:
    """Flatten a schema's allOf/$ref chain into (properties, required, discriminators).

    discriminators is a list of (propertyName, mapping) found anywhere along the chain,
    outermost (root parent) first.
    """
    properties: dict = {}
    required: set = set()
    discriminators: list = []

    def walk(node: dict) -> None:
        if "$ref" in node:
            walk(resolve_ref(spec, node["$ref"]))
            return
        for entry in node.get("allOf", []):
            walk(entry)
        if node.get("discriminator"):
            disc = node["discriminator"]
            discriminators.append((disc["propertyName"], disc.get("mapping", {})))
        for name, prop in (node.get("properties") or {}).items():
            properties[name] = _merge_property(properties.get(name), prop)
        required.update(node.get("required", []))

    walk(schema)
    return properties, required, discriminators


def _merge_property(existing: dict | None, new: dict) -> dict:
    """Union two declarations of the same property: keep first description/default, widen numeric bounds."""
    if existing is None:
        return new
    merged = {**new, **{k: v for k, v in existing.items() if v is not None}}
    for bound, pick in (("minimum", min), ("maximum", max)):
        values = [s[bound] for s in (existing, new) if bound in s]
        if values:
            merged[bound] = pick(values)
    return merged


def classify_input_field(name: str, schema: dict, hint: str | None) -> tuple[str, str | list, str]:
    """Map a property schema to (field_kind, comfy_type, detection_note)."""
    schema, _ = unwrap_nullable(schema)
    description = schema.get("description", "")
    type_value = schema.get("type")
    lower = name.lower()

    if hint:
        hinted = {
            "image": ("image_inline", "IMAGE"),
            "image_list": ("image_list", "IMAGE"),
            "image_url": ("image_url", "IMAGE"),
            "video": ("video_url", "VIDEO"),
            "audio": ("audio_url", "AUDIO"),
            "json": ("json", "STRING"),
            "string": ("value", "STRING"),
        }
        if hint not in hinted:
            raise ValueError(f"Unknown field_types hint '{hint}' for field '{name}'")
        kind, comfy = hinted[hint]
        return kind, comfy, f"override:{hint}"

    # Network/ControlNet lists get dedicated typed sockets fed by the CivitaiLoraLoader /
    # CivitaiControlNet helper nodes, instead of a raw JSON text widget.
    if name in ("loras",):
        return "lora_array", "CIVITAI_LORAS", "network:loras"
    if name == "additionalNetworks":
        return "network_map", "CIVITAI_LORAS", "network:additionalNetworks"
    if name == "controlNets":
        return "controlnet_array", "CIVITAI_CONTROLNETS", "network:controlNets"

    if type_value == "string" and re.search(r"DataURL|Base64", description):
        if re.search(r"image|mask|cover", lower) or re.search(r"image", description, re.I):
            return "image_inline", "IMAGE", "desc:dataurl-image"
        if "video" in lower or re.search(r"video", description, re.I):
            return "video_url", "VIDEO", "desc:dataurl-video"
        if "audio" in lower or re.search(r"audio", description, re.I):
            return "audio_url", "AUDIO", "desc:dataurl-audio"
        return "image_inline", "IMAGE", "desc:dataurl-fallback-image"

    if type_value == "string" and schema.get("format") == "uri":
        if "video" in lower or re.search(r"\bvideo\b", description, re.I):
            return "video_url", "VIDEO", "uri:video"
        if "audio" in lower or re.search(r"\baudio\b", description, re.I):
            return "audio_url", "AUDIO", "uri:audio"
        if "image" in lower or re.search(r"\bimage\b", description, re.I):
            return "image_url", "IMAGE", "uri:image"
        return "value", "STRING", "uri:plain-url"

    if type_value == "string" and lower in IMAGE_FIELD_NAMES:
        return "image_inline", "IMAGE", "name:image"

    if type_value == "array":
        items = schema.get("items", {})
        items_unwrapped, _ = unwrap_nullable(items) if isinstance(items, dict) else ({}, False)
        if items_unwrapped.get("type") == "string" and lower in {"images", "sourceimages", "referenceimages"}:
            return "image_list", "IMAGE", "name:image-array"
        if items_unwrapped.get("type") == "string" and "enum" not in items_unwrapped and "$ref" not in items:
            return "json", "STRING", "array-of-strings:json"
        return "json", "STRING", "array:json"

    if "enum" in schema:
        return "value", list(schema["enum"]), "enum"
    if type_value == "integer":
        return "value", "INT", "int"
    if type_value == "number":
        return "value", "FLOAT", "float"
    if type_value == "boolean":
        return "value", "BOOLEAN", "bool"
    if type_value == "string":
        return "value", "STRING", "string"
    return "json", "STRING", "object:json"


def widget_options(name: str, schema: dict, comfy_type: str | list, required: bool) -> dict:
    schema, _ = unwrap_nullable(schema)
    options: dict = {}
    description = schema.get("description")
    if description:
        options["tooltip"] = " ".join(description.split())

    if isinstance(comfy_type, list):
        default = schema.get("default")
        if default in comfy_type:
            options["default"] = default
        return options

    lower = name.lower()
    if comfy_type == "INT":
        minimum = schema.get("minimum", 0)
        maximum = schema.get("maximum", 2**31 - 1)
        if lower == "seed":
            maximum = schema.get("maximum", SEED_MAX_DEFAULT)
            options["control_after_generate"] = True
        options.update(
            {"default": int(schema.get("default", minimum)), "min": int(minimum), "max": int(maximum), "step": 1}
        )
    elif comfy_type == "FLOAT":
        minimum = schema.get("minimum", 0.0)
        maximum = schema.get("maximum", 2**31 - 1)
        options.update({"default": float(schema.get("default", minimum)), "min": float(minimum), "max": float(maximum)})
        options["step"] = 0.01
    elif comfy_type == "BOOLEAN":
        options["default"] = bool(schema.get("default", False))
    elif comfy_type == "STRING":
        options["default"] = schema.get("default") or ""
        if lower in PROMPT_LIKE_NAMES:
            options["multiline"] = True
    return options


def classify_output_field(spec: dict, name: str, schema: dict) -> OutputIR:
    schema, _ = unwrap_nullable(schema)
    blob_kinds = {"ImageBlob": "image", "VideoBlob": "video", "AudioBlob": "audio", "Blob": "audio_or_video"}

    if "$ref" in schema:
        kind = blob_kinds.get(ref_name(schema["$ref"]))
        return OutputIR(name, kind or "json")
    if schema.get("type") == "array":
        items = schema.get("items", {})
        if isinstance(items, dict) and "$ref" in items:
            item_kind = blob_kinds.get(ref_name(items["$ref"]))
            if item_kind == "image":
                return OutputIR(name, "image_list")
        return OutputIR(name, "json")
    if schema.get("type") in ("string", "integer", "number", "boolean"):
        return OutputIR(name, "string")
    return OutputIR(name, "json")
