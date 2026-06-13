"""Generate civitai_comfy_nodes/generated/*.py from spec/v2-consumers.json.

Run from the repo root: python -m codegen.generate
Prints a media-detection audit table so spec drift shows up in code review.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from . import emit, ir

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "spec" / "v2-consumers.json"
OVERRIDES_PATH = Path(__file__).resolve().parent / "overrides.json"
OUTPUT_DIR = REPO_ROOT / "civitai_comfy_nodes" / "generated"

RECIPE_PATH_RE = re.compile(r"^/v2/consumer/recipes/(\w+)$")

# recipe -> (module, category); every non-skipped recipe must be listed.
MODULES = {
    "textToImage": ("image", "Civitai/Image"),
    "imageGen": ("image", "Civitai/Image"),
    "imageUpscaler": ("image", "Civitai/Image"),
    "imageBackgroundRemoval": ("image", "Civitai/Image"),
    "videoGen": ("video", "Civitai/Video"),
    "videoUpscaler": ("video", "Civitai/Video"),
    "videoInterpolation": ("video", "Civitai/Video"),
    "videoEnhancement": ("video", "Civitai/Video"),
    "textToSpeech": ("audio", "Civitai/Audio"),
    "transcription": ("audio", "Civitai/Audio"),
    "audioCaptioning": ("audio", "Civitai/Audio"),
    "aceStepAudio": ("audio", "Civitai/Audio"),
    "chatCompletion": ("text", "Civitai/Text"),
    "promptEnhancement": ("text", "Civitai/Text"),
    "mediaCaptioning": ("text", "Civitai/Text"),
    "mediaRating": ("analysis", "Civitai/Analysis"),
    "wdTagging": ("analysis", "Civitai/Analysis"),
    "xGuardModeration": ("analysis", "Civitai/Analysis"),
    "ageClassification": ("analysis", "Civitai/Analysis"),
    "training": ("training", "Civitai/Training"),
    "imageResourceTraining": ("training", "Civitai/Training"),
    "comfy": ("misc", "Civitai/Misc"),
    "customComfy": ("misc", "Civitai/Misc"),
    "echo": ("misc", "Civitai/Misc"),
    "polyGen": ("misc", "Civitai/Misc"),
}


def load_inputs() -> tuple[dict, dict, str]:
    spec = json.loads(SPEC_PATH.read_text())
    overrides = json.loads(OVERRIDES_PATH.read_text())
    spec_sha = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()[:12]
    return spec, overrides, spec_sha


def list_recipes(spec: dict, overrides: dict) -> list[tuple[str, dict, dict]]:
    """Return (recipe_name, input_schema, output_schema) sorted by name, minus skips."""
    skip = set(overrides.get("_skip", []))
    step_types = set(ir.resolve_ref(spec, "#/components/schemas/WorkflowStep")["discriminator"]["mapping"])
    recipes = []
    for path, item in sorted(spec["paths"].items()):
        match = RECIPE_PATH_RE.match(path)
        if not match or match.group(1) in skip:
            continue
        name = match.group(1)
        if name not in step_types:
            raise SystemExit(f"Recipe '{name}' has no WorkflowStep mapping — add it to _skip in overrides.json")
        if name not in MODULES:
            raise SystemExit(f"Recipe '{name}' has no MODULES entry in codegen/generate.py — assign it a module")
        post = item["post"]
        input_schema = post["requestBody"]["content"]["application/json"]["schema"]
        output_schema = post["responses"]["200"]["content"]["application/json"]["schema"]
        recipes.append((name, input_schema, output_schema))
    return recipes


def build_nodes(spec: dict, overrides: dict) -> list[ir.NodeIR]:
    nodes = []
    for recipe, input_schema, output_schema in list_recipes(spec, overrides):
        recipe_overrides = overrides.get(recipe, {})
        resolved_input = ir.resolve_ref(spec, input_schema["$ref"]) if "$ref" in input_schema else input_schema
        discriminator = resolved_input.get("discriminator")

        if discriminator and recipe_overrides.get("expand", True):
            mapping = discriminator["mapping"]
            for variant_key, variant_ref in mapping.items():
                nodes.append(
                    build_node(
                        spec,
                        recipe,
                        output_schema,
                        recipe_overrides,
                        variant_key=variant_key,
                        variant_schema=ir.resolve_ref(spec, variant_ref),
                        disc_prop=discriminator["propertyName"],
                    )
                )
        else:
            nodes.append(build_node(spec, recipe, output_schema, recipe_overrides))
    return nodes


def build_node(
    spec: dict,
    recipe: str,
    output_schema: dict,
    recipe_overrides: dict,
    *,
    variant_key: str | None = None,
    variant_schema: dict | None = None,
    disc_prop: str | None = None,
) -> ir.NodeIR:
    module, category = MODULES[recipe]
    display_overrides = recipe_overrides.get("display", {})
    field_hints = recipe_overrides.get("field_types", {})
    skip_fields = set(recipe_overrides.get("skip_fields", []))
    skip_outputs = set(recipe_overrides.get("skip_outputs", []))

    if variant_key is not None:
        properties, required, combos, warnings = ir.flatten_variant(spec, variant_schema, disc_prop)
        properties.pop(disc_prop, None)
        variant_display = display_overrides.get(variant_key, ir.pascal_case(variant_key))
        class_name = f"Civitai{ir.pascal_case(recipe)}{ir.pascal_case(variant_key)}"
        display_name = f"Civitai {ir.title_case(recipe)} ({variant_display})"
        node_discriminator = {disc_prop: variant_key}
    else:
        properties, required, _discs = ir.merge_allof_chain(spec, _recipe_input_schema(spec, recipe))
        combos, warnings = {}, []
        class_name = f"Civitai{ir.pascal_case(recipe)}"
        display_name = f"Civitai {ir.title_case(recipe)}"
        node_discriminator = {}

    node = ir.NodeIR(
        class_name=class_name,
        display_name=display_name,
        recipe=recipe,
        step_type=recipe,
        discriminator=node_discriminator,
        category=category,
        module=module,
        description=f"{recipe} recipe via Civitai Orchestration",
        warnings=warnings,
    )

    for prop_name, prop_schema in properties.items():
        if prop_name in skip_fields or prop_name in node_discriminator:
            continue
        prop_schema = ir.deref_property(spec, prop_schema)
        if prop_name in combos:
            node.fields.append(_combo_field(prop_name, prop_schema, combos[prop_name], prop_name in required))
            continue
        kind, comfy_type, detected = ir.classify_input_field(prop_name, prop_schema, field_hints.get(prop_name))
        widget = ir.snake_case(prop_name) + ("_json" if kind == "json" else "")
        options = ir.widget_options(prop_name, prop_schema, comfy_type, prop_name in required)
        if isinstance(comfy_type, list) and prop_name not in required and "default" not in options:
            # optional enum without a spec default: let the server default by omitting ""
            comfy_type = ["", *comfy_type]
            options["default"] = ""
        if kind == "json":
            options["multiline"] = True
            options.setdefault("default", "")
        node.fields.append(
            ir.FieldIR(
                widget=widget,
                api=prop_name,
                kind=kind,
                comfy_type=comfy_type,
                options=options,
                required=prop_name in required,
                detected_as=detected,
            )
        )

    out_resolved = ir.resolve_ref(spec, output_schema["$ref"]) if "$ref" in output_schema else output_schema
    out_props, _, _ = ir.merge_allof_chain(spec, out_resolved)
    for prop_name, prop_schema in out_props.items():
        if prop_name in skip_outputs:
            continue
        node.outputs.append(ir.classify_output_field(spec, prop_name, prop_schema))

    return node


def _combo_field(prop_name: str, prop_schema: dict, options: list, required: bool) -> ir.FieldIR:
    """A nested-discriminator property rendered as a dropdown; optional ones get an omit ('') choice."""
    schema, _ = ir.unwrap_nullable(prop_schema)
    choices = list(options)
    default = schema.get("default")
    if not required and "" not in choices:
        choices = ["", *choices]
    widget_opts: dict = {}
    if schema.get("description"):
        widget_opts["tooltip"] = " ".join(schema["description"].split())
    if default in choices:
        widget_opts["default"] = default
    return ir.FieldIR(
        widget=ir.snake_case(prop_name),
        api=prop_name,
        kind="value",
        comfy_type=choices,
        options=widget_opts,
        required=required,
        detected_as="nested-discriminator",
    )


def _recipe_input_schema(spec: dict, recipe: str) -> dict:
    post = spec["paths"][f"/v2/consumer/recipes/{recipe}"]["post"]
    return post["requestBody"]["content"]["application/json"]["schema"]


def print_audit(nodes: list[ir.NodeIR]) -> None:
    print(f"\nGenerated {len(nodes)} nodes\n")
    print("MEDIA / SPECIAL FIELD AUDIT (field -> detection rule)")
    for node in nodes:
        special = [f for f in node.fields if f.kind != "value" or f.detected_as in ("nested-discriminator",)]
        if not special:
            continue
        print(f"  {node.class_name} [{node.recipe}]")
        for f in special:
            type_label = f.comfy_type if isinstance(f.comfy_type, str) else "COMBO"
            print(f"    {f.api:<24} kind={f.kind:<13} type={type_label:<7} via {f.detected_as}")
    print("\nOUTPUTS")
    seen = set()
    for node in nodes:
        signature = (node.recipe, tuple((o.api, o.kind) for o in node.outputs))
        if signature in seen:
            continue
        seen.add(signature)
        outs = ", ".join(f"{o.api}:{o.kind}" for o in node.outputs)
        print(f"  {node.recipe:<24} {outs}")


def main() -> None:
    spec, overrides, spec_sha = load_inputs()
    nodes = build_nodes(spec, overrides)
    emit.write_modules(nodes, OUTPUT_DIR, spec_sha)
    try:
        subprocess.run([sys.executable, "-m", "ruff", "format", "-q", str(OUTPUT_DIR)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("WARNING: ruff format skipped (not installed?) — generated files are unformatted", file=sys.stderr)
    print_audit(nodes)
    for node in nodes:
        for warning in node.warnings:
            print(f"WARNING {node.class_name}: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
