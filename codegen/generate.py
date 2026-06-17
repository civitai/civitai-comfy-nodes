"""Generate civitai_comfy_nodes/generated/*.py from spec/v2-consumers.json.

Run from the repo root: python -m codegen.generate
Prints a media-detection audit table so spec drift shows up in code review.
"""

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import emit, ir

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "spec" / "v2-consumers.json"
OVERRIDES_PATH = Path(__file__).resolve().parent / "overrides.json"
OUTPUT_DIR = REPO_ROOT / "civitai_comfy_nodes" / "generated"

RECIPE_PATH_RE = re.compile(r"^/v2/consumer/recipes/(\w+)$")

# recipe -> (module, category); every non-skipped recipe must be listed.
MODULES = {
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
    "training": ("training", "Civitai/Training"),
    "imageResourceTraining": ("training", "Civitai/Training"),
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


@dataclass
class _Variant:
    """One concrete node before assembly.

    fixed   : discriminator selections shown as separate nodes (engine=sdcpp, operation=editImage)
              — ordered shallow→deep; they form the node name + DISCRIMINATOR.
    combos  : discriminator selections collapsed to a dropdown because their sibling subtrees were
              structurally identical (model=[gpt-image-1, gpt-image-1.5, ...]) — prop -> (keys, default).
    props   : remaining property schemas (the real input fields).
    required: required property names along this path.
    """

    fixed: list
    combos: dict
    props: dict
    required: set


def build_nodes(spec: dict, overrides: dict) -> list[ir.NodeIR]:
    from collections import Counter

    pending = []  # (recipe, variant, outputs, recipe_overrides, base_name)
    for recipe, input_schema, output_schema in list_recipes(spec, overrides):
        recipe_overrides = overrides.get(recipe, {})
        resolved_input = ir.resolve_ref(spec, input_schema["$ref"]) if "$ref" in input_schema else input_schema
        discriminator = resolved_input.get("discriminator")
        outputs = build_outputs(spec, output_schema, recipe_overrides)

        if discriminator and recipe_overrides.get("expand", True):
            variants = expand_collapse(spec, input_schema, frozenset())
        else:
            props, required, _ = ir.merge_allof_chain(spec, input_schema)
            variants = [_Variant(fixed=[], combos={}, props=props, required=set(required))]

        skip_variants = recipe_overrides.get("skip_variants", [])
        for variant in variants:
            fixed_dict = {p: k for p, k in variant.fixed}
            if any(all(fixed_dict.get(k) == v for k, v in sel.items()) for sel in skip_variants):
                continue
            base = f"Civitai{ir.pascal_case(recipe)}" + "".join(ir.pascal_case(k) for _, k in variant.fixed)
            pending.append((recipe, variant, outputs, recipe_overrides, base))

    # A fixed path is ambiguous when a deeper discriminator collapsed into more than one dropdown
    # group (e.g. fal/qwen2 split into create-ops vs edit-ops); name those by the group's lead op.
    ambiguous = {base for base, n in Counter(p[4] for p in pending).items() if n > 1}

    provider_engines = frozenset(overrides.get("_provider_engines", []))

    # How many engines serve each (recipe, ecosystem) — drives whether the engine is shown as its
    # own menu sub-level (only when an ecosystem is reachable through more than one engine).
    engines_by_eco: dict = {}
    for recipe, variant, _outputs, _ro, _base in pending:
        eco, engine_facet, _rest = _menu_facets(variant.fixed, provider_engines)
        if eco is not None and engine_facet is not None:
            engines_by_eco.setdefault((recipe, eco), set()).add(engine_facet)

    used: set[str] = set()
    nodes = []
    for recipe, variant, outputs, recipe_overrides, base in pending:
        nodes.append(
            assemble_node(
                spec, recipe, variant, outputs, recipe_overrides, base,
                base in ambiguous, used, engines_by_eco, provider_engines,
            )
        )
    return nodes


def _menu_facets(fixed: list, provider_engines: set | frozenset = frozenset()) -> tuple[str | None, str | None, list]:
    """Split a variant's discriminator path into (ecosystem, engine, rest) for menu placement.

    The spec nests engine -> ecosystem (sdcpp/comfy span many ecosystems) while cloud engines
    *are* the ecosystem. To present one consistent ecosystem-first tree:
      - ecosystem = the nested `ecosystem` discriminator if present; else, for a provider/aggregator
        engine (e.g. fal — hosts independent model families), the nested `model` value; else the
        top discriminator value (the engine is its own ecosystem, e.g. openai/flux2).
      - engine    = the `engine` value, surfaced only when a distinct ecosystem level exists.
      - rest      = the remaining deeper facets (model/version/provider/operation), in order.
    """
    if not fixed:
        return None, None, []
    fd = dict(fixed)
    if "ecosystem" in fd:
        rest = [k for p, k in fixed if p not in ("engine", "ecosystem")]
        return fd["ecosystem"], fd.get("engine"), rest
    if fd.get("engine") in provider_engines and "model" in fd:
        rest = [k for p, k in fixed if p not in ("engine", "model")]
        return fd["model"], fd["engine"], rest
    top_prop, top_key = fixed[0]
    return top_key, None, [k for _p, k in fixed[1:]]


def expand_collapse(spec: dict, ref_or_schema, resolved: frozenset) -> list[_Variant]:
    """Recursively expand every discriminator into separate variants, then collapse a level back
    into a dropdown when all its sibling subtrees are structurally identical (smart split)."""
    props, required, discs = ir.merge_allof_chain(spec, ref_or_schema)
    pending = [(p, m) for (p, m) in discs if p not in resolved]
    if not pending:
        return [_Variant(fixed=[], combos={}, props=props, required=set(required))]

    prop, mapping = pending[0]
    children = {
        key: expand_collapse(spec, ir.resolve_ref(spec, sub_ref), resolved | {prop}) for key, sub_ref in mapping.items()
    }
    default_key = _discriminator_default(spec, props.get(prop))

    # Group sibling keys by the structural signature of their subtree (ignoring this prop's value).
    groups: dict = {}
    for key, child_variants in children.items():
        groups.setdefault(_subtree_signature(spec, child_variants, prop), []).append(key)

    result: list[_Variant] = []
    for keys in groups.values():
        representative = children[keys[0]]
        if len(keys) >= 2 and prop != "engine":
            combo_default = default_key if default_key in keys else keys[0]
            for v in representative:
                result.append(
                    _Variant(
                        fixed=list(v.fixed),
                        combos={prop: (keys, combo_default), **v.combos},
                        props=dict(v.props),
                        required=set(v.required),
                    )
                )
        else:
            for key in keys:
                for v in children[key]:
                    result.append(
                        _Variant(
                            fixed=[(prop, key), *v.fixed],
                            combos=dict(v.combos),
                            props=dict(v.props),
                            required=set(v.required),
                        )
                    )
    return result


def _discriminator_default(spec: dict, prop_schema: dict | None) -> str | None:
    if not prop_schema:
        return None
    return ir.deref_property(spec, prop_schema).get("default")


def _subtree_signature(spec: dict, variants: list[_Variant], exclude_prop: str):
    """A hashable canonical form of a list of variants, ignoring `exclude_prop` — two subtrees with
    the same signature render to the same node(s) and so can be merged into one dropdown."""
    items = []
    for v in variants:
        decided = {p for p, _ in v.fixed} | set(v.combos) | {exclude_prop}
        fields = sorted(
            _field_signature(ir.FieldIR(*_field_tuple(spec, name, schema)))
            for name, schema in v.props.items()
            if name not in decided
        )
        combos = tuple(sorted((p, tuple(keys)) for p, (keys, _d) in v.combos.items()))
        fixed = tuple(v.fixed)
        req = tuple(sorted(r for r in v.required if r != exclude_prop))
        items.append((fixed, combos, req, tuple(fields)))
    return tuple(sorted(items, key=repr))


def _field_signature(field: ir.FieldIR):
    comfy = tuple(field.comfy_type) if isinstance(field.comfy_type, list) else field.comfy_type
    opts = tuple(sorted((k, repr(val)) for k, val in field.options.items() if k != "tooltip"))
    return (field.widget, field.kind, comfy, opts, field.required)


def _field_tuple(spec: dict, prop_name: str, prop_schema: dict, hint: str | None = None, required: bool = False):
    """Shared field derivation (used for both signatures and emitted nodes)."""
    schema = ir.deref_property(spec, prop_schema)
    kind, comfy_type, detected = ir.classify_input_field(prop_name, schema, hint)
    widget = ir.snake_case(prop_name) + ("_json" if kind == "json" else "")
    options = ir.widget_options(prop_name, schema, comfy_type, required)
    if isinstance(comfy_type, list) and not required and "default" not in options:
        comfy_type = ["", *comfy_type]
        options["default"] = ""
    if kind == "json":
        options["multiline"] = True
        options.setdefault("default", "")
    return widget, prop_name, kind, comfy_type, options, required, detected


def build_outputs(spec: dict, output_schema: dict, recipe_overrides: dict) -> list[ir.OutputIR]:
    skip_outputs = set(recipe_overrides.get("skip_outputs", []))
    resolved = ir.resolve_ref(spec, output_schema["$ref"]) if "$ref" in output_schema else output_schema
    out_props, _, _ = ir.merge_allof_chain(spec, resolved)
    return [
        ir.classify_output_field(spec, name, schema) for name, schema in out_props.items() if name not in skip_outputs
    ]


def assemble_node(
    spec: dict,
    recipe: str,
    variant: _Variant,
    outputs: list[ir.OutputIR],
    recipe_overrides: dict,
    base_name: str,
    ambiguous: bool,
    used: set,
    engines_by_eco: dict,
    provider_engines: frozenset = frozenset(),
) -> ir.NodeIR:
    module, base_category = MODULES[recipe]
    display_overrides = recipe_overrides.get("display", {})
    field_hints = recipe_overrides.get("field_types", {})
    skip_fields = set(recipe_overrides.get("skip_fields", []))
    decided = {p for p, _ in variant.fixed} | set(variant.combos)

    fixed_dict = {p: key for p, key in variant.fixed}

    def disp(key):
        return display_overrides.get(key, key)

    eco_key, engine_facet, rest_keys = _menu_facets(variant.fixed, provider_engines)
    show_engine = engine_facet is not None and len(engines_by_eco.get((recipe, eco_key), ())) > 1

    path_labels = []
    if eco_key is not None:
        path_labels.append(disp(eco_key))
        if show_engine:
            path_labels.append(disp(engine_facet))
        path_labels.extend(disp(k) for k in rest_keys)

    combo_reps = [keys[0] for keys, _default in variant.combos.values()]

    class_name = base_name
    if ambiguous and combo_reps:
        class_name = base_name + "".join(ir.pascal_case(k) for k in combo_reps)
        path_labels = path_labels + [disp(k) for k in combo_reps]
    suffix = 2
    while class_name in used:
        class_name = f"{base_name}{suffix}"
        suffix += 1
    used.add(class_name)

    # Variant nodes are titled by their discriminator path (ecosystem-first), which doubles as the
    # category submenu (Civitai/Image/<ecosystem>[/<engine>]). Non-discriminated nodes keep a
    # descriptive name (e.g. "Civitai Image Upscaler").
    display_name = " / ".join(path_labels) if path_labels else f"Civitai {ir.title_case(recipe)}"
    if eco_key is None:
        category = base_category
    else:
        category = f"{base_category}/{disp(eco_key)}" + (f"/{disp(engine_facet)}" if show_engine else "")

    node = ir.NodeIR(
        class_name=class_name,
        display_name=display_name,
        recipe=recipe,
        step_type=recipe,
        discriminator=fixed_dict,
        category=category,
        module=module,
        description=f"{recipe} recipe via Civitai Orchestration",
    )

    # Collapsed discriminators become required dropdowns whose value flows into the payload like any field.
    for prop, (keys, default) in variant.combos.items():
        node.fields.append(
            ir.FieldIR(
                widget=ir.snake_case(prop),
                api=prop,
                kind="value",
                comfy_type=list(keys),
                options={"default": default} if default in keys else {},
                required=True,
                detected_as="discriminator-combo",
            )
        )

    for prop_name, prop_schema in variant.props.items():
        if prop_name in decided or prop_name in skip_fields:
            continue
        tup = _field_tuple(spec, prop_name, prop_schema, field_hints.get(prop_name), prop_name in variant.required)
        node.fields.append(ir.FieldIR(*tup))

    node.outputs = list(outputs)
    return node


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
