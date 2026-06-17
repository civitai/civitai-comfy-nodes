// Civitai Model Selector UX: when its `path` output is wired into a standard loader's file widget
// (ckpt_name, lora_name, …), ComfyUI's frontend statically validates that widget's stored value
// against the local file list and reddens the node with "missing value" — even though the real
// value arrives from the link at run time (the backend uses the link, so it runs fine). Set a
// harmless placeholder on the fed widget so the static check passes; the link still overrides it.
import { app } from "../../scripts/app.js";

const PLACEHOLDER = "⬇ from Civitai (downloaded on run)";
// The selector's download outputs (primary file + extra components), all fed into loader combos.
const DOWNLOAD_OUTPUTS = new Set(["path", "vae", "clip", "clip 2", "clip 3"]);

// The component outputs carry filenames, so they're AnyType and LiteGraph offers them to *every*
// file-name combo (unet_name, clip_name, vae_name are all the same generic COMBO type — the drag
// highlight can't tell them apart). Guard the *completion* of an obviously-wrong wire by the target
// widget's name: e.g. the `vae` output won't connect to a `clip_name`/`unet_name` input. `path` is
// the primary file and stays unrestricted; unrecognized targets are always allowed (never block a
// wire we don't understand).
const OWN_KEYWORDS = {
  vae: ["vae"],
  clip: ["clip", "text_encoder", "textencoder", "encoder", "t5"],
};
const MODEL_KEYWORDS = ["unet", "diffusion", "ckpt", "checkpoint"];

function mismatchRule(outputName) {
  if (outputName === "vae") return { own: OWN_KEYWORDS.vae, foreign: [...OWN_KEYWORDS.clip, ...MODEL_KEYWORDS] };
  if (outputName === "clip" || outputName === "clip 2" || outputName === "clip 3")
    return { own: OWN_KEYWORDS.clip, foreign: [...OWN_KEYWORDS.vae, ...MODEL_KEYWORDS] };
  return null; // path / air -> unrestricted
}

function fedComboWidget(node, slot) {
  const input = node?.inputs?.[slot];
  if (!input) return null;
  const name = input.widget?.name || input.name;
  const widget = name && node.widgets?.find((w) => w.name === name);
  return widget && Array.isArray(widget.options?.values) ? widget : null;
}

app.registerExtension({
  name: "civitai.model-selector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CivitaiModelSelector") return;

    const onConnectOutput = nodeType.prototype.onConnectOutput;
    nodeType.prototype.onConnectOutput = function (outputIndex, inputType, inputObj, targetNode, targetIndex) {
      if (onConnectOutput && onConnectOutput.apply(this, arguments) === false) return false;
      const rule = mismatchRule(this.outputs?.[outputIndex]?.name);
      if (!rule) return true;
      const targetName = (inputObj?.name || targetNode?.inputs?.[targetIndex]?.name || "").toLowerCase();
      if (!targetName) return true; // can't identify the target -> allow
      const own = rule.own.some((k) => targetName.includes(k));
      const foreign = rule.foreign.some((k) => targetName.includes(k));
      if (foreign && !own) {
        const outName = this.outputs[outputIndex].name;
        try {
          app.extensionManager?.toast?.add({
            severity: "warn",
            summary: "Civitai Model Selector",
            detail: `The ${outName} output goes into a ${outName} input, not "${targetName}".`,
            life: 4000,
          });
        } catch {}
        return false;
      }
      return true;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link) {
      onConnectionsChange?.apply(this, arguments);
      const OUTPUT = window.LiteGraph?.OUTPUT ?? 2;
      if (type !== OUTPUT || !link || !DOWNLOAD_OUTPUTS.has(this.outputs?.[index]?.name)) return;
      const target = app.graph?.getNodeById?.(link.target_id);
      const widget = target && fedComboWidget(target, link.target_slot);
      if (!widget) return;
      if (connected) {
        if (!widget.options.values.includes(PLACEHOLDER)) widget.options.values.push(PLACEHOLDER);
        if (!widget.options.values.includes(widget.value)) widget.value = PLACEHOLDER;
      } else if (widget.value === PLACEHOLDER) {
        widget.options.values = widget.options.values.filter((v) => v !== PLACEHOLDER);
        widget.value = widget.options.values[0] ?? "";
      }
      app.graph.setDirtyCanvas(true, true);
    };
  },
});
