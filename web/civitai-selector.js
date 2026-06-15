// Civitai Model Selector UX: when its `path` output is wired into a standard loader's file widget
// (ckpt_name, lora_name, …), ComfyUI's frontend statically validates that widget's stored value
// against the local file list and reddens the node with "missing value" — even though the real
// value arrives from the link at run time (the backend uses the link, so it runs fine). Set a
// harmless placeholder on the fed widget so the static check passes; the link still overrides it.
import { app } from "../../scripts/app.js";

const PLACEHOLDER = "⬇ from Civitai (downloaded on run)";

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
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link) {
      onConnectionsChange?.apply(this, arguments);
      const OUTPUT = window.LiteGraph?.OUTPUT ?? 2;
      if (type !== OUTPUT || !link || this.outputs?.[index]?.name !== "path") return;
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
