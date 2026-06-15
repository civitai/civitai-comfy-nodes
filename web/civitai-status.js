// On-node run status for civitai-comfy-nodes recipe nodes. After a run, shows the Civitai
// workflow id and the Buzz cost — one entry per transaction, so a charge split across currencies
// lists each wallet (e.g. "11 Blue Buzz, 5 Green Buzz") — directly on the node, so no separate
// display node is needed. Fed by the `civitai_status` UI payload returned from base.py run().
import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const WIDGET_NAME = "civitai_status";

function statusWidget(node) {
  let widget = node.widgets?.find((w) => w.name === WIDGET_NAME);
  if (widget) return widget;
  widget = ComfyWidgets["STRING"](node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
  widget.inputEl.readOnly = true;
  widget.inputEl.style.opacity = "0.75";
  widget.inputEl.style.fontSize = "11px";
  widget.inputEl.style.border = "none";
  // Status is a run result, not a graph input — keep it out of the saved workflow.
  widget.serializeValue = () => undefined;
  return widget;
}

function render(node, info) {
  const lines = [];
  if (info.workflow_id) lines.push(`Workflow: ${info.workflow_id}`);
  lines.push(`Cost: ${info.cost || "—"}`);
  const widget = statusWidget(node);
  widget.value = lines.join("\n");
  requestAnimationFrame(() => {
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
    app.graph.setDirtyCanvas(true, false);
  });
}

app.registerExtension({
  name: "Civitai.WorkflowStatus",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData?.category?.startsWith("Civitai")) return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const info = message?.[WIDGET_NAME]?.[0];
      if (info) render(this, info);
    };
  },
});
