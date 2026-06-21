// Civitai settings live in ComfyUI's native Settings dialog (Settings -> Civitai). The values must
// reach the server (orchestrator URL resolution + customComfy offload), so the pack's
// ~/.civitai/comfy-settings.json stays the source of truth: each change POSTs to /civitai/config, and
// on startup we pull that file back into the settings store so the dialog reflects the backend.
import { app } from "../../scripts/app.js";

const IDS = {
  url: "Civitai.orchestratorUrl",
  vram: "Civitai.minVramGb",
  mature: "Civitai.allowMatureContent",
  sage: "Civitai.useSageAttention",
  gpu: "Civitai.gpuGeneration",
};

// Start suppressed: ComfyUI may fire each setting's onChange with its persisted value during init.
// We only begin POSTing after setup() has pulled the server's settings in, so an init callback can't
// clobber ~/.civitai/comfy-settings.json before we've read it.
let suppressPush = true;

function toast(severity, summary, detail) {
  try {
    app.extensionManager.toast.add({ severity, summary, detail, life: 5000 });
  } catch (e) {
    console[severity === "error" ? "error" : "log"](`[civitai-settings] ${summary}: ${detail ?? ""}`);
  }
}

async function pushConfig(payload) {
  if (suppressPush) return;
  try {
    const res = await fetch("/civitai/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || res.status);
  } catch (e) {
    toast("error", "Civitai settings", String(e.message || e));
  }
}

app.registerExtension({
  name: "civitai.settings",
  settings: [
    {
      id: IDS.url,
      name: "Orchestrator URL",
      category: ["Civitai", "Connection", "Orchestrator URL"],
      type: "text",
      defaultValue: "",
      tooltip:
        "Civitai Orchestration endpoint. Empty = https://orchestration.civitai.com. A CIVITAI_ORCHESTRATION_URL env var set on the server overrides this.",
      onChange: (value) => pushConfig({ orchestratorUrl: (value || "").trim() }),
    },
    {
      id: IDS.vram,
      name: "Required VRAM",
      category: ["Civitai", "Offload", "Required VRAM"],
      type: "combo",
      defaultValue: 0,
      options: [
        { text: "Auto", value: 0 },
        { text: "24 GB", value: 24 },
      ],
      tooltip: "Minimum GPU VRAM the offload worker must have. Auto = no requirement.",
      onChange: (value) => pushConfig({ minVramGb: Number(value) || null }),
    },
    {
      id: IDS.mature,
      name: "Allow mature content",
      category: ["Civitai", "Content", "Allow mature content"],
      type: "combo",
      defaultValue: "auto",
      options: [
        { text: "Auto", value: "auto" },
        { text: "On", value: "true" },
        { text: "Off", value: "false" },
      ],
      tooltip: "Whether submitted workflows may return mature content. Auto = account default.",
      onChange: (value) => pushConfig({ allowMatureContent: value }),
    },
    {
      id: IDS.sage,
      name: "Use Sage Attention",
      category: ["Civitai", "Offload", "Use Sage Attention"],
      type: "boolean",
      defaultValue: true,
      tooltip: "Launch the offload worker's ComfyUI with --use-sage-attention.",
      onChange: (value) => pushConfig({ useSageAttention: !!value }),
    },
    {
      id: IDS.gpu,
      name: "GPU generation",
      category: ["Civitai", "Offload", "GPU generation"],
      type: "combo",
      defaultValue: "Ada",
      options: [{ text: "Ada", value: "Ada" }],
      tooltip: "The GPU generation offloaded jobs run on. Informational for now.",
    },
  ],
  async setup() {
    let cfg;
    try {
      cfg = await (await fetch("/civitai/config")).json();
    } catch (e) {
      console.warn("[civitai-settings] could not load config", e);
      return;
    }
    suppressPush = true;
    try {
      app.ui.settings.setSettingValue(IDS.url, cfg.orchestratorUrl || "");
      app.ui.settings.setSettingValue(IDS.vram, cfg.minVramGb || 0);
      app.ui.settings.setSettingValue(IDS.mature, cfg.allowMatureContent || "auto");
      app.ui.settings.setSettingValue(IDS.sage, !!cfg.useSageAttention);
      if (cfg.gpuGeneration) app.ui.settings.setSettingValue(IDS.gpu, cfg.gpuGeneration);
    } finally {
      setTimeout(() => {
        suppressPush = false;
      }, 0);
    }
  },
});
