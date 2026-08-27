// ~/.civitai/comfy-settings.json is the source of truth (the server reads it); the dialog mirrors it.
import { app } from "../../scripts/app.js";

const IDS = {
  enableLink: "Civitai.enableLink",
  linkUrl: "Civitai.linkUrl",
};

// ComfyUI fires onChange during init with stale persisted values; don't POST until setup() pulled the server's.
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
      id: IDS.enableLink,
      name: "Civitai Link",
      category: ["Civitai", "Features", "Civitai Link"],
      type: "boolean",
      defaultValue: true,
      tooltip:
        "Pair with civitai.com (Civitai sidebar tab) so the site's download button drops models into your model folders. Applies immediately.",
      onChange: (value) => pushConfig({ enableLink: !!value }),
    },
    {
      id: IDS.linkUrl,
      name: "Civitai Link server URL",
      category: ["Civitai", "Connection", "Civitai Link server URL"],
      type: "text",
      defaultValue: "",
      tooltip:
        "Civitai Link relay. Empty = https://link.civitai.com. A CIVITAI_LINK_URL env var set on the server overrides this.",
      onChange: (value) => pushConfig({ linkUrl: (value || "").trim() }),
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
      app.ui.settings.setSettingValue(IDS.enableLink, cfg.enableLink !== false);
      app.ui.settings.setSettingValue(IDS.linkUrl, cfg.linkUrl || "");
    } finally {
      setTimeout(() => {
        suppressPush = false;
      }, 0);
    }
  },
});
