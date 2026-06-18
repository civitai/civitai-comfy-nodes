// Adds a "Run in Civitai" action that sends the current ComfyUI API prompt to the pack's
// /civitai/offload/run route. The backend handles OAuth, local model AIR lookup, and nodepack AIRs.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let stylesInjected = false;

function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .cvo-run {
      border: 1px solid var(--border-color, #3f3f46);
      background: var(--comfy-input-bg, #27272a);
      color: var(--input-text, #e4e4e7);
      border-radius: 6px;
      padding: 6px 10px;
      font: inherit;
      cursor: pointer;
      white-space: nowrap;
    }
    .cvo-run:hover { border-color: #2563eb; }
    .cvo-run[disabled] { opacity: .6; cursor: progress; }
    .cvo-run-inline { height: 32px; margin-left: 6px; }
    .cvo-run-menu {
      width: 100%;
      text-align: left;
    }
    .cvo-run-menu[disabled] { opacity: .6; cursor: progress; }
  `;
  document.head.appendChild(style);
}

function toast(severity, summary, detail) {
  try {
    app.extensionManager.toast.add({ severity, summary, detail, life: 5000 });
  } catch (e) {
    console[severity === "error" ? "error" : "log"](`[civitai-offload] ${summary}: ${detail ?? ""}`);
  }
}

function selectedNodeIds() {
  return Object.values(app.canvas?.selected_nodes || {}).map((node) => String(node.id));
}

async function currentPrompt() {
  if (!app.graphToPrompt) throw new Error("This ComfyUI frontend does not expose graphToPrompt()");
  const graph = await app.graphToPrompt();
  const prompt = graph?.output || graph?.prompt || graph;
  if (!prompt || typeof prompt !== "object") throw new Error("Could not serialize the current workflow as an API prompt");
  return { prompt, workflow: graph?.workflow || app.graph?.serialize?.() };
}

async function runInCivitai(button) {
  button.disabled = true;
  const oldText = button.textContent;
  button.textContent = "Submitting...";
  try {
    const payload = await currentPrompt();
    payload.selectedNodeIds = selectedNodeIds();
    payload.wait = 5;
    payload.runLocalTail = true;
    // Replay the remote run's /ws frames (progress + previews) onto this tab's canvas.
    payload.liveProgress = true;
    payload.clientId = api.clientId;
    const res = await fetch("/civitai/offload/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (res.status === 401 || data.error === "auth_required") {
      throw new Error("Connect Civitai OAuth or save an API key first");
    }
    if (!res.ok || data.error) throw new Error(data.error || `request failed (${res.status})`);
    const id = data.workflow?.id || data.workflow?.workflowId || "submitted";
    const local = data.local?.queue?.prompt_id ? ` Local continuation ${data.local.queue.prompt_id} queued.` : "";
    const warnings = data.offload?.warnings?.length ? ` ${data.offload.warnings.join(" ")}` : "";
    toast("success", "Submitted to Civitai", `${id}.${local}${warnings}`);
  } catch (e) {
    toast("error", "Civitai offload failed", String(e.message || e));
  } finally {
    button.disabled = false;
    button.textContent = oldText;
  }
}

function findQueueButton() {
  const byId = document.querySelector(
    '#queue-button, .comfyui-button-queue, [data-testid="queue-button"]'
  );
  if (byId) return byId;
  const buttons = [...document.querySelectorAll("button")];
  return buttons.find((button) => /queue|run/i.test(button.textContent || ""));
}

function installRunButton() {
  if (document.querySelector(".cvo-run-inline")) return true;
  const queue = findQueueButton();
  if (!queue || !queue.parentElement) return false;
  const button = document.createElement("button");
  button.className = "cvo-run cvo-run-inline";
  button.type = "button";
  button.title = "Submit this workflow through Civitai customComfy offload";
  button.textContent = "Run in Civitai";
  button.addEventListener("click", () => runInCivitai(button));
  const group = queue.closest(".queue-button-group") || queue.parentElement;
  group.insertAdjacentElement("afterend", button);
  return true;
}

function normalizedText(el) {
  return (el.textContent || "").replace(/\s+/g, " ").trim();
}

function findOpenRunMenu() {
  const runModeButtons = [...document.querySelectorAll("button")].filter((button) => {
    const text = normalizedText(button);
    return text === "Run" || text === "Run (On Change)" || text === "Run (Instant)";
  });
  for (const button of runModeButtons) {
    let el = button.parentElement;
    while (el && el !== document.body) {
      const texts = [...el.querySelectorAll("button")].map(normalizedText);
      if (texts.includes("Run") && texts.some((text) => text === "Run (On Change)" || text === "Run (Instant)")) {
        return el;
      }
      el = el.parentElement;
    }
  }
  return null;
}

function installDropdownItem() {
  const menu = findOpenRunMenu();
  if (!menu || menu.querySelector(".cvo-run-menu")) return false;
  const reference =
    [...menu.querySelectorAll("button")].find((item) => normalizedText(item) === "Run (Instant)") ||
    menu.querySelector("button");
  const button = document.createElement("button");
  button.className = reference?.className || "cvo-run-menu";
  button.classList.add("cvo-run-menu");
  button.type = "button";
  button.textContent = "Run in Civitai";
  button.title = "Submit this workflow through Civitai customComfy offload";
  if (reference) {
    const style = window.getComputedStyle(reference);
    button.style.fontFamily = style.fontFamily;
    button.style.fontSize = style.fontSize;
    button.style.fontWeight = style.fontWeight;
    button.style.lineHeight = style.lineHeight;
    button.style.padding = style.padding;
    button.style.minHeight = style.minHeight;
  }
  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    runInCivitai(button);
  });
  menu.appendChild(button);
  return true;
}

app.registerExtension({
  name: "civitai.offload",
  async setup() {
    injectStyles();
    let tries = 0;
    const timer = setInterval(() => {
      tries += 1;
      if (installRunButton() || tries > 40) clearInterval(timer);
    }, 500);
    const observer = new MutationObserver(() => installDropdownItem());
    observer.observe(document.body, { childList: true, subtree: true });
    document.addEventListener("click", () => setTimeout(installDropdownItem, 0), true);
  },
  nodeCreated(node) {
    if (node.comfyClass === "CivitaiOffloadStart") node.color = "#1d4ed8";
    if (node.comfyClass === "CivitaiOffloadEnd") node.color = "#0f766e";
  },
});
