// Adds a "Run on Civitai" action that sends the current ComfyUI API prompt to the pack's
// /civitai/offload/run route. The backend handles OAuth, local model AIR lookup, and nodepack AIRs.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let stylesInjected = false;
let civitaiRunMode = false;
let civitaiRunInProgress = false;
let activeOffloadPromise = null;
let activeOffloadWorkflowId = null;

const NATIVE_RUN_LABELS = new Set(["Run", "Run (On Change)", "Run (Instant)"]);
const CIVITAI_MENU_LABEL = "Run on Civitai";
const CIVITAI_BUTTON_LABEL = "Run on Civitai";
const SUBMITTING_LABEL = "Submitting...";
const QUEUE_PROMPT_PATCH = "__civitaiOffloadQueuePrompt";
const INTERRUPT_PATCH = "__civitaiOffloadInterrupt";
const MAX_SAFE_SEED = Number.MAX_SAFE_INTEGER;
const TERMS_STORAGE_KEY = "civitai.offload.billingTermsAccepted.v1";

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
    .cvo-run-menu {
      width: 100%;
      text-align: left;
    }
    .cvo-run-menu[disabled] { opacity: .6; cursor: progress; }
    .cvo-modal-overlay {
      position: fixed;
      inset: 0;
      z-index: 11000;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0, 0, 0, .6);
      padding: 20px;
    }
    .cvo-modal {
      background: var(--comfy-menu-bg, #202020);
      color: var(--input-text, #e4e4e7);
      border: 1px solid var(--border-color, #3f3f46);
      border-radius: 10px;
      max-width: 520px;
      width: 100%;
      padding: 20px 22px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, .5);
      font: inherit;
    }
    .cvo-modal-title { margin: 0 0 10px; font-size: 1.15em; font-weight: 600; }
    .cvo-modal-intro { margin: 0 0 12px; }
    .cvo-modal-list {
      margin: 0 0 12px;
      padding-left: 18px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .cvo-modal-list li { line-height: 1.45; }
    .cvo-modal-note { margin: 0 0 16px; opacity: .7; font-size: .9em; }
    .cvo-modal-actions { display: flex; justify-content: flex-end; gap: 10px; }
    .cvo-modal-btn {
      border: 1px solid var(--border-color, #3f3f46);
      background: var(--comfy-input-bg, #27272a);
      color: var(--input-text, #e4e4e7);
      border-radius: 6px;
      padding: 7px 14px;
      font: inherit;
      cursor: pointer;
    }
    .cvo-modal-btn:hover { border-color: #2563eb; }
    .cvo-modal-accept { background: #2563eb; border-color: #2563eb; color: #fff; }
    .cvo-modal-accept:hover { background: #1d4ed8; border-color: #1d4ed8; }
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

async function currentPrompt() {
  if (!app.graphToPrompt) throw new Error("This ComfyUI frontend does not expose graphToPrompt()");
  const graph = await app.graphToPrompt();
  return promptPayloadFromGraph(graph);
}

function selectedNodeIds() {
  return Object.values(app.canvas?.selected_nodes || {}).map((node) => String(node.id));
}

function promptPayloadFromGraph(graph) {
  const prompt = graph?.output || graph?.prompt || graph;
  if (!looksLikeApiPrompt(prompt)) throw new Error("Could not serialize the current workflow as an API prompt");
  return { prompt, workflow: graph?.workflow || app.graph?.serialize?.() };
}

function randomSeed() {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi?.getRandomValues) {
    const values = new Uint32Array(2);
    cryptoApi.getRandomValues(values);
    return ((values[0] & 0x1fffff) * 0x100000000 + values[1]) % MAX_SAFE_SEED;
  }
  return Math.floor(Math.random() * MAX_SAFE_SEED);
}

function serializedWorkflowNodes(workflow) {
  if (!workflow || typeof workflow !== "object") return [];
  if (Array.isArray(workflow.nodes)) return workflow.nodes;
  if (workflow.workflow && typeof workflow.workflow === "object") return serializedWorkflowNodes(workflow.workflow);
  return [];
}

function syncGraphSeed(nodeId, seed) {
  const graphNode = app.graph?.getNodeById?.(Number(nodeId)) || app.graph?.getNodeById?.(nodeId);
  const widget = graphNode?.widgets?.find((item) => item?.name === "seed") || graphNode?.widgets?.[0];
  if (widget && typeof widget.value !== "undefined") widget.value = seed;
}

function applySeedControls(payload) {
  const prompt = payload?.prompt;
  if (!looksLikeApiPrompt(prompt)) return payload;
  const nodesById = new Map(serializedWorkflowNodes(payload.workflow).map((node) => [String(node.id), node]));
  for (const [nodeId, node] of Object.entries(prompt)) {
    const inputs = node?.inputs;
    if (!inputs || !Object.prototype.hasOwnProperty.call(inputs, "seed")) continue;
    const workflowNode = nodesById.get(String(nodeId));
    const widgets = workflowNode?.widgets_values;
    if (!Array.isArray(widgets) || widgets[1] !== "randomize") continue;
    const seed = randomSeed();
    inputs.seed = seed;
    widgets[0] = seed;
    syncGraphSeed(nodeId, seed);
  }
  return payload;
}

function looksLikeApiPrompt(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  return Object.values(value).some(
    (node) => node && typeof node === "object" && typeof node.class_type === "string" && node.inputs && typeof node.inputs === "object"
  );
}

async function submitOffload(payload) {
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
  return data;
}

function offloadQueueResult(data, number) {
  // The workflow id is the prompt_id the server keys the running row, events and history on, so don't
  // fabricate a fallback that would desync them.
  const id = data?.workflow?.id || data?.workflow?.workflowId;
  if (!id) return { prompt_id: null, number: number || 0, node_errors: {} };
  return { prompt_id: String(id), number: number || 0, node_errors: {} };
}

function hasAcceptedBillingTerms() {
  try {
    return localStorage.getItem(TERMS_STORAGE_KEY) === "true";
  } catch (e) {
    return false;
  }
}

function setBillingTermsAccepted() {
  try {
    localStorage.setItem(TERMS_STORAGE_KEY, "true");
  } catch (e) {
    // localStorage unavailable (private mode / blocked) — re-prompt next run rather than fail.
  }
}

function showBillingTermsDialog() {
  injectStyles();
  return new Promise((resolve) => {
    let settled = false;
    const overlay = document.createElement("div");
    overlay.className = "cvo-modal-overlay";
    overlay.innerHTML = `
      <div class="cvo-modal" role="dialog" aria-modal="true" aria-labelledby="cvo-terms-title">
        <h2 id="cvo-terms-title" class="cvo-modal-title">Run on Civitai — billing terms</h2>
        <p class="cvo-modal-intro">Running this workflow on Civitai spends Buzz. Before your first run, please acknowledge:</p>
        <ul class="cvo-modal-list">
          <li><strong>You're billed for compute time, not results.</strong> Buzz is charged for the full duration your workflow runs on Civitai's hardware, regardless of the outcome. The only exception is a system failure on our side — you aren't charged for those.</li>
          <li><strong>Canceling stops the run, but not the bill for time already used.</strong> You can cancel at any time, but Buzz for the compute time spent up to that point is still charged.</li>
          <li><strong>Models and node packs are fetched before the meter starts.</strong> Civitai downloads the required models and custom node packs in advance, before billing begins. But if your workflow uses custom node packs that download assets on the fly, or calls external API nodes, you're billed for the time spent waiting on those during the run.</li>
        </ul>
        <p class="cvo-modal-note">You'll only see this once on this browser.</p>
        <div class="cvo-modal-actions">
          <button type="button" class="cvo-modal-btn cvo-modal-cancel">Cancel</button>
          <button type="button" class="cvo-modal-btn cvo-modal-accept">Accept &amp; Run</button>
        </div>
      </div>`;
    const finish = (accepted) => {
      if (settled) return;
      settled = true;
      document.removeEventListener("keydown", onKey, true);
      overlay.remove();
      resolve(accepted);
    };
    function onKey(event) {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        finish(false);
      }
    }
    overlay.addEventListener("mousedown", (event) => {
      if (event.target === overlay) finish(false);
    });
    overlay.querySelector(".cvo-modal-cancel").addEventListener("click", () => finish(false));
    overlay.querySelector(".cvo-modal-accept").addEventListener("click", () => finish(true));
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(overlay);
    overlay.querySelector(".cvo-modal-accept").focus();
  });
}

async function runInCivitai(button, graph = null, { throwOnError = false } = {}) {
  if (activeOffloadPromise) return activeOffloadPromise;
  if (!hasAcceptedBillingTerms()) {
    const accepted = await showBillingTermsDialog();
    if (!accepted) return null;
    setBillingTermsAccepted();
  }
  civitaiRunInProgress = true;
  const queueButton = button || findQueueButton();
  const oldText = queueButton ? normalizedText(queueButton) : "";
  if (queueButton) {
    queueButton.disabled = true;
    setButtonLabel(queueButton, SUBMITTING_LABEL);
  }
  activeOffloadPromise = (async () => {
    const payload = graph ? promptPayloadFromGraph(graph) : await currentPrompt();
    payload.selectedNodeIds = selectedNodeIds();
    return submitOffload(applySeedControls(payload));
  })();
  try {
    const data = await activeOffloadPromise;
    const id = data.workflow?.id || data.workflow?.workflowId || "submitted";
    // Remember the running workflow so a queue cancel/interrupt can stop it on Civitai (and the bill).
    activeOffloadWorkflowId = data.workflow?.id || data.workflow?.workflowId || null;
    const warnings = data.offload?.warnings?.length ? ` ${data.offload.warnings.join(" ")}` : "";
    toast("success", "Submitted to Civitai", `${id}.${warnings}`);
    return data;
  } catch (e) {
    toast("error", "Civitai offload failed", String(e.message || e));
    if (throwOnError) throw e;
    return null;
  } finally {
    if (queueButton) {
      queueButton.disabled = false;
      setButtonLabel(queueButton, civitaiRunMode ? CIVITAI_BUTTON_LABEL : oldText);
    }
    civitaiRunInProgress = false;
    activeOffloadPromise = null;
  }
}

function findQueueButton() {
  const direct = [...document.querySelectorAll('[data-testid="queue-button"], .comfyui-button-queue, #queue-button')];
  const visible = direct.find(isVisible);
  if (visible) return visible;
  if (direct[0]) return direct[0];
  const buttons = [...document.querySelectorAll("button")];
  return buttons.find(
    (button) =>
      isVisible(button) && (NATIVE_RUN_LABELS.has(normalizedText(button)) || normalizedText(button) === CIVITAI_BUTTON_LABEL)
  );
}

function isVisible(el) {
  return !!(el && (el.offsetParent || el.getClientRects?.().length));
}

function normalizedText(el) {
  return (el.textContent || "").replace(/\s+/g, " ").trim();
}

function findButtonLabelTextNode(button) {
  const walker = document.createTreeWalker(button, NodeFilter.SHOW_TEXT);
  let fallback = null;
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const text = (node.nodeValue || "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    if (
      NATIVE_RUN_LABELS.has(text) ||
      text === CIVITAI_MENU_LABEL ||
      text === CIVITAI_BUTTON_LABEL ||
      text === SUBMITTING_LABEL
    ) {
      return node;
    }
    fallback ||= node;
  }
  return fallback;
}

function setButtonLabel(button, label) {
  if (!button) return;
  const textNode = findButtonLabelTextNode(button);
  if (textNode) {
    if (normalizedText({ textContent: textNode.nodeValue }) === label) return;
    const hasLeadingSpace = /^\s/.test(textNode.nodeValue || "");
    const hasTrailingSpace = /\s$/.test(textNode.nodeValue || "");
    textNode.nodeValue = `${hasLeadingSpace ? " " : ""}${label}${hasTrailingSpace ? " " : ""}`;
  } else {
    button.appendChild(document.createTextNode(label));
  }
}

function setCivitaiRunMode(enabled) {
  civitaiRunMode = enabled;
  const queue = findQueueButton();
  if (queue) {
    queue.dataset.civitaiRunMode = enabled ? "true" : "false";
    if (enabled) setButtonLabel(queue, CIVITAI_BUTTON_LABEL);
  }
  updateOpenMenuState();
}

function buttonFromEvent(event) {
  const target = event.target;
  return target instanceof Element ? target.closest("button") : null;
}

function closeRunMenu() {
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", code: "Escape", bubbles: true }));
}

function updateOpenMenuState() {
  const menu = findOpenRunMenu();
  if (!menu) return;
  const civitaiItem = menu.querySelector(".cvo-run-menu");
  if (!civitaiItem) return;
  civitaiItem.setAttribute("aria-selected", civitaiRunMode ? "true" : "false");
  civitaiItem.classList.toggle("p-highlight", civitaiRunMode);
  civitaiItem.classList.toggle("p-focus", civitaiRunMode);
}

function syncRunModeUi() {
  installDropdownItem();
  const queue = findQueueButton();
  if (queue) queue.dataset.civitaiRunMode = civitaiRunMode ? "true" : "false";
  if (civitaiRunMode && !civitaiRunInProgress) setButtonLabel(queue, CIVITAI_BUTTON_LABEL);
}

function installQueuePromptOverride() {
  if (api[QUEUE_PROMPT_PATCH] || typeof api.queuePrompt !== "function") return;
  const originalQueuePrompt = api.queuePrompt.bind(api);
  api[QUEUE_PROMPT_PATCH] = originalQueuePrompt;
  api.queuePrompt = async function civitaiAwareQueuePrompt(number, graph, options) {
    if (!civitaiRunMode) return originalQueuePrompt(number, graph, options);
    const data = await runInCivitai(findQueueButton(), graph, { throwOnError: true });
    if (!data) return { prompt_id: null, number, node_errors: {} };
    return offloadQueueResult(data, number);
  };
}

function cancelActiveOffload() {
  const id = activeOffloadWorkflowId;
  if (!id) return Promise.resolve();
  return fetch("/civitai/offload/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workflowId: id }),
  }).catch(() => {});
}

// The offload isn't in the local executor, so api.interrupt() (the cancel button) can't stop it —
// route it to the orchestrator cancel too, then run the native interrupt for any local prompt.
function installInterruptOverride() {
  if (api[INTERRUPT_PATCH] || typeof api.interrupt !== "function") return;
  const originalInterrupt = api.interrupt.bind(api);
  api[INTERRUPT_PATCH] = originalInterrupt;
  api.interrupt = async function civitaiAwareInterrupt(...args) {
    if (activeOffloadWorkflowId) await cancelActiveOffload();
    return originalInterrupt(...args);
  };
}

function isQueueButtonClick(button) {
  if (!button) return false;
  if (button.dataset.civitaiRunMode === "true") return true;
  if (normalizedText(button) === CIVITAI_BUTTON_LABEL) return true;
  return button.matches?.('#queue-button, .comfyui-button-queue, [data-testid="queue-button"]') || false;
}

function commitActiveWidgetValue() {
  const active = document.activeElement;
  if (!(active instanceof HTMLElement)) return;
  if (!["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)) return;
  active.dispatchEvent(new Event("input", { bubbles: true }));
  active.dispatchEvent(new Event("change", { bubbles: true }));
  active.blur();
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
  button.textContent = "Run on Civitai";
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
    setCivitaiRunMode(true);
    closeRunMenu();
  });
  menu.appendChild(button);
  updateOpenMenuState();
  return true;
}

async function offloadEnabled() {
  try {
    const cfg = await (await fetch("/civitai/config")).json();
    return cfg.enableOffload !== false;
  } catch (e) {
    return true; // default on if the config route is unavailable
  }
}

app.registerExtension({
  name: "civitai.offload",
  async setup() {
    if (!(await offloadEnabled())) return;
    injectStyles();
    installQueuePromptOverride();
    installInterruptOverride();
    api.addCustomEventListener("civitai.offload.status", (event) => {
      const detail = event.detail || {};
      activeOffloadWorkflowId = null; // offload reached a terminal state; cancel no longer applies
      if (detail.state === "error") {
        toast("error", "Civitai offload failed", String(detail.message || "Unknown error"));
      } else if (detail.state === "done") {
        const wf = detail.workflowId ? ` (${detail.workflowId})` : "";
        toast("success", "Civitai offload complete", `Results downloaded${wf}.`);
      }
    });
    const observer = new MutationObserver(() => syncRunModeUi());
    observer.observe(document.body, { childList: true, subtree: true });
    document.addEventListener(
      "click",
      (event) => {
        const button = buttonFromEvent(event);
        if (!button) {
          setTimeout(syncRunModeUi, 0);
          return;
        }
        const text = normalizedText(button);
        if (button.classList.contains("cvo-run-menu")) return;
        if (civitaiRunMode && isQueueButtonClick(button)) {
          commitActiveWidgetValue();
          setTimeout(syncRunModeUi, 0);
          return;
        }
        if (NATIVE_RUN_LABELS.has(text)) {
          setCivitaiRunMode(false);
          setTimeout(syncRunModeUi, 0);
          return;
        }
        setTimeout(syncRunModeUi, 0);
      },
      true
    );
  },
  nodeCreated(node) {
    if (node.comfyClass === "CivitaiOffloadStart") node.color = "#1d4ed8";
    if (node.comfyClass === "CivitaiOffloadEnd") node.color = "#0f766e";
  },
});
