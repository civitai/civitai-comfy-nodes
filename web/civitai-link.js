import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SESSION_PROXY_RE = /^\/sessions\/[^/]+\/proxy\//;
const LINK_HELP_URL = "https://civitai.com/user/account";
const SHOWN_ACTIVITIES = 5;

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .cvl-panel { border-bottom: 1px solid #27272a; padding: 10px 12px; display: flex; flex-direction: column;
      gap: 8px; color: #e4e4e7; font: 13px system-ui, sans-serif; }
    .cvl-head { display: flex; align-items: center; gap: 8px; }
    .cvl-title { font-weight: 600; }
    .cvl-dot { width: 8px; height: 8px; border-radius: 50%; background: #71717a; flex: none; }
    .cvl-dot.ok { background: #22c55e; } .cvl-dot.warn { background: #f59e0b; } .cvl-dot.err { background: #ef4444; }
    .cvl-status { color: #a1a1aa; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .cvl-row { display: flex; gap: 6px; }
    .cvl-row input { flex: 1; min-width: 0; background: #27272a; color: #e4e4e7; border: 1px solid #3f3f46;
      border-radius: 7px; padding: 6px 9px; font: inherit; letter-spacing: .15em; text-transform: lowercase; outline: none; }
    .cvl-btn { background: #27272a; color: #e4e4e7; border: 1px solid #3f3f46; border-radius: 7px; padding: 5px 9px;
      font: inherit; cursor: pointer; }
    .cvl-btn:hover { border-color: #2563eb; }
    .cvl-btn.primary { background: #2563eb; border-color: #2563eb; color: #fff; }
    .cvl-hint { color: #a1a1aa; font-size: 12px; line-height: 1.4; }
    .cvl-hint a { color: #60a5fa; }
    .cvl-err { color: #f87171; font-size: 12px; }
    .cvl-act { display: flex; flex-direction: column; gap: 2px; }
    .cvl-item { display: flex; flex-direction: column; gap: 4px; font-size: 12px; padding: 5px 6px;
      border-radius: 6px; background: #1f1f23; }
    .cvl-item.done { opacity: .75; }
    .cvl-item-line { display: flex; gap: 7px; align-items: center; }
    .cvl-glyph { flex: none; width: 16px; height: 16px; border-radius: 4px; display: inline-flex; align-items: center;
      justify-content: center; font-size: 11px; font-weight: 700; background: #27272a; color: #d4d4d8; }
    .cvl-glyph.remove { color: #fca5a5; }
    .cvl-item-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .cvl-pill { flex: none; font-size: 10px; font-weight: 600; letter-spacing: .02em; text-transform: uppercase;
      padding: 1px 6px; border-radius: 999px; background: #27272a; color: #a1a1aa; }
    .cvl-pill.processing { background: rgba(37,99,235,.18); color: #93c5fd; }
    .cvl-pill.success { background: rgba(34,197,94,.16); color: #86efac; }
    .cvl-pill.error { background: rgba(239,68,68,.16); color: #fca5a5; }
    .cvl-item-err { color: #f87171; font-size: 11px; white-space: normal; line-height: 1.3; }
    .cvl-cancel { flex: none; width: 18px; height: 18px; border-radius: 4px; border: none; background: transparent;
      color: #a1a1aa; font-size: 13px; line-height: 1; cursor: pointer; padding: 0; }
    .cvl-cancel:hover { background: #3f3f46; color: #fca5a5; }
    .cvl-bar { height: 4px; border-radius: 2px; background: #27272a; overflow: hidden; }
    .cvl-bar > div { height: 100%; background: #2563eb; transition: width .3s; }
    .cvl-rail-badge { position: absolute; right: 6px; top: 6px; width: 11px; height: 11px; border-radius: 50%;
      background: #27272a; pointer-events: none; display: none; }
    .cvl-rail-badge.busy { display: block; background: conic-gradient(#3b82f6 var(--cvl-progress, 0%), #3f3f46 0);
      box-shadow: 0 0 0 2px #18181b; animation: cvl-pulse 1.4s ease-in-out infinite; }
    .cvl-rail-badge.busy::after { content: ""; position: absolute; inset: 2.5px; border-radius: 50%; background: #18181b; }
    .cvl-rail-badge.done { display: block; background: #22c55e; box-shadow: 0 0 0 2px #18181b; }
    .cvl-rail-badge.error { display: block; background: #ef4444; box-shadow: 0 0 0 2px #18181b; }
    @keyframes cvl-pulse { 50% { opacity: .55; } }
  `;
  document.head.appendChild(style);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(severity, summary, detail) {
  try {
    app.extensionManager.toast.add({ severity, summary, detail, life: 5000 });
  } catch (e) {
    console[severity === "error" ? "error" : "log"](`[civitai-link] ${summary}: ${detail ?? ""}`);
  }
}

function isHostedSession() {
  return SESSION_PROXY_RE.test(location.pathname);
}

// Same DOM probe as civitai-catalog.js (module scripts can't share it): the Model Library header
// has no stable test id, so find the refresh button near its search input.
function libraryRefreshButton() {
  const input = [...document.querySelectorAll("input[placeholder]")].find(
    (i) => !i.closest(".cvc-backdrop") && /search.*model|model.*search/i.test(i.placeholder)
  );
  for (let scope = input?.parentElement; scope && scope !== document.body; scope = scope.parentElement) {
    const btn = [...scope.querySelectorAll("button")].find(
      (b) =>
        b.querySelector(".pi-sync, [class*='refresh']") ||
        /refresh/i.test(b.getAttribute("aria-label") || b.title || "")
    );
    if (btn) return btn;
  }
  return null;
}

function refreshModelLists() {
  try {
    app.refreshComboInNodes?.();
  } catch (e) {
    console.warn("[civitai-link] refreshComboInNodes failed", e);
  }
  libraryRefreshButton()?.click();
}

let panelEl = null;
let state = null;
const activities = new Map();
const announced = new Set();

function resourceLabel(r) {
  const res = r.resource || {};
  return res.modelName ? `${res.modelName}${res.modelVersionName ? ` · ${res.modelVersionName}` : ""}` : res.name || r.id;
}

function statusLine() {
  if (!state) return ["", "Loading…"];
  if (!state.available)
    return [
      "err",
      "Needs python-socketio in ComfyUI's own Python: " +
        '<ComfyUI python> -m pip install "python-socketio[client]" — then restart. ' +
        "(Windows portable: python_embeded\\python.exe)",
    ];
  if (!state.enabled) return ["", "Disabled in Settings › Civitai › Civitai Link"];
  if (!state.paired) return ["", "Not paired"];
  if (state.connected && state.joined) return state.roomReady ? ["ok", "Paired · civitai.com connected"] : ["warn", "Paired · open civitai.com to connect"];
  if (state.lastError) return ["err", `Reconnecting… ${state.lastError}`];
  return ["warn", "Connecting…"];
}

const STATE_LABELS = {
  add: { pending: "queued", processing: "", success: "downloaded", error: "failed", canceled: "canceled" },
  remove: { pending: "queued", processing: "removing", success: "removed", error: "failed", canceled: "canceled" },
};

function activityMarkup() {
  const items = [...activities.values()].slice(-SHOWN_ACTIVITIES).reverse();
  if (!items.length) return "";
  return `<div class="cvl-act">${items
    .map((r) => {
      const isRemove = r.type === "resources:remove";
      const kind = isRemove ? "remove" : "add";
      const inFlight = r.status === "processing" || r.status === "pending";
      const pct = r.status === "success" ? 100 : Number(r.progress) || 0;
      const label = STATE_LABELS[kind][r.status] ?? r.status;
      const pill = !isRemove && r.status === "processing" ? `${Math.floor(pct)}%` : label;
      const bar = !isRemove && inFlight ? `<div class="cvl-bar"><div style="width:${pct}%"></div></div>` : "";
      const err = r.status === "error" && r.error ? `<div class="cvl-item-err">${esc(r.error)}</div>` : "";
      const cancel = !isRemove && inFlight ? `<button class="cvl-cancel" data-id="${esc(r.id)}" title="Cancel download">✕</button>` : "";
      return `<div class="cvl-item ${inFlight ? "" : "done"}"><div class="cvl-item-line">
          <span class="cvl-glyph ${kind}" title="${isRemove ? "Remove" : "Download"}">${isRemove ? "✕" : "↓"}</span>
          <span class="cvl-item-name" title="${esc(resourceLabel(r))}">${esc(resourceLabel(r))}</span>
          <span class="cvl-pill ${esc(r.status)}">${esc(pill)}</span>${cancel}</div>${bar}${err}</div>`;
    })
    .join("")}</div>`;
}

let badgeTimer = null;
function railBadge() {
  const icon = document.querySelector(".side-tool-bar-container .cvg-civitai-icon, .cvg-civitai-icon");
  const button = icon?.closest("button");
  if (!button) return null;
  let badge = button.querySelector(".cvl-rail-badge");
  if (!badge) {
    injectStyles();
    if (getComputedStyle(button).position === "static") button.style.position = "relative";
    badge = document.createElement("span");
    badge.className = "cvl-rail-badge";
    button.appendChild(badge);
  }
  return badge;
}

function updateRailBadge(flash) {
  const badge = railBadge();
  if (!badge) return;
  const running = [...activities.values()].filter(
    (r) => r.type === "resources:add" && (r.status === "processing" || r.status === "pending")
  );
  if (running.length) {
    clearTimeout(badgeTimer);
    const pct = Math.min(...running.map((r) => Number(r.progress) || 0));
    badge.style.setProperty("--cvl-progress", `${pct}%`);
    badge.className = "cvl-rail-badge busy";
    return;
  }
  if (flash) {
    badge.className = `cvl-rail-badge ${flash}`;
    clearTimeout(badgeTimer);
    badgeTimer = setTimeout(() => (badge.className = "cvl-rail-badge"), 6000);
    return;
  }
  if (!/done|error/.test(badge.className)) badge.className = "cvl-rail-badge";
}

function render() {
  if (!panelEl) return;
  if (isHostedSession() || (state && state.hosted)) {
    panelEl.innerHTML = "";
    return;
  }
  injectStyles();
  const [tone, text] = statusLine();
  const paired = !!(state && state.paired);
  const canPair = !!(state && state.available && state.enabled);
  panelEl.innerHTML = `
    <div class="cvl-panel">
      <div class="cvl-head"><span class="cvl-dot ${tone}"></span><span class="cvl-title">Civitai Link</span>
        <span class="cvl-status" title="${esc(text)}">${esc(text)}</span>
        ${paired ? '<button class="cvl-btn cvl-unpair" title="Forget this pairing">Unpair</button>' : ""}
      </div>
      ${!paired && canPair ? `
        <div class="cvl-row"><input class="cvl-code" maxlength="6" placeholder="6-char code" autocomplete="off" spellcheck="false" />
          <button class="cvl-btn primary cvl-pair">Pair</button></div>
        <div class="cvl-hint">On civitai.com open <b>Civitai Link</b> (the link icon in the header), add an instance and paste its
          code here. Models you send from the site land in the matching <code>models/</code> folder.
          <a href="${LINK_HELP_URL}" target="_blank" rel="noopener">Account ↗</a></div>` : ""}
      <div class="cvl-err"></div>
      ${activityMarkup()}
    </div>`;
  const err = panelEl.querySelector(".cvl-err");
  const pairBtn = panelEl.querySelector(".cvl-pair");
  const input = panelEl.querySelector(".cvl-code");
  const pair = async () => {
    const code = (input?.value || "").trim();
    if (!code) return;
    pairBtn.disabled = true;
    err.textContent = "";
    try {
      const res = await fetch("/civitai/link/pair", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.status);
      state = data;
      render();
    } catch (e) {
      err.textContent = String(e.message || e);
      pairBtn.disabled = false;
    }
  };
  pairBtn?.addEventListener("click", pair);
  input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") pair();
  });
  for (const btn of panelEl.querySelectorAll(".cvl-cancel")) {
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        const res = await fetch("/civitai/link/cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: btn.dataset.id }),
        });
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.error || res.status);
      } catch (e) {
        err.textContent = `Cancel failed: ${e.message || e}`;
        btn.disabled = false;
      }
    });
  }
  panelEl.querySelector(".cvl-unpair")?.addEventListener("click", async () => {
    try {
      const res = await fetch("/civitai/link/unpair", { method: "POST" });
      state = await res.json();
      activities.clear();
      render();
    } catch (e) {
      err.textContent = String(e.message || e);
    }
  });
}

async function renderPanel(el) {
  panelEl = el;
  try {
    state = await (await fetch("/civitai/link/status")).json();
    activities.clear();
    for (const a of state.activities || []) activities.set(a.id, a);
  } catch (e) {
    state = null;
  }
  updateRailBadge();
  render();
}

function handleStatus(detail) {
  state = { ...(state || {}), ...(detail || {}) };
  render();
}

function handleActivity(r) {
  if (!r || !r.id) return;
  activities.set(r.id, r);
  while (activities.size > 60) activities.delete(activities.keys().next().value);
  const label = resourceLabel(r);
  let flash = null;
  if (r.type === "resources:add") {
    if (r.status === "processing" && !announced.has(r.id)) {
      announced.add(r.id);
      toast("info", "Civitai Link download started", label);
    } else if (r.status === "success") {
      toast("success", "Civitai Link download complete", label);
      refreshModelLists();
      flash = "done";
    } else if (r.status === "error") {
      toast("error", "Civitai Link download failed", `${label}: ${r.error || "unknown error"}`);
      flash = "error";
    } else if (r.status === "canceled") {
      toast("info", "Civitai Link download canceled", label);
    }
  } else if (r.type === "resources:remove" && r.status === "success") {
    toast("info", "Civitai Link removed a model", label);
    refreshModelLists();
  } else if (r.type === "resources:remove" && r.status === "error") {
    toast("error", "Civitai Link could not remove a model", `${label}: ${r.error || "unknown error"}`);
  }
  updateRailBadge(flash);
  render();
}

window.civitaiLink = { renderPanel };

app.registerExtension({
  name: "civitai.link",
  setup() {
    api.addCustomEventListener("civitai.link.status", (event) => handleStatus(event.detail));
    api.addCustomEventListener("civitai.link.activity", (event) => handleActivity(event.detail));
  },
});
