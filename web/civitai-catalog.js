// Civitai catalogue picker for the civitai-comfy-nodes pack. Adds a "Browse Civitai"
// button to the LoRA/Checkpoint loader nodes (and AIR `model` widgets) that opens a
// searchable card grid — pick a resource and its AIR drops straight into the widget.
// Backed by the pack's same-origin /civitai/catalog/search proxy (no CORS).
import { app } from "../../scripts/app.js";

const NODE_TARGETS = {
  CivitaiLoraLoader: { widget: "air", type: "LORA" },
  CivitaiCheckpointLoader: { widget: "air", type: "Checkpoint" },
};
const TYPES = ["Checkpoint", "LORA", "TextualInversion", "VAE", "Controlnet", "Upscaler"];

// Populated from /civitai/catalog/meta: { ecosystems: [{key,label}], nodeEcosystems: {NodeClass: key} }.
let META = { ecosystems: [], nodeEcosystems: {} };

// The ecosystem a node's resources must belong to. For a loader, trace its output downstream
// (through loader chains) to the recipe node it feeds.
function resolveEcosystem(node, seen) {
  seen = seen || new Set();
  if (!node || seen.has(node.id)) return "";
  seen.add(node.id);
  const own = META.nodeEcosystems[node.comfyClass];
  if (own) return own;
  for (const out of node.outputs || []) {
    for (const linkId of out.links || []) {
      const link = app.graph?.links?.[linkId];
      const target = link && app.graph.getNodeById(link.target_id);
      const eco = target && resolveEcosystem(target, seen);
      if (eco) return eco;
    }
  }
  return "";
}

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const css = `
    .cvc-backdrop { position: fixed; inset: 0; z-index: 2147483647; background: rgba(0,0,0,.6);
      display: flex; align-items: center; justify-content: center; font: 14px system-ui, sans-serif; }
    .cvc-modal { width: min(1100px, calc(100vw - 64px)); height: min(760px, calc(100vh - 64px));
      background: #18181b; color: #e4e4e7; border: 1px solid #3f3f46; border-radius: 14px;
      box-shadow: 0 24px 64px rgba(0,0,0,.5); display: flex; flex-direction: column; overflow: hidden; }
    .cvc-head { display: flex; gap: 10px; align-items: center; padding: 14px 16px; border-bottom: 1px solid #27272a; }
    .cvc-title { font-size: 16px; font-weight: 700; white-space: nowrap; }
    .cvc-input { flex: 1; box-sizing: border-box; background: #27272a; color: #e4e4e7;
      border: 1px solid #3f3f46; border-radius: 8px; padding: 9px 12px; font: inherit; outline: none; }
    .cvc-input:focus { border-color: #2563eb; }
    .cvc-select { background: #27272a; color: #e4e4e7; border: 1px solid #3f3f46; border-radius: 8px;
      padding: 9px 10px; font: inherit; outline: none; cursor: pointer; }
    .cvc-close { background: transparent; color: #a1a1aa; border: none; border-radius: 8px;
      width: 34px; height: 34px; font-size: 18px; cursor: pointer; }
    .cvc-close:hover { background: #27272a; color: #e4e4e7; }
    .cvc-status { padding: 6px 16px; color: #a1a1aa; font-size: 12px; min-height: 18px; }
    .cvc-grid { flex: 1; overflow-y: auto; padding: 4px 16px 16px; display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); grid-auto-rows: 300px; gap: 14px; align-content: start; }
    .cvc-card { height: 100%; background: #1f1f23; border: 1px solid #27272a; border-radius: 10px;
      overflow: hidden; cursor: pointer; display: flex; flex-direction: column; text-align: left; padding: 0;
      font: inherit; color: inherit; }
    .cvc-card:hover { border-color: #2563eb; }
    .cvc-thumb { position: relative; flex: 1; min-height: 0; background: #111113; }
    .cvc-thumb img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .cvc-badges { position: absolute; left: 8px; bottom: 8px; display: flex; gap: 4px; }
    .cvc-badge { background: rgba(24,24,27,.85); color: #d4d4d8; border-radius: 5px; padding: 2px 7px;
      font-size: 11px; font-weight: 600; }
    .cvc-link { position: absolute; right: 8px; top: 8px; background: rgba(24,24,27,.85); color: #d4d4d8;
      border-radius: 6px; padding: 2px 8px; font-size: 13px; text-decoration: none; line-height: 1.4; }
    .cvc-link:hover { background: #2563eb; color: #fff; }
    .cvc-meta { padding: 10px 10px 12px; min-width: 0; }
    .cvc-name { font-weight: 600; font-size: 13px; overflow: hidden; display: -webkit-box;
      -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    .cvc-sub { color: #a1a1aa; font-size: 11px; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .cvc-empty { grid-column: 1 / -1; color: #a1a1aa; text-align: center; padding: 48px 0; }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function openCatalog(targetWidget, defaultType, defaultEcosystem) {
  injectStyles();
  const types = META.types && META.types.length ? META.types : TYPES;
  const ecoOptions =
    `<option value="">Any ecosystem</option>` +
    META.ecosystems
      .map((e) => `<option value="${e.key}"${e.key === defaultEcosystem ? " selected" : ""}>${esc(e.label)}</option>`)
      .join("");
  const backdrop = document.createElement("div");
  backdrop.className = "cvc-backdrop";
  backdrop.innerHTML = `
    <div class="cvc-modal">
      <div class="cvc-head">
        <span class="cvc-title">🔍 Civitai</span>
        <input class="cvc-input" placeholder="Search Civitai models…" />
        <select class="cvc-select cvc-type">${types.map((t) => `<option value="${t}"${t === defaultType ? " selected" : ""}>${esc(t)}</option>`).join("")}</select>
        <select class="cvc-select cvc-eco" title="Filter by base-model ecosystem">${ecoOptions}</select>
        <button class="cvc-close" title="Close">✕</button>
      </div>
      <div class="cvc-status"></div>
      <div class="cvc-grid"></div>
    </div>`;
  document.body.appendChild(backdrop);

  const input = backdrop.querySelector(".cvc-input");
  const select = backdrop.querySelector(".cvc-type");
  const ecoSelect = backdrop.querySelector(".cvc-eco");
  const status = backdrop.querySelector(".cvc-status");
  const grid = backdrop.querySelector(".cvc-grid");

  const close = () => backdrop.remove();
  backdrop.addEventListener("mousedown", (e) => { if (e.target === backdrop) close(); });
  backdrop.querySelector(".cvc-close").addEventListener("click", close);
  document.addEventListener("keydown", function onKey(e) {
    if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); }
  });

  let reqId = 0;
  async function run() {
    const myId = ++reqId;
    const query = input.value.trim();
    const type = select.value;
    status.textContent = "Searching…";
    grid.innerHTML = "";
    try {
      const params = new URLSearchParams({ type });
      if (query) params.set("query", query);
      if (ecoSelect.value) params.set("ecosystem", ecoSelect.value);
      const res = await fetch(`/civitai/catalog/search?${params}`);
      const data = await res.json();
      if (myId !== reqId) return; // a newer search superseded this one
      if (data.error) { status.textContent = `Error: ${data.error}`; return; }
      render(data.entries || []);
    } catch (e) {
      if (myId === reqId) status.textContent = `Request failed: ${e}`;
    }
  }

  function render(entries) {
    status.textContent = entries.length ? `${entries.length} result${entries.length === 1 ? "" : "s"}` : "";
    if (!entries.length) { grid.innerHTML = `<div class="cvc-empty">No matching resources.</div>`; return; }
    grid.innerHTML = "";
    for (const e of entries) {
      const card = document.createElement("div");
      card.className = "cvc-card";
      card.setAttribute("role", "button");
      card.tabIndex = 0;
      const thumb = e.thumbnailUrl ? `<img src="${esc(e.thumbnailUrl)}" loading="lazy" />` : "";
      const link = e.modelUrl
        ? `<a class="cvc-link" href="${esc(e.modelUrl)}" target="_blank" rel="noopener" title="Open on Civitai">↗</a>`
        : "";
      card.innerHTML = `
        <div class="cvc-thumb">${thumb}
          <div class="cvc-badges"><span class="cvc-badge">${esc(e.baseModel || e.ecosystem)}</span></div>
          ${link}
        </div>
        <div class="cvc-meta">
          <div class="cvc-name">${esc(e.name)}</div>
          <div class="cvc-sub">${esc(e.versionName)} · ⬇ ${e.downloadCount ?? 0}</div>
        </div>`;
      // The ↗ link opens the model page; don't let that click also pick the card.
      card.querySelector(".cvc-link")?.addEventListener("click", (ev) => ev.stopPropagation());
      const pick = () => {
        targetWidget.value = e.air;
        targetWidget.callback?.(e.air, app.canvas, targetWidget.node, undefined, undefined);
        app.graph?.setDirtyCanvas?.(true, true);
        close();
      };
      card.addEventListener("click", pick);
      card.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); }
      });
      grid.appendChild(card);
    }
  }

  let debounce;
  input.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(run, 300); });
  select.addEventListener("change", run);
  ecoSelect.addEventListener("change", run);
  input.focus();
  run();
}

function targetFor(node) {
  const mapped = NODE_TARGETS[node.comfyClass];
  if (mapped) {
    const w = node.widgets?.find((x) => x.name === mapped.widget);
    if (w) return { widget: w, type: mapped.type };
  }
  if (node.comfyClass?.startsWith("Civitai")) {
    const w = node.widgets?.find((x) => x.name === "model" && x.type !== "combo");
    if (w) return { widget: w, type: "Checkpoint" };
  }
  return null;
}

app.registerExtension({
  name: "civitai.catalog",
  async setup() {
    try {
      META = await (await fetch("/civitai/catalog/meta")).json();
    } catch (e) {
      console.warn("[civitai-catalog] could not load ecosystem metadata", e);
    }
  },
  nodeCreated(node) {
    const target = targetFor(node);
    if (!target) return;
    // Append at the end so it never shifts the data widgets' serialization order.
    // Ecosystem is resolved at click time so it reflects the node's current wiring.
    const button = node.addWidget("button", "🔍 Browse Civitai", null, () =>
      openCatalog(target.widget, target.type, resolveEcosystem(node))
    );
    button.serialize = false;
  },
});
