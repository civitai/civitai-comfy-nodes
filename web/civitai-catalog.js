// Civitai catalogue picker for the civitai-comfy-nodes pack. Adds a "Browse Civitai" button to the
// selector nodes (Model / LoRA / Embedding) that opens a searchable card grid — pick a resource and
// its AIR drops into the node's `air` widget. The picker defaults its type + ecosystem from where
// the selector is wired. Each selector also shows an on-node preview (thumbnail + name) of the
// current AIR, resolved via the /civitai/catalog/lookup proxy. Backed by the pack's same-origin
// /civitai/catalog/* proxies (no CORS).
import { app } from "../../scripts/app.js";

const NODE_TARGETS = {
  CivitaiModelSelector: { widget: "air", type: "Checkpoint" },
  CivitaiEmbeddingSelector: { widget: "air", type: "TextualInversion" },
};
// The multi-LoRA node manages its own rows UI (see setupLoraRows) rather than a single `air` widget.
const LORA_NODE = "CivitaiLoraLoader";
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

// Map the recipe input a Model Selector's `air` feeds (model/vae_model/clip_l_model/…) to a Civitai
// catalogue type, so the picker defaults to the model type that socket actually needs. Most-specific
// patterns first; the fallback is Checkpoint (covers `model`, `language_model`, …).
const AIR_TYPE_BY_INPUT = [
  [/clip[_-]?vision|clipvision/, "CLIPVision"],
  [/control/, "Controlnet"],
  [/upscal/, "Upscaler"],
  [/vae/, "VAE"],
  [/clip|t5|text[_-]?encoder|encoder/, "TextEncoder"],
  [/diffus|unet/, "UNet"],
];
function modelTypeForInputName(name) {
  const n = (name || "").toLowerCase();
  for (const [re, type] of AIR_TYPE_BY_INPUT) if (re.test(n)) return type;
  return "Checkpoint";
}

// For a Model Selector, deduce the catalogue type from where its `air` output is wired.
function resolveModelType(node) {
  const out = node.outputs?.find((o) => o.name === "air");
  for (const linkId of out?.links || []) {
    const link = app.graph?.links?.[linkId];
    const target = link && app.graph.getNodeById(link.target_id);
    const input = target?.inputs?.[link.target_slot];
    if (input) return modelTypeForInputName(input.name);
  }
  return null;
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
    .cvc-np { display: flex; gap: 9px; align-items: center; box-sizing: border-box; width: 100%; height: 100%;
      padding: 3px 4px; overflow: hidden; cursor: pointer; font: 12px system-ui, sans-serif; }
    .cvc-np-thumb { width: 54px; height: 54px; flex: 0 0 auto; border-radius: 7px; overflow: hidden;
      background: #111113; display: flex; align-items: center; justify-content: center; color: #52525b; font-size: 20px; }
    .cvc-np-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .cvc-np-meta { min-width: 0; flex: 1; overflow: hidden; }
    .cvc-np-name { font-weight: 600; font-size: 12px; color: #e4e4e7; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap; }
    .cvc-np-sub { font-size: 11px; color: #a1a1aa; margin-top: 3px; overflow: hidden;
      text-overflow: ellipsis; white-space: nowrap; }
    .cvc-np.cvc-np-empty { cursor: default; }
    .cvc-np.cvc-np-empty .cvc-np-name { color: #a1a1aa; font-weight: 500; }
    .cvl { display: flex; flex-direction: column; gap: 5px; box-sizing: border-box; width: 100%;
      padding: 4px 2px; overflow: hidden; font: 12px system-ui, sans-serif; }
    .cvl-rows { display: flex; flex-direction: column; gap: 4px; flex: 0 0 auto; }
    .cvl-row { display: flex; align-items: center; gap: 6px; background: #1f1f23; border: 1px solid #27272a;
      border-radius: 7px; padding: 3px 6px; box-sizing: border-box; }
    .cvl-row.cvl-off { opacity: .45; }
    .cvl-on { flex: 0 0 auto; cursor: pointer; margin: 0; }
    .cvl-thumb { width: 22px; height: 22px; border-radius: 4px; object-fit: cover; background: #111113; flex: 0 0 auto; }
    .cvl-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      cursor: pointer; color: #e4e4e7; }
    .cvl-name:hover { color: #60a5fa; }
    .cvl-tw { width: 76px; flex: 0 0 auto; box-sizing: border-box; background: #27272a; color: #e4e4e7;
      border: 1px solid #3f3f46; border-radius: 5px; padding: 2px 5px; font: inherit; }
    .cvl-str { width: 52px; flex: 0 0 auto; box-sizing: border-box; background: #27272a; color: #e4e4e7;
      border: 1px solid #3f3f46; border-radius: 5px; padding: 2px 4px; font: inherit; text-align: right;
      -moz-appearance: textfield; appearance: textfield; }
    .cvl-str::-webkit-inner-spin-button, .cvl-str::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
    .cvl-x { flex: 0 0 auto; background: transparent; border: none; color: #a1a1aa; cursor: pointer;
      font-size: 14px; line-height: 1; padding: 0 2px; }
    .cvl-x:hover { color: #f87171; }
    .cvl-add { background: #27272a; color: #e4e4e7; border: 1px dashed #3f3f46; border-radius: 7px;
      padding: 6px; cursor: pointer; font: inherit; flex: 0 0 auto; }
    .cvl-add:hover { border-color: #2563eb; color: #fff; }
    .cvl-empty { color: #71717a; text-align: center; padding: 6px 0; }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// openCatalog({ type, ecosystem, onPick }) — onPick(entry) receives the chosen catalogue entry.
function openCatalog({ type: defaultType, ecosystem: defaultEcosystem, onPick }) {
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
        onPick?.(e);
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

// ── On-node model preview (thumbnail + name) for the selector nodes ───────────────────────────────
const PREVIEW_HEIGHT = 64;

function previewState(node) {
  if (node.__cvcPreview) return node.__cvcPreview;
  injectStyles();
  const el = document.createElement("div");
  el.className = "cvc-np cvc-np-empty";
  el.innerHTML =
    `<div class="cvc-np-thumb">🖼</div>` +
    `<div class="cvc-np-meta"><div class="cvc-np-name"></div><div class="cvc-np-sub"></div></div>`;
  const widget = node.addDOMWidget("civitai_model_preview", "preview", el, { serialize: false });
  // Pin the element to the allocated widget width so a long model name truncates (ellipsis) instead
  // of growing the box past the node — the inner flex needs a *definite* width to shrink against, and
  // a freshly-picked node doesn't get one until a resize otherwise.
  widget.computeSize = (width) => {
    if (typeof width === "number" && width > 0) el.style.width = `${width}px`;
    return [width, PREVIEW_HEIGHT];
  };
  el.addEventListener("click", () => {
    const url = node.__cvcPreview?.entry?.modelUrl;
    if (url) window.open(url, "_blank", "noopener");
  });
  node.__cvcPreview = { widget, el, entry: null };
  return node.__cvcPreview;
}

function renderPreview(node, { entry = null, air = "", sub = "", empty = false } = {}) {
  const st = previewState(node);
  st.entry = entry;
  const thumb = st.el.querySelector(".cvc-np-thumb");
  const name = st.el.querySelector(".cvc-np-name");
  const subEl = st.el.querySelector(".cvc-np-sub");
  st.el.classList.toggle("cvc-np-empty", empty);
  if (empty) {
    thumb.innerHTML = "🖼";
    name.textContent = "No model selected";
    subEl.textContent = "Use Browse Civitai";
    st.el.title = "";
    return;
  }
  thumb.innerHTML = entry?.thumbnailUrl ? `<img src="${esc(entry.thumbnailUrl)}" loading="lazy" />` : "🖼";
  name.textContent = entry?.name || air || "Model";
  const bits = [entry?.versionName, entry?.baseModel].filter(Boolean);
  subEl.textContent = sub || bits.join(" · ") || air;
  st.el.title = entry?.modelUrl ? "Open on Civitai ↗" : "";
}

// Reflect the node's current `air` value: render from the stashed metadata when it matches, else
// resolve the AIR via the lookup proxy (handles pasted AIRs and graphs saved before this existed).
function refreshPreview(node, airWidget) {
  const air = (airWidget.value || "").trim();
  if (!air) { renderPreview(node, { empty: true }); return; }
  const stored = node.properties?.civitai_model;
  if (stored && stored.air === air) { renderPreview(node, { entry: stored }); return; }
  renderPreview(node, { air, sub: "Loading…" });
  clearTimeout(node.__cvcLookupT);
  node.__cvcLookupT = setTimeout(async () => {
    try {
      const res = await fetch(`/civitai/catalog/lookup?air=${encodeURIComponent(air)}`);
      const data = await res.json();
      if ((airWidget.value || "").trim() !== air) return; // air changed while loading
      if (data.entry) {
        node.properties = node.properties || {};
        node.properties.civitai_model = data.entry;
        renderPreview(node, { entry: data.entry });
      } else {
        renderPreview(node, { air, sub: "Details unavailable" });
      }
    } catch {
      if ((airWidget.value || "").trim() === air) renderPreview(node, { air, sub: "Details unavailable" });
    }
  }, 250);
}

function setupPreview(node, airWidget) {
  previewState(node); // create up-front so the preview sits above the Browse button
  const origCb = airWidget.callback;
  airWidget.callback = function (...a) {
    const r = origCb?.apply(this, a);
    refreshPreview(node, airWidget);
    return r;
  };
  const origConfigure = node.onConfigure;
  node.onConfigure = function () {
    const r = origConfigure?.apply(this, arguments);
    refreshPreview(node, airWidget);
    return r;
  };
  refreshPreview(node, airWidget);
}

// ── Multi-LoRA rows widget (one node holds many LoRAs: toggle / browse / strength / trigger) ──────
const LORA_ROW_H = 30;

function airTail(air) {
  const s = String(air || "");
  return s.split("@")[0].split(":").pop() || s;
}

function loraJsonWidget(node) {
  return node.widgets?.find((w) => w.name === "loras_json");
}

function loraRows(node) {
  const w = loraJsonWidget(node);
  try {
    const v = JSON.parse(w?.value || "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function writeLoraRows(node, rows) {
  const w = loraJsonWidget(node);
  if (w) {
    w.value = JSON.stringify(rows);
    w.callback?.(w.value);
  }
}

function loraState(node) {
  if (node.__cvlState) return node.__cvlState;
  injectStyles();
  const el = document.createElement("div");
  el.className = "cvl";
  el.innerHTML = `<div class="cvl-rows"></div><button class="cvl-add">＋ Add LoRA</button>`;
  const widget = node.addDOMWidget("civitai_loras", "loras", el, { serialize: false });
  widget.computeSize = (width) => {
    // Pin the element to the allocated widget width, else the fixed-width row controls stretch the
    // rows across the whole canvas (the framework doesn't constrain the DOM element on its own here).
    if (typeof width === "number" && width > 0) el.style.width = `${width}px`;
    // Height comes from a real measurement of the rendered rows (see resizeLoraNode); the row-count
    // estimate is only the first-paint fallback before that measurement lands.
    const estimate = 8 + (loraRows(node).length || 1) * LORA_ROW_H + 34;
    return [width, node.__cvlHeight || estimate];
  };
  el.querySelector(".cvl-add").addEventListener("click", () =>
    openCatalog({
      type: "LORA",
      ecosystem: resolveEcosystem(node),
      onPick: (e) => {
        const rows = loraRows(node);
        rows.push({
          air: e.air, name: e.name, strength: 1.0, triggerWord: "", on: true,
          thumbnailUrl: e.thumbnailUrl, versionName: e.versionName, baseModel: e.baseModel,
        });
        commitLoraRows(node, rows);
      },
    })
  );
  node.__cvlState = { widget, el };
  return node.__cvlState;
}

function commitLoraRows(node, rows) {
  writeLoraRows(node, rows);
  renderLoraRows(node);
}

function renderLoraRows(node) {
  const st = loraState(node);
  const rows = loraRows(node);
  const list = st.el.querySelector(".cvl-rows");
  list.innerHTML = "";
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "cvl-empty";
    empty.textContent = "No LoRAs yet — add one below.";
    list.appendChild(empty);
  }
  rows.forEach((row, i) => {
    const r = document.createElement("div");
    r.className = "cvl-row" + (row.on === false ? " cvl-off" : "");
    const thumb = row.thumbnailUrl ? `<img class="cvl-thumb" src="${esc(row.thumbnailUrl)}" />` : "";
    r.innerHTML =
      `<input type="checkbox" class="cvl-on" ${row.on === false ? "" : "checked"} title="Enable / disable" />` +
      thumb +
      `<div class="cvl-name" title="${esc(row.air)} — click to change">${esc(row.name || airTail(row.air))}</div>` +
      `<input class="cvl-tw" placeholder="trigger" value="${esc(row.triggerWord || "")}" title="Trigger word" />` +
      `<input class="cvl-str" type="number" step="0.05" value="${esc(row.strength ?? 1.0)}" title="Strength" />` +
      `<button class="cvl-x" title="Remove">✕</button>`;
    r.querySelector(".cvl-on").addEventListener("change", (e) => {
      row.on = e.target.checked;
      r.classList.toggle("cvl-off", !e.target.checked);
      writeLoraRows(node, rows);
    });
    r.querySelector(".cvl-name").addEventListener("click", () =>
      openCatalog({
        type: "LORA",
        ecosystem: resolveEcosystem(node),
        onPick: (e) => {
          Object.assign(row, {
            air: e.air, name: e.name,
            thumbnailUrl: e.thumbnailUrl, versionName: e.versionName, baseModel: e.baseModel,
          });
          commitLoraRows(node, rows);
        },
      })
    );
    r.querySelector(".cvl-tw").addEventListener("change", (e) => {
      row.triggerWord = e.target.value;
      writeLoraRows(node, rows);
    });
    r.querySelector(".cvl-str").addEventListener("change", (e) => {
      row.strength = parseFloat(e.target.value);
      if (Number.isNaN(row.strength)) row.strength = 1.0;
      writeLoraRows(node, rows);
    });
    r.querySelector(".cvl-x").addEventListener("click", () => {
      rows.splice(i, 1);
      commitLoraRows(node, rows);
    });
    list.appendChild(r);
  });
  // Measure the actual rendered height next frame (layout px, zoom-independent) and resize to fit.
  requestAnimationFrame(() => resizeLoraNode(node));
}

function resizeLoraNode(node) {
  const st = node.__cvlState;
  if (!st) return;
  const list = st.el.querySelector(".cvl-rows");
  const add = st.el.querySelector(".cvl-add");
  // offsetHeight is layout px (unaffected by the canvas zoom transform); .cvl-rows never shrinks
  // (flex 0 0 auto) so this is the true content height even while the element is momentarily clipped.
  node.__cvlHeight = (list?.offsetHeight || 0) + (add?.offsetHeight || 30) + 13; // padding(8) + gap(5)
  const width = Math.max(node.size?.[0] || 0, 360);
  const sized = node.computeSize?.() || [width, node.size?.[1] || 0];
  node.setSize?.([width, sized[1]]);
  node.setDirtyCanvas?.(true, true);
}

function setupLoraRows(node) {
  const w = loraJsonWidget(node);
  if (w) {
    // The JSON string is the serialized source of truth; hide its raw widget, show the rows UI.
    w.hidden = true;
    w.type = "hidden";
    w.computeSize = () => [0, -4];
  }
  loraState(node);
  renderLoraRows(node);
  if ((node.size?.[0] || 0) < 360) node.setSize?.([360, node.size?.[1] || 140]);
  const origConfigure = node.onConfigure;
  node.onConfigure = function () {
    const r = origConfigure?.apply(this, arguments);
    renderLoraRows(node);
    return r;
  };
}

function targetFor(node) {
  // Only the dedicated Civitai selector nodes get a Browse button. Recipe nodes take their models
  // via CIVITAI_AIR sockets (wire a Model Selector), so there's no model widget to attach to.
  const mapped = NODE_TARGETS[node.comfyClass];
  if (mapped) {
    const w = node.widgets?.find((x) => x.name === mapped.widget);
    if (w) return { widget: w, type: mapped.type };
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
    if (node.comfyClass === LORA_NODE) {
      setupLoraRows(node);
      return;
    }
    const target = targetFor(node);
    if (!target) return;
    setupPreview(node, target.widget);
    // Append at the end so it never shifts the data widgets' serialization order.
    // Type + ecosystem are resolved at click time so they reflect the node's current wiring
    // (a Model Selector feeding a `vae` socket on an sd1 node defaults to VAE / sd1).
    const button = node.addWidget("button", "🔍 Browse Civitai", null, () =>
      openCatalog({
        type: resolveModelType(node) || target.type,
        ecosystem: resolveEcosystem(node),
        onPick: (e) => {
          node.properties = node.properties || {};
          node.properties.civitai_model = e; // metadata for the on-node preview (persists in the graph)
          target.widget.value = e.air;
          target.widget.callback?.(e.air, app.canvas, node, undefined, undefined);
        },
      })
    );
    button.serialize = false;
  },
});
