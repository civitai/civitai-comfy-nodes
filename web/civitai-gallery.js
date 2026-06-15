// "Civitai" sidebar tab: browse your Civitai generation history (every media type, every source)
// pulled from the orchestrator via the pack's same-origin /civitai/workflows/* proxy, and pull any
// result back into the graph. The native "Media Assets" panel is core ComfyUI and can't take a
// sub-tab, so this registers a sibling sidebar tab.
import { app } from "../../scripts/app.js";

const PAGE = 60;
// kind -> the loader node + its file widget; gracefully degrades when a node type isn't installed.
const LOADERS = {
  image: { types: ["LoadImage"], widget: "image" },
  video: { types: ["VHS_LoadVideo", "LoadVideo"], widget: "video" },
  audio: { types: ["LoadAudio"], widget: "audio" },
  model3d: { types: ["Load3D"], widget: "model" },
};

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const css = `
    .cvg-civitai-icon::before { content: ""; display: inline-block; width: 1.25em; height: 1.25em;
      vertical-align: -0.22em; background-color: currentColor;
      -webkit-mask: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNzggMTc4Ij48cGF0aCBkPSJNODkuMywyOS4ybDUyLDMwdjYwbC01MiwzMGwtNTItMzB2LTYwTDg5LjMsMjkuMiBNODkuMywxLjVsLTc2LDQzLjl2ODcuOGw3Niw0My45bDc2LTQzLjlWNDUuNEw4OS4zLDEuNUw4OS4zLDEuNXoiLz48cG9seWdvbiBwb2ludHM9IjEwNC4xLDk3LjIgODkuMiwxMDUuNyA3NC4zLDk3LjIgNzQuMyw4MC4yIDg5LjIsNzEuNyAxMDQuMSw4MC4yIDEyMi4zLDgwLjIgMTIyLjMsNjkuNyA4OS4zLDUwLjcgNTYuMyw2OS43IDU2LjMsMTA3LjggODkuMywxMjYuOCAxMjIuMywxMDcuOCAxMjIuMyw5Ny4yICIvPjwvc3ZnPg==") center/contain no-repeat;
      mask: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNzggMTc4Ij48cGF0aCBkPSJNODkuMywyOS4ybDUyLDMwdjYwbC01MiwzMGwtNTItMzB2LTYwTDg5LjMsMjkuMiBNODkuMywxLjVsLTc2LDQzLjl2ODcuOGw3Niw0My45bDc2LTQzLjlWNDUuNEw4OS4zLDEuNUw4OS4zLDEuNXoiLz48cG9seWdvbiBwb2ludHM9IjEwNC4xLDk3LjIgODkuMiwxMDUuNyA3NC4zLDk3LjIgNzQuMyw4MC4yIDg5LjIsNzEuNyAxMDQuMSw4MC4yIDEyMi4zLDgwLjIgMTIyLjMsNjkuNyA4OS4zLDUwLjcgNTYuMyw2OS43IDU2LjMsMTA3LjggODkuMywxMjYuOCAxMjIuMywxMDcuOCAxMjIuMyw5Ny4yICIvPjwvc3ZnPg==") center/contain no-repeat; }
    .cvg-root { display: flex; flex-direction: column; height: 100%; color: #e4e4e7;
      font: 13px system-ui, sans-serif; box-sizing: border-box; }
    .cvg-bar { display: flex; gap: 8px; align-items: center; padding: 8px 10px; border-bottom: 1px solid #27272a; }
    .cvg-bar select, .cvg-btn { background: #27272a; color: #e4e4e7; border: 1px solid #3f3f46;
      border-radius: 7px; padding: 6px 9px; font: inherit; outline: none; cursor: pointer; }
    .cvg-btn:hover { border-color: #2563eb; }
    .cvg-spacer { flex: 1; }
    .cvg-scroll { flex: 1; overflow-y: auto; padding: 8px; }
    .cvg-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; align-content: start; }
    .cvg-card { position: relative; background: #1f1f23; border: 1px solid #27272a; border-radius: 9px;
      overflow: hidden; aspect-ratio: 1; cursor: pointer; }
    .cvg-card:hover { border-color: #2563eb; }
    .cvg-card .cvg-media { width: 100%; height: 100%; object-fit: cover; display: block; }
    .cvg-ph { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
      font-size: 30px; color: #71717a; background: #111113; }
    .cvg-badge { position: absolute; left: 5px; top: 5px; background: rgba(24,24,27,.85); color: #d4d4d8;
      border-radius: 5px; padding: 1px 6px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .cvg-add { position: absolute; right: 5px; bottom: 5px; width: 24px; height: 24px; border-radius: 6px;
      border: none; background: rgba(37,99,235,.92); color: #fff; font-size: 16px; line-height: 1; cursor: pointer;
      opacity: 0; transition: opacity .1s; }
    .cvg-card:hover .cvg-add { opacity: 1; }
    .cvg-msg { color: #a1a1aa; text-align: center; padding: 32px 12px; line-height: 1.5; }
    .cvg-connect { padding: 18px 16px; display: flex; flex-direction: column; gap: 12px; }
    .cvg-connect h3 { margin: 0; font-size: 14px; }
    .cvg-connect p { margin: 0; color: #a1a1aa; line-height: 1.5; }
    .cvg-connect input { box-sizing: border-box; width: 100%; background: #27272a; color: #e4e4e7;
      border: 1px solid #3f3f46; border-radius: 8px; padding: 9px 11px; font: inherit; outline: none; }
    .cvg-primary { background: #2563eb; border-color: #2563eb; color: #fff; }
    .cvg-row { display: flex; gap: 8px; }
    .cvg-err { color: #f87171; font-size: 12px; min-height: 16px; }
    .cvg-lb { position: fixed; inset: 0; z-index: 2147483647; background: rgba(0,0,0,.78);
      display: flex; align-items: center; justify-content: center; padding: 32px; }
    .cvg-lb-inner { max-width: min(1000px, 92vw); max-height: 90vh; display: flex; flex-direction: column;
      gap: 12px; align-items: center; }
    .cvg-lb-inner img, .cvg-lb-inner video { max-width: 100%; max-height: 72vh; border-radius: 10px; }
    .cvg-lb-meta { color: #d4d4d8; font-size: 12px; text-align: center; line-height: 1.6; word-break: break-all; }
    .cvg-lb-actions { display: flex; gap: 8px; }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(severity, summary, detail) {
  try {
    app.extensionManager.toast.add({ severity, summary, detail, life: 4000 });
  } catch (e) {
    console[severity === "error" ? "error" : "log"](`[civitai-gallery] ${summary}: ${detail ?? ""}`);
  }
}

function nodeTypeFor(kind) {
  const reg = window.LiteGraph?.registered_node_types || {};
  for (const t of LOADERS[kind]?.types || []) if (reg[t]) return t;
  return null;
}

async function importMedia(media) {
  const res = await fetch("/civitai/workflows/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blobId: media.blobId, url: media.url, kind: media.kind }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `import failed (${res.status})`);
  return data.name;
}

function fileWidget(node, kind) {
  const want = LOADERS[kind]?.widget;
  return (
    node.widgets?.find((w) => w.name === want) ||
    node.widgets?.find((w) => Array.isArray(w.options?.values))
  );
}

function placeOnNode(node, kind, name) {
  const widget = fileWidget(node, kind);
  if (!widget) return false;
  if (Array.isArray(widget.options?.values) && !widget.options.values.includes(name)) {
    widget.options.values.push(name);
  }
  widget.value = name;
  widget.callback?.(name);
  app.graph.setDirtyCanvas(true, true);
  return true;
}

function viewCenter() {
  try {
    const c = app.canvas;
    const rect = c.canvas.getBoundingClientRect();
    return c.ds.convertCanvasToOffset([rect.width / 2, rect.height / 2]);
  } catch (e) {
    return [200, 200];
  }
}

async function addToCanvas(media, pos) {
  const type = nodeTypeFor(media.kind);
  if (!type) {
    toast("warn", "No loader node", `Install a loader for ${media.kind} to add it to the graph.`);
    return;
  }
  let name;
  try {
    name = await importMedia(media);
  } catch (e) {
    toast("error", "Import failed", String(e.message || e));
    return;
  }
  const node = window.LiteGraph.createNode(type);
  app.graph.add(node);
  node.pos = pos || viewCenter();
  if (!placeOnNode(node, media.kind, name)) {
    toast("warn", "Added node", "Couldn't auto-set the file widget; pick it manually.");
  }
}

async function fillSelected(media) {
  const selected = Object.values(app.canvas?.selected_nodes || {});
  if (!selected.length) {
    toast("warn", "No node selected", "Select a loader node first, or use Add to canvas.");
    return;
  }
  let name;
  try {
    name = await importMedia(media);
  } catch (e) {
    toast("error", "Import failed", String(e.message || e));
    return;
  }
  let filled = 0;
  for (const node of selected) if (placeOnNode(node, media.kind, name)) filled++;
  if (!filled) toast("warn", "No compatible node", "The selected node has no file widget for this media.");
}

function openLightbox(media, item) {
  const lb = document.createElement("div");
  lb.className = "cvg-lb";
  const src = media.url || media.previewUrl;
  const view =
    media.kind === "video"
      ? `<video src="${esc(src)}" controls autoplay loop></video>`
      : media.kind === "audio"
        ? `<audio src="${esc(src)}" controls autoplay></audio>`
        : media.kind === "model3d"
          ? `<a class="cvg-btn cvg-primary" href="${esc(src)}" target="_blank" rel="noopener">Open 3D model ↗</a>`
          : `<img src="${esc(src)}" />`;
  const when = item.createdAt ? new Date(item.createdAt).toLocaleString() : "";
  const cost = item.cost != null ? ` · cost ${esc(item.cost)}` : "";
  const prompt = item.meta?.prompt ? `<div>${esc(item.meta.prompt)}</div>` : "";
  lb.innerHTML = `
    <div class="cvg-lb-inner">
      ${view}
      <div class="cvg-lb-meta">${prompt}<div>${esc(item.workflowId)} · ${esc(item.status || "")}${cost}</div><div>${esc(when)}</div></div>
      <div class="cvg-lb-actions">
        <button class="cvg-btn cvg-primary cvg-lb-add">Add to canvas</button>
        <button class="cvg-btn cvg-lb-fill">Set on selected node</button>
        <a class="cvg-btn" href="${esc(src)}" target="_blank" rel="noopener">Open ↗</a>
      </div>
    </div>`;
  const close = () => lb.remove();
  lb.addEventListener("mousedown", (e) => { if (e.target === lb) close(); });
  lb.querySelector(".cvg-lb-add").addEventListener("click", () => { addToCanvas(media); close(); });
  lb.querySelector(".cvg-lb-fill").addEventListener("click", () => { fillSelected(media); close(); });
  document.addEventListener("keydown", function onKey(e) {
    if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); }
  });
  document.body.appendChild(lb);
}

function thumbHtml(media) {
  // VideoBlob/AudioBlob/3D have no preview image — render a first-frame <video> or a glyph instead.
  if (media.kind === "video")
    return `<video class="cvg-media" src="${esc(media.url)}#t=0.5" muted loop playsinline preload="metadata"></video>`;
  if (media.kind === "audio") return `<div class="cvg-ph">♪</div>`;
  if (media.kind === "model3d") return `<div class="cvg-ph">3D</div>`;
  return `<img class="cvg-media" loading="lazy" src="${esc(media.previewUrl || media.url)}" />`;
}

function card(media, item) {
  const el = document.createElement("div");
  el.className = "cvg-card";
  el.draggable = true;
  el.innerHTML = `${thumbHtml(media)}
    <span class="cvg-badge">${esc(media.kind)}</span>
    <button class="cvg-add" title="Add to canvas">＋</button>`;
  el.addEventListener("click", (e) => { if (!e.target.closest(".cvg-add")) openLightbox(media, item); });
  el.querySelector(".cvg-add").addEventListener("click", (e) => { e.stopPropagation(); addToCanvas(media); });
  const video = el.querySelector("video");
  if (video) {
    el.addEventListener("mouseenter", () => video.play?.().catch(() => {}));
    el.addEventListener("mouseleave", () => { video.pause?.(); video.currentTime = 0; });
  }
  el.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("application/x-civitai-media", JSON.stringify(media));
    e.dataTransfer.effectAllowed = "copy";
  });
  return el;
}

function renderConnect(el, onDone) {
  el.innerHTML = "";
  const box = document.createElement("div");
  box.className = "cvg-connect";
  box.innerHTML = `
    <h3>Connect to Civitai</h3>
    <p>Set <code>CIVITAI_API_TOKEN</code> on the server, or connect here to browse your generations.</p>
    <button class="cvg-btn cvg-primary cvg-oauth">Connect with Civitai (OAuth)</button>
    <p>or paste an API key from civitai.com/user/account:</p>
    <input class="cvg-key" type="password" placeholder="civitai_xxx…" />
    <div class="cvg-row"><button class="cvg-btn cvg-primary cvg-savekey">Save key</button></div>
    <div class="cvg-err"></div>`;
  el.appendChild(box);
  const err = box.querySelector(".cvg-err");
  box.querySelector(".cvg-oauth").addEventListener("click", async () => {
    err.textContent = "Opening browser login on the server… complete it, then this will load.";
    try {
      const res = await fetch("/civitai/auth/login", { method: "POST" });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.status);
      onDone();
    } catch (e) {
      err.textContent = `OAuth failed (expected on remote servers — use an API key): ${e.message || e}`;
    }
  });
  box.querySelector(".cvg-savekey").addEventListener("click", async () => {
    const apiKey = box.querySelector(".cvg-key").value.trim();
    if (!apiKey) { err.textContent = "Enter a key first."; return; }
    err.textContent = "Validating…";
    try {
      const res = await fetch("/civitai/auth/api-key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apiKey }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.status);
      onDone();
    } catch (e) {
      err.textContent = String(e.message || e);
    }
  });
}

function renderGallery(el) {
  el.innerHTML = "";
  const root = document.createElement("div");
  root.className = "cvg-root";
  root.innerHTML = `
    <div class="cvg-bar">
      <select class="cvg-filter">
        <option value="">All media</option>
        <option value="image">Images</option>
        <option value="video">Videos</option>
        <option value="audio">Audio</option>
        <option value="model3d">3D</option>
      </select>
      <span class="cvg-spacer"></span>
      <button class="cvg-btn cvg-refresh" title="Refresh">↻</button>
      <button class="cvg-btn cvg-disconnect" title="Disconnect">⏏</button>
    </div>
    <div class="cvg-scroll"><div class="cvg-grid"></div><div class="cvg-sentinel"></div><div class="cvg-msg" style="display:none"></div></div>`;
  el.appendChild(root);

  const grid = root.querySelector(".cvg-grid");
  const sentinel = root.querySelector(".cvg-sentinel");
  const msg = root.querySelector(".cvg-msg");
  const filterSel = root.querySelector(".cvg-filter");

  const state = { cursor: null, loading: false, done: false, shown: 0 };
  const kind = () => filterSel.value;

  function reset() {
    state.cursor = null; state.done = false; state.loading = false; state.shown = 0;
    grid.innerHTML = ""; msg.style.display = "none";
    loadMore();
  }

  async function loadMore() {
    if (state.loading || state.done) return;
    state.loading = true;
    try {
      const params = new URLSearchParams({ take: String(PAGE) });
      if (state.cursor) params.set("cursor", state.cursor);
      const res = await fetch(`/civitai/workflows/list?${params}`);
      const data = await res.json();
      if (res.status === 401 || data.error === "auth_required") { init(el); return; }
      if (!res.ok || data.error) throw new Error(data.error || res.status);
      const want = kind();
      for (const item of data.items || []) {
        for (const media of item.media || []) {
          if (want && media.kind !== want) continue;
          grid.appendChild(card(media, item));
          state.shown++;
        }
      }
      state.cursor = data.next || null;
      if (!state.cursor) state.done = true;
      if (!state.shown && state.done) { msg.textContent = "No generations yet."; msg.style.display = "block"; }
      // keep filling while the viewport isn't covered yet
      if (!state.done && grid.getBoundingClientRect().bottom < root.getBoundingClientRect().bottom) {
        state.loading = false;
        return loadMore();
      }
    } catch (e) {
      msg.textContent = `Error: ${e.message || e}`;
      msg.style.display = "block";
    } finally {
      state.loading = false;
    }
  }

  filterSel.addEventListener("change", reset);
  root.querySelector(".cvg-refresh").addEventListener("click", reset);
  root.querySelector(".cvg-disconnect").addEventListener("click", async () => {
    await fetch("/civitai/auth/logout", { method: "POST" });
    init(el);
  });

  el._cvgObserver?.disconnect();
  const observer = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) loadMore(); });
  observer.observe(sentinel);
  el._cvgObserver = observer;

  loadMore();
}

async function init(el) {
  injectStyles();
  el.classList.add("cvg-root");
  el.innerHTML = `<div class="cvg-msg">Loading…</div>`;
  try {
    const status = await (await fetch("/civitai/auth/status")).json();
    if (status.authenticated) renderGallery(el);
    else renderConnect(el, () => renderGallery(el));
  } catch (e) {
    el.innerHTML = `<div class="cvg-msg">Civitai gallery unavailable: ${esc(e.message || e)}</div>`;
  }
}

function installCanvasDrop() {
  const canvas = app.canvas?.canvas;
  if (!canvas || canvas._cvgDrop) return;
  canvas._cvgDrop = true;
  canvas.addEventListener("dragover", (e) => {
    if (e.dataTransfer?.types?.includes("application/x-civitai-media")) e.preventDefault();
  });
  canvas.addEventListener("drop", (e) => {
    const raw = e.dataTransfer?.getData("application/x-civitai-media");
    if (!raw) return;
    e.preventDefault(); e.stopPropagation();
    let pos;
    try { pos = app.canvas.convertEventToCanvasOffset(e); } catch (_e) { pos = undefined; }
    addToCanvas(JSON.parse(raw), pos);
  });
}

app.registerExtension({
  name: "civitai.gallery",
  async setup() {
    injectStyles();
    installCanvasDrop();
    try {
      app.extensionManager.registerSidebarTab({
        id: "civitai.generated",
        icon: "cvg-civitai-icon",
        title: "Civitai",
        tooltip: "Your Civitai generations",
        type: "custom",
        render: (el) => init(el),
      });
    } catch (e) {
      console.warn("[civitai-gallery] registerSidebarTab unavailable", e);
    }
  },
});
