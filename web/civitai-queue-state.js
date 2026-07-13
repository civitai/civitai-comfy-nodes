// Queue-phase labels for "Run on Civitai" offload runs. The offloaded job runs
// on the orchestrator, but the node pack injects a synthetic row into ComfyUI's
// native queue (server_routes._inject_running_queue), where it surfaces as
// "Running" from the first poll. While the assigned worker is downloading the
// job's models (orchestration "preparing") that row's label is rewritten to
// "Downloading models… NN%"; once claimed ("processing") it reads "Starting…"
// until the worker's first real execution frame (relayed through the trace
// tail), after which the vanilla progress UI takes over.
//
// The backend delivers `civitai.queue_state` frames over the local /ws on phase
// changes and download-progress ticks (server_routes._send_queue_state). The
// injected row's wire status stays `in_progress` (the frontend zod-rejects any
// status outside pending|in_progress|completed|failed|cancelled), so the phase
// label must be patched into the DOM — and actively restored, because Vue won't
// re-render a row whose polled data never changed.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const DOWNLOADING = "Downloading models…";
const STARTING = "Starting…";
// Vanilla texts we may replace: the queued labels, plus "Running" — the injected
// row reports in_progress, so the sidebar renders it "Running" from the first
// poll, which is exactly the "Downloading…"/"Starting…" window. Those labels are
// locale-dependent, so resolve them through the editor's vue-i18n (mounted on
// #vue-app); the English regex covers the window before the app is up.
const VANILLA_EN = /^(In queue|Job queued|Running)/;
const VANILLA_KEYS = ["queue.inQueue", "queue.jobAddedToQueue", "g.running"];
let vanillaCache = { at: 0, texts: [] };
function vanillaTexts() {
  if (Date.now() - vanillaCache.at < 2000) return vanillaCache.texts;
  const texts = [];
  try {
    const t = document.getElementById("vue-app")?.__vue_app__?.config?.globalProperties?.$t;
    if (t) for (const key of VANILLA_KEYS) {
      const s = t(key);
      if (typeof s === "string" && s && s !== key) texts.push(s);
    }
  } catch { /* app not up yet */ }
  vanillaCache = { at: Date.now(), texts };
  return texts;
}
const isVanilla = (text) => VANILLA_EN.test(text) || vanillaTexts().some((s) => text.startsWith(s));
const TERMINAL = new Set(["succeeded", "failed", "expired", "canceled", "cancelled"]);

const jobs = new Map(); // prompt_id -> { status, progress, running }

// The label this prompt should show right now, or null for the vanilla text.
function labelFor(entry) {
  if (!entry) return null;
  if (entry.status === "preparing") {
    if (typeof entry.progress !== "number" || !isFinite(entry.progress)) return DOWNLOADING;
    return `${DOWNLOADING} ${Math.min(99, Math.max(0, Math.round(entry.progress * 100)))}%`;
  }
  if (entry.status === "processing" && !entry.running) return STARTING;
  return null;
}

function onQueueState(d) {
  if (!d || typeof d !== "object" || !d.prompt_id) return;
  const id = String(d.prompt_id);
  const status = String(d.status || "").toLowerCase();
  if (TERMINAL.has(status)) jobs.delete(id);
  else
    jobs.set(id, {
      status,
      progress: d.progress,
      // A regression out of processing (claim expiry) means the next run boots
      // fresh; only a still-processing prompt keeps its running mark.
      running: status === "processing" && jobs.get(id)?.running === true,
    });
  render();
}

// First real compute activity: stop labeling, the native progress UI owns the row.
function markRunning(promptId) {
  const mark = (entry) => { if (entry.status === "processing") entry.running = true; };
  if (promptId != null && jobs.has(String(promptId))) mark(jobs.get(String(promptId)));
  else if (promptId == null) jobs.forEach(mark); // older frames omit prompt_id
  render();
}

// --- DOM patching -----------------------------------------------------------

function findVanillaLabel(root) {
  for (const el of root.querySelectorAll("span,div,p")) {
    if (el.children.length > 0) continue;
    const text = (el.textContent || "").trim();
    if (isVanilla(text) || text.startsWith(DOWNLOADING) || text.startsWith(STARTING)) return el;
  }
  return null;
}

function patch(el, text) {
  if (!el.dataset.cvqOrig) el.dataset.cvqOrig = el.textContent;
  el.classList.add("cvq-dl-label");
  if (el.textContent !== text) el.textContent = text;
}

function restore(el) {
  const text = el.textContent || "";
  if ((text.startsWith(DOWNLOADING) || text.startsWith(STARTING)) && el.dataset.cvqOrig)
    el.textContent = el.dataset.cvqOrig;
  el.classList.remove("cvq-dl-label");
  delete el.dataset.cvqOrig;
}

function render() {
  const labeled = [...jobs.values()].map(labelFor).filter(Boolean);

  // Sidebar/asset rows are keyed by job id — patch (or restore) each tracked row.
  for (const row of document.querySelectorAll("[data-job-id]")) {
    const text = labelFor(jobs.get(row.getAttribute("data-job-id")));
    const el = row.querySelector(".cvq-dl-label") || (text ? findVanillaLabel(row) : null);
    if (!el) continue;
    if (text) patch(el, text);
    else restore(el);
  }

  // The floating progress overlay shows the same label without a job-id hook.
  // Only safe to patch when exactly one prompt wants a label (no ambiguity).
  for (const el of document.querySelectorAll(".cvq-dl-label")) {
    if (el.closest("[data-job-id]")) continue;
    if (labeled.length !== 1) restore(el);
  }
  if (labeled.length === 1) {
    for (const el of document.querySelectorAll("span,p")) {
      if (el.closest("[data-job-id]") || el.children.length > 0) continue;
      const text = (el.textContent || "").trim();
      if (isVanilla(text) || el.classList.contains("cvq-dl-label")) patch(el, labeled[0]);
    }
  }
}

// --- wiring (mirrors civitai-buzz.js) --------------------------------------

function nodeOf(detail) {
  if (detail && typeof detail === "object") return detail.node ?? detail.display_node ?? null;
  return detail ?? null;
}

function attachApi(api) {
  api.addEventListener("civitai.queue_state", (e) => onQueueState(e.detail));
  api.addEventListener("unhandled", (e) => {
    if (e.detail && e.detail.type === "civitai.queue_state") onQueueState(e.detail.detail);
  });

  // Worker compute signals end the "Starting…" window. The synthetic executing
  // frame the backend emits on the processing transition carries node:null, so
  // it deliberately doesn't count.
  api.addEventListener("progress", (e) => markRunning(e.detail && e.detail.prompt_id));
  api.addEventListener("executing", (e) => {
    if (nodeOf(e.detail) != null)
      markRunning(e.detail && typeof e.detail === "object" ? e.detail.prompt_id : null);
  });
  for (const type of ["execution_success", "execution_error", "execution_interrupted"])
    api.addEventListener(type, (e) => {
      if (e.detail && e.detail.prompt_id) jobs.delete(String(e.detail.prompt_id));
      render();
    });

  // Primary, version-proof path: sniff the raw socket, re-attaching on reconnect.
  let sock = null;
  const sniff = () => {
    const s = api.socket;
    if (!s || s === sock) return;
    sock = s;
    s.addEventListener("message", (ev) => {
      if (typeof ev.data !== "string" || ev.data.indexOf("civitai.queue_state") === -1) return;
      try { const m = JSON.parse(ev.data); if (m && m.type === "civitai.queue_state") onQueueState(m.data); } catch { /* not ours */ }
    });
  };
  sniff();
  api.addEventListener("status", sniff);
  api.addEventListener("reconnected", sniff);

  // On (re)attach the live phase frame may have already fired; native /api/jobs
  // prunes passthrough fields, so the node pack exposes the still-active phases
  // on its own route — one fetch restores the label instead of a bare "Running".
  fetch("/civitai/offload/active")
    .then((r) => (r.ok ? r.json() : null))
    .then((body) => {
      for (const job of (body && body.jobs) || []) {
        if (job.civitai_orch_status)
          onQueueState({ prompt_id: job.id, status: job.civitai_orch_status, progress: job.civitai_preparation_progress });
      }
    })
    .catch(() => { /* label just waits for the next live frame */ });
}

function startRestoreLoops() {
  // Vue re-renders drop or reset patched nodes — keep re-applying while any
  // prompt is preparing/starting, and re-check on DOM churn (debounced).
  setInterval(() => { if (jobs.size > 0) render(); }, 500);
  let renderPending = false;
  new MutationObserver(() => {
    if (renderPending || jobs.size === 0) return;
    renderPending = true;
    setTimeout(() => { renderPending = false; render(); }, 100);
  }).observe(document.body, { childList: true, subtree: true });
}

app.registerExtension({
  name: "civitai.queue-state",
  async setup() {
    attachApi(api);
    startRestoreLoops();
  },
});
