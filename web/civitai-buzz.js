// Live Buzz cost meter for "Run on Civitai" offload runs, shown ON the offload marker node(s) via a
// read-only node WIDGET — NOT by scraping ComfyUI's Vue queue / job-details DOM, and NOT by drawing
// on the litegraph canvas (recent ComfyUI renders nodes as DOM, so canvas drawing is occluded). The
// widget API is ComfyUI's own cross-version/skin surface for on-node content (same path as
// civitai-status.js), so this survives frontend and theme changes. The backend
// (server_routes._send_buzz) pushes `civitai.buzz` ws frames during the offload poll: the pinned
// rate while running and the final charge at terminal. We tick the per-second cost between frames
// (anchored on the worker's first compute frame, replayed via the trace tail) and snap to the final
// value. The anchor node ids (the Civitai Offload Start/End markers) are published by
// civitai-offload.js; runs without markers report their cost via the completion toast instead.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const PHASE = { IDLE: "idle", PREPARING: "preparing", RUNNING: "running", FINAL: "final" };
const TICK_MS = 250;
const WIDGET_NAME = "civitai_buzz";

const run = {
  phase: PHASE.IDLE,
  rate: 0,
  anchorMs: null,
  clockOffset: 0,
  finalCost: null,
  display: null,
};
let timer = null;

const nowAligned = () => Date.now() + run.clockOffset;
const num = (v) => (typeof v === "number" && isFinite(v) ? v : null);

// --- shared state from civitai-offload.js -------------------------------------------------------
// The offload id scopes the meter to the offloaded run (native lifecycle events fire for local runs
// too). The anchor ids are the offload marker nodes the cost shows on.
function activeOffloadId() {
  try {
    return window.__civitaiActiveOffloadWorkflowId ?? null;
  } catch (e) {
    return null;
  }
}

function isOffloadRun(promptId) {
  const id = activeOffloadId();
  return id != null && promptId != null && String(promptId) === String(id);
}

function anchorNodes() {
  let ids = null;
  try {
    ids = window.__civitaiActiveOffloadAnchors;
  } catch (e) {
    ids = null;
  }
  if (!Array.isArray(ids) || !ids.length) return [];
  const graph = app.graph;
  if (!graph?.getNodeById) return [];
  const nodes = [];
  for (const id of ids) {
    const node = graph.getNodeById(id) ?? graph.getNodeById(Number(id));
    if (node) nodes.push(node);
  }
  return nodes;
}

// --- on-node widget rendering -------------------------------------------------------------------

function buzzWidget(node) {
  let widget = node.widgets?.find((w) => w.name === WIDGET_NAME);
  if (widget) return widget;
  // multiline gives a textarea inputEl that renders the value in both canvas- and DOM-node ComfyUI
  // (a single-line STRING widget exposes no inputEl in DOM-node mode); mirrors civitai-status.js.
  widget = ComfyWidgets["STRING"](node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
  if (widget.inputEl) {
    widget.inputEl.readOnly = true;
    widget.inputEl.style.opacity = "0.9";
    widget.inputEl.style.fontWeight = "600";
    widget.inputEl.style.fontSize = "11px";
    widget.inputEl.style.border = "none";
    widget.inputEl.style.background = "transparent";
  }
  // The cost is a run result, not a graph input — keep it out of the saved workflow.
  widget.serializeValue = () => undefined;
  return widget;
}

function label() {
  return `⚡ ${run.display == null ? 0 : run.display} Buzz`;
}

// Update both the widget model and (in DOM-node ComfyUI) its input element, so the value shows live
// in either render mode. Best-effort: a missing ComfyWidgets / failed create must never break a run.
function render() {
  const idle = run.phase === PHASE.IDLE;
  const text = idle ? "" : label();
  const color = run.phase === PHASE.FINAL ? "#69db7c" : "#ffd43b";
  for (const node of anchorNodes()) {
    try {
      const widget = buzzWidget(node);
      widget.value = text;
      if (widget.inputEl) {
        widget.inputEl.value = text;
        widget.inputEl.style.color = color;
      }
    } catch (e) {
      /* widget API unavailable — fall back to the completion toast */
    }
  }
  try {
    app.graph?.setDirtyCanvas(true, true);
  } catch (e) {
    /* canvas not ready */
  }
}

// --- meter state machine ------------------------------------------------------------------------

function stopTick() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function startTick() {
  stopTick();
  timer = setInterval(() => {
    if (run.phase !== PHASE.RUNNING || run.rate <= 0 || run.anchorMs == null) return;
    const elapsed = Math.max(0, (nowAligned() - run.anchorMs) / 1000);
    run.display = Math.ceil(Math.max(1, run.rate * elapsed));
    render();
  }, TICK_MS);
}

function anchorCompute() {
  if (run.phase === PHASE.FINAL || run.rate <= 0) return;
  if (run.phase === PHASE.RUNNING && run.anchorMs != null) return;
  if (run.anchorMs == null) run.anchorMs = nowAligned();
  run.phase = PHASE.RUNNING;
  startTick();
  render();
}

function beginPreparing() {
  if (run.phase === PHASE.PREPARING || run.phase === PHASE.RUNNING) return;
  run.phase = PHASE.PREPARING;
  run.anchorMs = null;
  run.finalCost = null;
  run.display = 0;
  stopTick();
  render();
}

function freeze() {
  run.phase = PHASE.FINAL;
  stopTick();
  if (run.finalCost != null) run.display = Math.ceil(Math.max(0, run.finalCost));
  render();
}

function onLifecycleEnd() {
  if (run.phase === PHASE.IDLE) return;
  freeze();
}

function onBuzz(d) {
  if (!d || typeof d !== "object") return;
  const rate = num(d.buzz_per_second);
  if (rate != null && rate > 0) run.rate = rate;
  const computedAt = num(d.computed_at);
  if (computedAt != null) run.clockOffset = computedAt - Date.now();

  if (d.terminal) {
    // A seed frame (reconnect replay of a past job) only warms state; don't settle the live meter.
    if (!d.seed) {
      run.finalCost = num(d.estimated_cost);
      freeze();
    }
    return;
  }

  if (run.phase === PHASE.IDLE || run.phase === PHASE.FINAL) beginPreparing();
  const startedAt = num(d.started_at);
  if (startedAt != null) {
    run.anchorMs = startedAt;
    anchorCompute();
  } else {
    render();
  }
}

function nodeOf(detail) {
  if (detail && typeof detail === "object") return detail.node ?? detail.display_node ?? null;
  return detail ?? null;
}

function attachListeners() {
  // Native lifecycle events fire for LOCAL runs too — scope them to the offloaded prompt_id (carried,
  // rewritten, on every forwarded trace frame). The custom civitai.buzz frames are already
  // offload-only, so onBuzz needs no such guard.
  api.addEventListener("execution_start", (e) => {
    if (isOffloadRun(e.detail?.prompt_id)) beginPreparing();
  });
  api.addEventListener("progress", (e) => {
    if (isOffloadRun(e.detail?.prompt_id)) anchorCompute();
  });
  api.addEventListener("executing", (e) => {
    if (isOffloadRun(e.detail?.prompt_id) && nodeOf(e.detail) != null) anchorCompute();
  });
  api.addEventListener("execution_success", (e) => {
    if (isOffloadRun(e.detail?.prompt_id)) onLifecycleEnd();
  });
  api.addEventListener("execution_error", (e) => {
    if (isOffloadRun(e.detail?.prompt_id)) onLifecycleEnd();
  });
  api.addEventListener("execution_interrupted", (e) => {
    if (isOffloadRun(e.detail?.prompt_id)) onLifecycleEnd();
  });
  api.addEventListener("civitai.buzz", (e) => onBuzz(e.detail));
  api.addEventListener("unhandled", (e) => {
    if (e.detail && e.detail.type === "civitai.buzz") onBuzz(e.detail.detail);
  });

  let sock = null;
  const sniff = () => {
    const s = api.socket;
    if (!s || s === sock) return;
    sock = s;
    s.addEventListener("message", (ev) => {
      if (typeof ev.data !== "string" || ev.data.indexOf("civitai.buzz") === -1) return;
      try {
        const m = JSON.parse(ev.data);
        if (m && m.type === "civitai.buzz") onBuzz(m.data);
      } catch {
        /* not ours */
      }
    });
  };
  sniff();
  api.addEventListener("status", sniff);
  api.addEventListener("reconnected", sniff);
}

app.registerExtension({
  name: "civitai.buzz",
  async setup() {
    attachListeners();
  },
});
