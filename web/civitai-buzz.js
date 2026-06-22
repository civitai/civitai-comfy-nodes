// Live Buzz cost meter for "Run on Civitai" offload runs. The backend
// (server_routes._send_buzz) pushes `civitai.buzz` ws frames during the offload
// poll — the pinned rate while running and the final charge at terminal. This
// extension shows a ⚡ cost line on the running job's progress row (ComfyUI's
// QueueProgressOverlay), ticking the per-second cost between frames (anchored on
// the worker's first compute frame, replayed via the trace tail) and snapping to
// the final value. Nothing is shown when idle.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PHASE = { IDLE: "idle", PREPARING: "preparing", RUNNING: "running", FINAL: "final" };
const TICK_MS = 250;

const run = {
  phase: PHASE.IDLE,
  rate: 0,
  anchorMs: null,
  clockOffset: 0,
  finalCost: null,
  display: null,
};
let timer = null;
let badge = null;
const txnCache = new Map(); // prompt_id -> { transactions: [{amount,currency,refund}], total }

function injectStyles() {
  const style = document.createElement("style");
  style.textContent = `
    .cvz-buzz-q { position:absolute; right:4px; top:50%; transform:translateY(-50%); z-index:2;
      padding:1px 6px; border-radius:9px; font:600 11px/1.4 system-ui,sans-serif; white-space:nowrap;
      color:#ffd43b; background:rgba(255,212,59,.18); border:1px solid rgba(255,212,59,.45); }
    .cvz-buzz-q[data-final="1"] { color:#69db7c; background:rgba(105,219,124,.18); border-color:rgba(105,219,124,.5); }`;
  document.head.appendChild(style);
}

const fmtRate = (r) => (Number.isInteger(r) ? String(r) : r.toFixed(2).replace(/\.?0+$/, ""));

// Overlay the live value at the right of ComfyUI's running-progress panel (the "Total: …%" row).
// We attach to the text COLUMN (which is position:relative and — unlike the row — Vue doesn't
// reconcile our node away) and absolutely-position it, so it shows through the run without adding
// a line to the fixed-height row. Re-attached if Vue re-renders; gone when the panel is.
function render() {
  const totalSpan = document.querySelector('span[title^="Total:"]');
  const col = totalSpan && (totalSpan.closest(".flex-col") || totalSpan.parentElement?.parentElement);
  if (!col || run.phase === PHASE.IDLE) { if (badge) badge.remove(); return; }
  if (!badge) { badge = document.createElement("span"); badge.className = "cvz-buzz-q"; }
  if (badge.parentElement !== col) col.appendChild(badge);
  badge.textContent = `⚡ ${run.display == null ? 0 : run.display}`;
  badge.dataset.final = run.phase === PHASE.FINAL ? "1" : "0";
  badge.title = run.rate > 0 ? `${fmtRate(run.rate)} Buzz/sec` : "";
}

// --- job-details popup: a "Transactions" row from the cached terminal frame -----

function txnText(entry) {
  const parts = (entry?.transactions || [])
    .map((t) => `${t.amount} ${t.currency ? t.currency + " " : ""}Buzz${t.refund ? " refunded" : ""}`);
  if (parts.length) return parts.join(", ");
  return entry?.total != null ? `${entry.total} Buzz` : null;
}

// ComfyUI's Job Details flyout is a static panel; when one is open, find its Job ID, match the
// cached transactions, and append a styled "Transactions" row.
function injectJobDetails() {
  for (const panel of document.querySelectorAll(".bg-interface-panel-surface")) {
    if (panel.querySelector(".cvz-buzz-txn")) continue;
    const container = panel.querySelector(".flex.flex-col");
    if (!container) continue;
    const jobIdLabel = [...container.querySelectorAll(".grid > div")]
      .find((d) => /^\s*Job ID\s*$/.test(d.textContent || ""));
    const promptId = jobIdLabel?.nextElementSibling?.querySelector("span")?.textContent?.trim();
    const text = promptId && txnText(txnCache.get(promptId));
    if (!text) continue;

    const grid = document.createElement("div");
    grid.className = "grid grid-cols-2 items-center gap-2 cvz-buzz-txn";
    const label = document.createElement("div");
    label.className = jobIdLabel.className;
    label.textContent = "Transactions";
    const value = jobIdLabel.nextElementSibling.cloneNode(false);
    const span = document.createElement("span");
    span.className = "block min-w-0 truncate";
    span.textContent = text;
    span.title = text;
    value.replaceChildren(span);
    grid.append(label, value);
    container.appendChild(grid);
  }
}

const nowAligned = () => Date.now() + run.clockOffset;
const num = (v) => (typeof v === "number" && isFinite(v) ? v : null);

function stopTick() { if (timer) { clearInterval(timer); timer = null; } }

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
    run.finalCost = num(d.estimated_cost);
    if (d.prompt_id) txnCache.set(String(d.prompt_id), {
      transactions: Array.isArray(d.transactions) ? d.transactions : [],
      total: num(d.cost_total),
    });
    freeze();
    return;
  }

  if (run.phase === PHASE.IDLE || run.phase === PHASE.FINAL) beginPreparing();
  const startedAt = num(d.started_at);
  if (startedAt != null) { run.anchorMs = startedAt; anchorCompute(); }
  else render();
}

function nodeOf(detail) {
  if (detail && typeof detail === "object") return detail.node ?? detail.display_node ?? null;
  return detail ?? null;
}

function attachListeners() {
  api.addEventListener("progress", () => anchorCompute());
  api.addEventListener("executing", (e) => { if (nodeOf(e.detail) != null) anchorCompute(); });
  api.addEventListener("execution_success", onLifecycleEnd);
  api.addEventListener("execution_error", onLifecycleEnd);
  api.addEventListener("execution_interrupted", onLifecycleEnd);
  api.addEventListener("civitai.buzz", (e) => onBuzz(e.detail));
  api.addEventListener("unhandled", (e) => { if (e.detail && e.detail.type === "civitai.buzz") onBuzz(e.detail.detail); });

  let sock = null;
  const sniff = () => {
    const s = api.socket;
    if (!s || s === sock) return;
    sock = s;
    s.addEventListener("message", (ev) => {
      if (typeof ev.data !== "string" || ev.data.indexOf("civitai.buzz") === -1) return;
      try { const m = JSON.parse(ev.data); if (m && m.type === "civitai.buzz") onBuzz(m.data); } catch { /* not ours */ }
    });
  };
  sniff();
  api.addEventListener("status", sniff);
  api.addEventListener("reconnected", sniff);
}

app.registerExtension({
  name: "civitai.buzz",
  async setup() {
    injectStyles();
    attachListeners();
    // Keep the badge alive through the overlay's Vue re-renders and after the tick stops.
    setInterval(() => { if (run.phase !== PHASE.IDLE) render(); }, 300);

    // Inject the Transactions row whenever a Job Details popup appears (debounced; idempotent).
    let injectPending = false;
    new MutationObserver(() => {
      if (injectPending) return;
      injectPending = true;
      setTimeout(() => { injectPending = false; injectJobDetails(); }, 100);
    }).observe(document.body, { childList: true, subtree: true });
  },
});
