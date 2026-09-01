// "Run on Civitai" + "Pay with" wallet picker, mounted right after ComfyUI's Run button. The
// wallet choice lives in the pack settings (not the browser) so the backend can pin the charge.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SESSION_PROXY_RE = /^\/sessions\/[^/]+\/proxy\//;
const ACCOUNTS = [
  { key: "blue", label: "Blue Buzz", color: "#4dabf7" },
  { key: "green", label: "Green Buzz", color: "#40c057" },
  { key: "yellow", label: "Yellow Buzz", color: "#f59f00" },
];
const MOUNT_INTERVAL_MS = 500;
const SETTLE_REFRESH_MS = [0, 1500, 4000];

let group = null;
let runButton = null;
let payPill = null;
let menuEl = null;
let selected = ACCOUNTS[0].key;
let balances = null; // { blue, green, yellow } | null while unknown
let signedOut = false;
let offloadEnabled = true;
let refreshing = false;
let refreshTimers = [];

function injectStyles() {
  const style = document.createElement("style");
  style.textContent = `
    .cvr-group { display: inline-flex; align-items: center; gap: 6px; }
    .cvr-run, .cvr-pay {
      display: inline-flex; align-items: center; gap: 7px; height: 32px; box-sizing: border-box;
      padding: 0 12px; border-radius: 8px; border: 1px solid transparent; font: 600 13px/1 inherit;
      white-space: nowrap; cursor: pointer; user-select: none;
    }
    .cvr-run { background: #2563eb; border-color: #2563eb; color: #fff; }
    .cvr-run:hover { background: #1d4ed8; border-color: #1d4ed8; }
    .cvr-run[disabled] { opacity: .6; cursor: progress; }
    .cvr-run .cvg-civitai-icon::before { width: 1.1em; height: 1.1em; vertical-align: -0.15em; }
    .cvr-pay {
      background: var(--comfy-input-bg, #27272a); border-color: var(--border-color, #3f3f46);
      color: var(--input-text, #e4e4e7);
    }
    .cvr-pay:hover { border-color: #2563eb; }
    .cvr-dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }
    .cvr-bal { opacity: .85; font-weight: 500; font-variant-numeric: tabular-nums; }
    .cvr-caret { opacity: .6; font-size: 10px; }
    .cvr-menu {
      position: fixed; z-index: 11000; min-width: 220px; padding: 5px; border-radius: 10px;
      background: var(--comfy-menu-bg, #202020); color: var(--input-text, #e4e4e7);
      border: 1px solid var(--border-color, #3f3f46); box-shadow: 0 12px 40px rgba(0, 0, 0, .5);
      font: 13px system-ui, sans-serif;
    }
    .cvr-head { display: flex; align-items: center; justify-content: space-between; }
    .cvr-title { padding: 4px 8px 6px; font-size: 11px; font-weight: 600; opacity: .7; }
    .cvr-refresh {
      display: flex; align-items: center; justify-content: center; width: 22px; height: 22px;
      margin-right: 4px; padding: 0; border: none; border-radius: 6px; background: transparent;
      color: inherit; opacity: .7; font-size: 14px; line-height: 1; cursor: pointer;
    }
    .cvr-refresh:hover { opacity: 1; background: rgba(127, 127, 127, .2); }
    .cvr-refresh-icon { display: inline-block; line-height: 1; }
    .cvr-refresh-icon.spin { animation: cvr-spin .7s linear infinite; }
    @keyframes cvr-spin { to { transform: rotate(360deg); } }
    .cvr-opt {
      display: flex; align-items: center; gap: 8px; padding: 7px 8px; border-radius: 7px;
      font-weight: 500; cursor: pointer;
    }
    .cvr-opt:hover { background: rgba(127, 127, 127, .2); }
    .cvr-opt .cvr-bal { margin-left: auto; }
    .cvr-check { width: 14px; text-align: center; color: #40c057; }
    .cvr-note { padding: 6px 8px 4px; font-size: 11px; opacity: .7; }
  `;
  document.head.appendChild(style);
}

function toast(severity, summary, detail) {
  try {
    app.extensionManager.toast.add({ severity, summary, detail, life: 5000 });
  } catch (e) {
    console[severity === "error" ? "error" : "log"](`[civitai-run-bar] ${summary}: ${detail ?? ""}`);
  }
}

const account = (key) => ACCOUNTS.find((a) => a.key === key) || ACCOUNTS[0];
const balanceOf = (key) => (balances && typeof balances[key] === "number" ? balances[key] : null);
const formatBalance = (n) => (typeof n === "number" && isFinite(n) ? n.toLocaleString() : "—");

function dot(color) {
  const el = document.createElement("span");
  el.className = "cvr-dot";
  el.style.background = color;
  return el;
}

function renderPill() {
  if (!payPill) return;
  const a = account(selected);
  payPill.replaceChildren();
  const label = document.createElement("span");
  label.textContent = a.label;
  const bal = document.createElement("span");
  bal.className = "cvr-bal";
  bal.textContent = signedOut ? "sign in" : formatBalance(balanceOf(selected));
  const caret = document.createElement("span");
  caret.className = "cvr-caret";
  caret.textContent = "▾";
  payPill.append(dot(a.color), label, bal, caret);
  payPill.title = signedOut
    ? "Connect Civitai in the sidebar tab to see balances"
    : "Buzz wallet Civitai runs are charged to";
  if (menuEl && !refreshing) {
    fillMenu(menuEl);
    positionMenu();
  }
}

function fillMenu(menu) {
  menu.replaceChildren();
  const head = document.createElement("div");
  head.className = "cvr-head";
  const title = document.createElement("span");
  title.className = "cvr-title";
  title.textContent = "Pay with";
  const refresh = document.createElement("button");
  refresh.className = "cvr-refresh";
  refresh.type = "button";
  refresh.title = "Refresh balances";
  refresh.onclick = manualRefresh;
  const refreshIcon = document.createElement("span");
  refreshIcon.className = "cvr-refresh-icon" + (refreshing ? " spin" : "");
  refreshIcon.textContent = "↻";
  refresh.appendChild(refreshIcon);
  head.append(title, refresh);
  menu.appendChild(head);
  for (const a of ACCOUNTS) {
    const opt = document.createElement("div");
    opt.className = "cvr-opt";
    const check = document.createElement("span");
    check.className = "cvr-check";
    check.textContent = a.key === selected ? "✓" : "";
    const label = document.createElement("span");
    label.textContent = a.label;
    const bal = document.createElement("span");
    bal.className = "cvr-bal";
    bal.textContent = formatBalance(balanceOf(a.key));
    opt.append(check, dot(a.color), label, bal);
    opt.onclick = () => choose(a.key);
    menu.appendChild(opt);
  }
  if (signedOut) {
    const note = document.createElement("div");
    note.className = "cvr-note";
    note.textContent = "Connect Civitai (sidebar tab) to see balances.";
    menu.appendChild(note);
  }
}

// Portaled to <body> and fixed-positioned: the draggable Run toolbar establishes its own stacking
// context, so a nested menu renders behind the queue panel and gets clipped.
function openMenu() {
  closeMenu();
  menuEl = document.createElement("div");
  menuEl.className = "cvr-menu";
  menuEl.onclick = (e) => e.stopPropagation();
  fillMenu(menuEl);
  document.body.appendChild(menuEl);
  positionMenu();
  fetchBalances();
}

function closeMenu() {
  if (menuEl) {
    menuEl.remove();
    menuEl = null;
  }
}

function positionMenu() {
  if (!menuEl || !payPill) return;
  const r = payPill.getBoundingClientRect();
  const width = menuEl.offsetWidth || 220;
  const left = Math.max(8, Math.min(r.right - width, window.innerWidth - width - 8));
  menuEl.style.left = `${Math.round(left)}px`;
  menuEl.style.top = `${Math.round(r.bottom + 6)}px`;
}

async function choose(key) {
  const previous = selected;
  selected = key;
  closeMenu();
  renderPill();
  try {
    const res = await fetch("/civitai/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ buzzAccount: key }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || `HTTP ${res.status}`);
  } catch (e) {
    selected = previous;
    renderPill();
    toast("error", "Could not save the Buzz wallet", String(e.message || e));
  }
}

async function fetchBalances() {
  try {
    const res = await fetch("/civitai/buzz/accounts", { cache: "no-store" });
    if (res.status === 401) {
      signedOut = true;
      balances = null;
    } else if (res.ok) {
      const data = await res.json();
      signedOut = false;
      balances = { blue: data.blue, green: data.green, yellow: data.yellow };
    } else {
      return; // transient: keep the last known numbers
    }
    renderPill();
  } catch (e) {
    /* transient; picker still usable */
  }
}

async function manualRefresh(e) {
  e.stopPropagation();
  if (refreshing) return;
  refreshing = true;
  if (menuEl) fillMenu(menuEl);
  const minSpin = new Promise((r) => setTimeout(r, 700));
  try {
    await Promise.all([fetchBalances(), minSpin]);
  } finally {
    refreshing = false;
    if (menuEl) fillMenu(menuEl);
  }
}

// Charges are post-billed and settle a beat after a run finishes, so re-read a few times.
function refreshBalancesSoon() {
  refreshTimers.forEach(clearTimeout);
  refreshTimers = SETTLE_REFRESH_MS.map((ms) => setTimeout(fetchBalances, ms));
}

async function loadConfig() {
  try {
    const cfg = await (await fetch("/civitai/config")).json();
    if (ACCOUNTS.some((a) => a.key === cfg.buzzAccount)) selected = cfg.buzzAccount;
    offloadEnabled = cfg.enableOffload !== false;
  } catch (e) {
    /* defaults */
  }
}

function buildGroup() {
  group = document.createElement("div");
  group.className = "cvr-group";
  runButton = document.createElement("button");
  runButton.className = "cvr-run";
  runButton.type = "button";
  runButton.title =
    "Run on Civitai's GPUs: the selected nodes, else the nodes inside a group titled \"Civitai\", else the whole graph";
  const icon = document.createElement("span");
  icon.className = "cvg-civitai-icon";
  runButton.append(icon, document.createTextNode(" Run on Civitai"));
  runButton.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    const offload = window.civitaiOffload;
    if (!offload) return toast("warn", "Run on Civitai unavailable", "The offload extension did not load.");
    offload.run(runButton);
  });
  payPill = document.createElement("button");
  payPill.className = "cvr-pay";
  payPill.type = "button";
  payPill.addEventListener("click", (e) => {
    e.stopPropagation();
    if (menuEl) closeMenu();
    else openMenu();
  });
  group.append(runButton, payPill);
  renderPill();
}

// Keep the group right after the native Run cluster through the toolbar's Vue re-renders.
function ensureMounted() {
  const queueGroup =
    document.querySelector(".actionbar-container .queue-button-group") ||
    document.querySelector('[data-testid="queue-button"]')?.closest(".queue-button-group");
  if (!queueGroup || !offloadEnabled) {
    if (group?.isConnected) group.remove();
    closeMenu();
    return;
  }
  if (queueGroup.nextElementSibling !== group) queueGroup.after(group);
  if (menuEl) positionMenu();
}

app.registerExtension({
  name: "civitai.runBar",
  async setup() {
    if (SESSION_PROXY_RE.test(location.pathname)) return; // hosted sessions ship their own picker
    injectStyles();
    await loadConfig();
    buildGroup();
    ensureMounted();
    setInterval(ensureMounted, MOUNT_INTERVAL_MS);
    document.addEventListener("click", closeMenu);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeMenu();
    });
    window.addEventListener("resize", positionMenu);
    document.addEventListener("civitai.config.changed", async (e) => {
      if (!("enableOffload" in (e.detail || {}))) return;
      await loadConfig();
      ensureMounted();
    });
    api.addEventListener("execution_success", refreshBalancesSoon);
    api.addEventListener("execution_error", refreshBalancesSoon);
    api.addCustomEventListener("civitai.offload.status", refreshBalancesSoon);
    api.addCustomEventListener("civitai.buzz", (e) => {
      if (e.detail?.terminal) refreshBalancesSoon();
    });
    fetchBalances();
  },
});
