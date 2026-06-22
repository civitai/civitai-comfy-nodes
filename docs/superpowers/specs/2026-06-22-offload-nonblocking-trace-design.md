# Non-blocking offload with early workflowId + background trace/poll

**Date:** 2026-06-22
**Branch:** offload-hybrid-live-trace

## Problem

When a user clicks "Run on Civitai", `POST /civitai/offload/run` blocks for the
*entire* run: build → `submit_steps(wait=5)` → `_poll_workflow_to_terminal` →
`_run_local_tail` (download + publish + continuation). The browser's `fetch` does not
resolve until all of that finishes, so:

- The queue button stays disabled for the whole run.
- The frontend never sees the `workflowId`/`traceUrl` until the end.
- Live `/ws` progress is supposed to come from the server-side trace tail, but the
  synchronous flow means nothing useful reaches the UI until completion.

The fix is to decouple the HTTP response from the long-running work: return the
`workflowId` as soon as the workflow is submitted, and run the trace subscription +
poll-to-completion + local finalize in the background, pushing live progress over the
existing ComfyUI `/ws` replay.

## Approach (chosen: server-side background)

Split the one blocking request into **submit (synchronous, fast) + finalize (background)**.

```
POST /civitai/offload/run
  ├─ build + submit_steps(wait=0)          -> returns workflow.id immediately
  ├─ spawn background daemon thread --------------------------+
  └─ return {workflow, offload, traceUrl?}  -> frontend re-enables now
                                                              |
   background thread (_offload_finalize):                     v
     1. _start_trace_tail(sid)   subscribe to trace, replay frames onto /ws (exists)
     2. _poll_workflow_to_terminal(wait=10)  drive to completion (exists)
     3. tail.drain()
     4. _run_local_tail()        download result, publish executed/history, queue continuation
     5. send_sync("civitai.offload.status", {state, ...}, sid)  completion/error toast
```

Rejected alternative — *frontend-driven*: route only submits; the browser opens the
trace stream itself, parses frames in JS, and polls a status endpoint. Cleaner sid story
but requires porting the binary frame parser to JS and adding status/finalize endpoints.
Not chosen — the server-side tail + local-tail already exist and work; this reuses them.

## Backend changes — `civitai_comfy_nodes/server_routes.py`

- Split `_offload_run` into:
  - `_offload_submit(prompt, selected_node_ids, workflow, whatif, *, live_progress)` —
    builds the offload and calls `submit_steps(wait=0)`. Returns `{workflow, build, config}`.
    Errors here propagate synchronously (HTTP 400/502 as today).
  - `_offload_finalize(prompt, build, config, workflow, comfy_base_url, *, sid)` — runs in
    a daemon thread: `_start_trace_tail(sid)` → `_poll_workflow_to_terminal(wait=10)` →
    `tail.drain()` → `_run_local_tail()`. Wrapped so any exception is caught and reported.
- Route `_civitai_offload_run`:
  - `whatif`: unchanged — submit synchronously and return the cost estimate (no tail, no
    background).
  - normal: submit synchronously, spawn `_offload_finalize` in a daemon thread, return
    `{workflow, offload, traceUrl}` immediately. `traceUrl` is included only if already
    present on the submit response (usually null this early; the tail resolves it by polling).
- Add `_push_offload_status(sid, state, detail)` → `PromptServer.instance.send_sync(
  "civitai.offload.status", {"state": state, ...}, sid)`. Best-effort; no-op outside ComfyUI.
- Submit uses `wait=0`. The poll loop already uses `wait=10`. The request body's `wait`
  field becomes unused for the submit (kept for back-compat / ignored).

## Frontend changes — `web/civitai-offload.js`

- `submitOffload` resolves right after submit, so the button re-enables and the
  "Submitted to Civitai (workflow X)" toast fires immediately (existing `finally`/toast).
- Stop sending `payload.wait = 5` (or send 0) — submit no longer long-polls.
- Add `api.addEventListener("civitai.offload.status", handler)` to toast background
  completion and, importantly, surface background failures (those no longer return as an
  HTTP error).
- `offloadQueueResult` already falls back to `workflow.id` when `data.local` is absent — no
  change needed.

## Error handling shift

Failures **after** submit (poll timeout, download failure, continuation failure) now
surface via the `civitai.offload.status` `/ws` event instead of an HTTP error response.
Submit-time failures stay synchronous HTTP errors.

## Live-progress rendering (verify, do not pre-solve)

Replayed events carry the **remote** `prompt_id` but the **same node ids** (offload
preserves node ids). Legacy `progress`/`executing`/`executed` render per-node by id, so
node progress bars + highlighting should appear. `progress_state` keys on `prompt_id` and
may be ignored by the newer frontend — acceptable, since per-node events cover the canvas.
Confirm live; only add `prompt_id` remapping if the canvas demonstrably needs it.

## Decisions (defaults, changeable)

- Trace replay stays **server-side and sid-targeted** to the submitting tab (not broadcast).
  If `client_id` is missing, `send_sync(sid=None)` already broadcasts as a fallback.
- **No Cancel button** in this pass. The early `workflowId` newly makes
  `PUT /workflows/{id} {status: canceled}` feasible, but that is separable scope.

## Testing

- Rework `_offload_run` tests in `tests/test_server_routes.py` for the submit/finalize split:
  - Route returns before the workflow is terminal (submit result only).
  - `_offload_finalize` runs the tail + poll + local tail.
  - `civitai.offload.status` pushed on success and on error.
  - `whatif` path unchanged (synchronous cost estimate, no background thread).
- Existing `trace_tail` unit tests are unaffected.
- `python -m pytest tests -q` stays green without ComfyUI/network.
