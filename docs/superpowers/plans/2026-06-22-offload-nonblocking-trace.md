# Non-blocking Offload with Background Trace/Poll — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `POST /civitai/offload/run` return the workflowId as soon as the job is submitted, and run the trace subscription + poll-to-completion + local finalize in a background thread that pushes progress over the ComfyUI `/ws`.

**Architecture:** Split the current blocking `_offload_run` into `_offload_submit` (synchronous, fast: build + `submit_steps(wait=0)`) and `_offload_finalize` (daemon thread: start trace tail → poll to terminal with `wait=10` → drain tail → run local tail). The route returns right after submit. A new `civitai.offload.status` ws event surfaces background completion/failure to the originating tab.

**Tech Stack:** Python (aiohttp routes inside ComfyUI, `requests`, threads), pytest (no ComfyUI/network), vanilla JS ComfyUI extension.

## Global Constraints

- The package MUST import and pass `pytest` without ComfyUI installed (`from server import PromptServer` is guarded with try/except and treated as a no-op when absent).
- Do NOT add comments by default (user global rule). A short comment is allowed only for non-obvious "why" (e.g. the sid-broadcast fallback, the daemon-thread decoupling). Match the existing density in `server_routes.py`.
- Tests run with: `python -m pytest tests -q` (use the repo's interpreter — `python` on this machine). Output is ruff-formatted.
- `submit_steps` for the offload run uses `wait=0`. The poll loop uses `wait=10` (already the case in `_poll_workflow_to_terminal`).
- Custom ws event name: exactly `civitai.offload.status`. Status payload shape: `{"state": "done"|"error", ...fields}`.

---

### Task 1: `_push_offload_status` helper

**Files:**
- Modify: `civitai_comfy_nodes/server_routes.py` (add helper near `_extract_trace_url`, around the trace-tail helpers)
- Test: `tests/test_server_routes.py`

**Interfaces:**
- Produces: `_push_offload_status(sid: str | None, state: str, **fields) -> None` — best-effort `send_sync("civitai.offload.status", {"state": state, **fields}, sid)`; no-op when ComfyUI's `server` module is unavailable or `send_sync` raises.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server_routes.py`:

```python
def test_push_offload_status_sends_custom_ws_event(monkeypatch):
    import sys
    import types

    class FakeServer:
        def __init__(self):
            self.calls = []

        def send_sync(self, event, data, sid=None):
            self.calls.append((event, data, sid))

    fake_server = FakeServer()
    monkeypatch.setitem(
        sys.modules,
        "server",
        types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake_server)),
    )

    sr._push_offload_status("browser-1", "done", workflowId="wf-1", promptId="p-9")

    assert fake_server.calls == [
        (
            "civitai.offload.status",
            {"state": "done", "workflowId": "wf-1", "promptId": "p-9"},
            "browser-1",
        )
    ]


def test_push_offload_status_is_noop_without_comfy_server():
    # No `server` module installed in the test env -> import fails -> silent no-op.
    sr._push_offload_status("browser-1", "error", message="boom")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_server_routes.py::test_push_offload_status_sends_custom_ws_event -v`
Expected: FAIL with `AttributeError: module 'civitai_comfy_nodes.server_routes' has no attribute '_push_offload_status'`

- [ ] **Step 3: Write minimal implementation**

In `civitai_comfy_nodes/server_routes.py`, add directly below `_extract_trace_url` (before `class _TraceTailHandle`):

```python
def _push_offload_status(sid: str | None, state: str, **fields) -> None:
    """Push a terminal offload status (`done`/`error`) to the originating tab over the local /ws.
    Best-effort: a no-op outside ComfyUI or if the socket is gone."""
    try:
        from server import PromptServer  # ComfyUI runtime
    except Exception:
        return
    try:
        PromptServer.instance.send_sync("civitai.offload.status", {"state": state, **fields}, sid)
    except Exception:
        _log.debug("Could not push offload status", exc_info=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_server_routes.py -k push_offload_status -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add civitai_comfy_nodes/server_routes.py tests/test_server_routes.py
git commit -m "feat(offload): add _push_offload_status ws helper"
```

---

### Task 2: `_offload_submit` (fast submit, wait=0)

**Files:**
- Modify: `civitai_comfy_nodes/server_routes.py` (add `_offload_submit`; leave the old `_offload_run` in place for now)
- Test: `tests/test_server_routes.py`

**Interfaces:**
- Consumes: `offload.build_custom_comfy_offload(...)`, `OrchestrationClient.submit_steps(steps, *, wait, whatif)`, `resolve_config`, `stored_min_vram_gb`, `stored_use_sage_attention`.
- Produces: `_offload_submit(prompt: dict, selected_node_ids: list[str] | None, workflow: dict | None, *, whatif: bool, do_tail: bool) -> dict` returning `{"config": ClientConfig, "build": OffloadBuild, "workflow": dict}`. Requests `trace="binary"` iff `do_tail`. Always submits with `wait=0`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server_routes.py`:

```python
def test_offload_submit_uses_wait_zero_and_requests_trace(monkeypatch):
    import types

    from civitai_comfy_nodes import client as client_mod
    from civitai_comfy_nodes import config as config_mod
    from civitai_comfy_nodes import offload as offload_mod

    captured = {}

    class FakeClient:
        def __init__(self, config):
            self.config = config
            self.upload_blob_file = lambda *a, **k: None

        def submit_steps(self, steps, *, wait, whatif=False):
            captured["wait"] = wait
            captured["whatif"] = whatif
            captured["steps"] = steps
            return {"id": "wf-1", "status": "queued"}

    fake_build = types.SimpleNamespace(
        steps=[{"$type": "customComfy", "input": {}}], as_dict=lambda: {"ok": True}
    )

    def fake_build_offload(prompt, **kwargs):
        captured["trace"] = kwargs.get("trace")
        return fake_build

    monkeypatch.setattr(
        config_mod, "resolve_config", lambda interactive=False: types.SimpleNamespace(token="t", timeout_minutes=5)
    )
    monkeypatch.setattr(config_mod, "stored_min_vram_gb", lambda: 24)
    monkeypatch.setattr(config_mod, "stored_use_sage_attention", lambda: False)
    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(offload_mod, "build_custom_comfy_offload", fake_build_offload)

    result = sr._offload_submit({"3": {}}, None, None, whatif=False, do_tail=True)

    assert captured["wait"] == 0
    assert captured["trace"] == "binary"
    assert result["workflow"] == {"id": "wf-1", "status": "queued"}
    assert result["build"] is fake_build
    assert result["config"].token == "t"


def test_offload_submit_omits_trace_when_not_tailing(monkeypatch):
    import types

    from civitai_comfy_nodes import client as client_mod
    from civitai_comfy_nodes import config as config_mod
    from civitai_comfy_nodes import offload as offload_mod

    captured = {}

    class FakeClient:
        def __init__(self, config):
            self.upload_blob_file = lambda *a, **k: None

        def submit_steps(self, steps, *, wait, whatif=False):
            captured["whatif"] = whatif
            return {"id": "wf-2"}

    def fake_build_offload(prompt, **kwargs):
        captured["trace"] = kwargs.get("trace")
        return types.SimpleNamespace(steps=[], as_dict=lambda: {})

    monkeypatch.setattr(
        config_mod, "resolve_config", lambda interactive=False: types.SimpleNamespace(token="t", timeout_minutes=5)
    )
    monkeypatch.setattr(config_mod, "stored_min_vram_gb", lambda: 24)
    monkeypatch.setattr(config_mod, "stored_use_sage_attention", lambda: False)
    monkeypatch.setattr(client_mod, "OrchestrationClient", FakeClient)
    monkeypatch.setattr(offload_mod, "build_custom_comfy_offload", fake_build_offload)

    sr._offload_submit({"3": {}}, None, None, whatif=True, do_tail=False)

    assert captured["trace"] is None
    assert captured["whatif"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_server_routes.py -k offload_submit -v`
Expected: FAIL with `AttributeError: ... has no attribute '_offload_submit'`

- [ ] **Step 3: Write minimal implementation**

In `civitai_comfy_nodes/server_routes.py`, add above the existing `_offload_run`:

```python
def _offload_submit(
    prompt: dict,
    selected_node_ids: list[str] | None,
    workflow: dict | None,
    *,
    whatif: bool,
    do_tail: bool,
) -> dict:
    """Build the customComfy offload and submit it with wait=0 so the caller gets the workflow id
    back immediately. The long-running poll + local replay happen later in `_offload_finalize`."""
    from . import offload
    from .client import OrchestrationClient
    from .config import resolve_config, stored_min_vram_gb, stored_use_sage_attention

    config = resolve_config(interactive=False)
    client = OrchestrationClient(config)
    build = offload.build_custom_comfy_offload(
        prompt,
        selected_node_ids=selected_node_ids,
        workflow=workflow,
        token=config.token,
        trace="binary" if do_tail else None,
        min_vram_gb=stored_min_vram_gb(),
        use_sage_attention=stored_use_sage_attention(),
        upload_blob_file=client.upload_blob_file,
    )
    submitted = client.submit_steps(build.steps, wait=0, whatif=whatif)
    return {"config": config, "build": build, "workflow": submitted}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_server_routes.py -k offload_submit -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add civitai_comfy_nodes/server_routes.py tests/test_server_routes.py
git commit -m "feat(offload): add _offload_submit (fast submit, wait=0)"
```

---

### Task 3: `_offload_finalize` (background orchestration)

**Files:**
- Modify: `civitai_comfy_nodes/server_routes.py` (add `_offload_finalize`)
- Test: `tests/test_server_routes.py`

**Interfaces:**
- Consumes: `_start_trace_tail(config, workflow, *, sid)` → handle with `.drain()`, `.stop()`; `_poll_workflow_to_terminal(client, workflow, timeout_minutes)`; `_run_local_tail(prompt, offload_result, comfy_base_url, *, client_id)`; `_push_offload_status` (Task 1); `OrchestrationClient`.
- Produces: `_offload_finalize(prompt: dict, build, config, workflow: dict, comfy_base_url: str, *, sid: str | None, do_tail: bool) -> None`. Runs to completion synchronously (the route runs it inside a daemon thread). On success pushes `state="done"` with `workflowId`/`promptId`; on any failure stops the tail (if running) and pushes `state="error"` with `message`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_server_routes.py`:

```python
def _finalize_env(monkeypatch, events, *, poll, local):
    import types

    from civitai_comfy_nodes import client as client_mod

    monkeypatch.setattr(client_mod, "OrchestrationClient", lambda config: object())

    fake_tail = types.SimpleNamespace(
        drain=lambda: events.append("drain"),
        stop=lambda: events.append("stop"),
        summary=lambda: None,
    )
    monkeypatch.setattr(
        sr, "_start_trace_tail", lambda config, wf, sid=None: (events.append("tail"), fake_tail)[1]
    )
    monkeypatch.setattr(sr, "_poll_workflow_to_terminal", poll)
    monkeypatch.setattr(sr, "_run_local_tail", local)
    monkeypatch.setattr(sr, "_push_offload_status", lambda sid, state, **f: events.append(("status", state, sid, f)))


def test_offload_finalize_pushes_done_on_success(monkeypatch):
    import types

    events = []
    _finalize_env(
        monkeypatch,
        events,
        poll=lambda client, wf, timeout: {"id": "wf-1", "status": "succeeded"},
        local=lambda prompt, result, base, client_id=None: {"queue": {"prompt_id": "p-9"}},
    )
    build = types.SimpleNamespace(as_dict=lambda: {"k": "v"})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize({"p": 1}, build, config, {"id": "wf-1"}, "http://localhost:8188", sid="browser-1", do_tail=True)

    assert ("status", "done", "browser-1", {"workflowId": "wf-1", "promptId": "p-9"}) in events
    assert "drain" in events
    assert "stop" not in events


def test_offload_finalize_pushes_error_and_stops_tail_when_poll_fails(monkeypatch):
    import types

    events = []

    def boom(client, wf, timeout):
        raise CivitaiNodeError("poll boom")

    _finalize_env(
        monkeypatch,
        events,
        poll=boom,
        local=lambda *a, **k: {"queue": {"prompt_id": "p-9"}},
    )
    build = types.SimpleNamespace(as_dict=lambda: {})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize({"p": 1}, build, config, {"id": "wf-1"}, "http://x", sid="browser-1", do_tail=True)

    assert ("status", "error", "browser-1", {"message": "poll boom"}) in events
    assert "stop" in events
    assert "drain" not in events


def test_offload_finalize_pushes_error_when_local_tail_fails(monkeypatch):
    import types

    events = []

    def boom_local(prompt, result, base, client_id=None):
        raise CivitaiNodeError("no assets")

    _finalize_env(
        monkeypatch,
        events,
        poll=lambda client, wf, timeout: {"id": "wf-1", "status": "succeeded"},
        local=boom_local,
    )
    build = types.SimpleNamespace(as_dict=lambda: {})
    config = types.SimpleNamespace(timeout_minutes=5, token="t")

    sr._offload_finalize({"p": 1}, build, config, {"id": "wf-1"}, "http://x", sid="browser-1", do_tail=True)

    assert ("status", "error", "browser-1", {"message": "no assets"}) in events
    assert "drain" in events  # poll succeeded, tail drained, then local failed
```

Confirm the test module imports `CivitaiNodeError` (it is used above). Near the top of `tests/test_server_routes.py`, if not already imported, add:

```python
from civitai_comfy_nodes.errors import CivitaiNodeError
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_server_routes.py -k offload_finalize -v`
Expected: FAIL with `AttributeError: ... has no attribute '_offload_finalize'`

- [ ] **Step 3: Write minimal implementation**

In `civitai_comfy_nodes/server_routes.py`, add below `_offload_submit`:

```python
def _offload_finalize(
    prompt: dict,
    build,
    config,
    workflow: dict,
    comfy_base_url: str,
    *,
    sid: str | None,
    do_tail: bool,
) -> None:
    """Background half of an offload run: tail the trace onto the local /ws, poll to completion,
    then download the result and queue the local continuation. Runs in a daemon thread, so it
    reports terminal state via a `civitai.offload.status` ws event instead of an HTTP response."""
    from .client import OrchestrationClient

    client = OrchestrationClient(config)
    tail = _start_trace_tail(config, workflow, sid=sid) if do_tail else None
    try:
        final = _poll_workflow_to_terminal(client, workflow, config.timeout_minutes)
    except Exception as exc:
        if tail is not None:
            tail.stop()
        _push_offload_status(sid, "error", message=str(exc))
        _log.warning("offload finalize: poll failed (%s)", exc, exc_info=True)
        return
    if tail is not None:
        tail.drain()

    offload_result = {"workflow": final, "offload": build.as_dict()}
    try:
        local = _run_local_tail(prompt, offload_result, comfy_base_url, client_id=sid)
    except Exception as exc:
        _push_offload_status(sid, "error", message=str(exc))
        _log.warning("offload finalize: local tail failed (%s)", exc, exc_info=True)
        return

    _push_offload_status(
        sid,
        "done",
        workflowId=final.get("id") or final.get("workflowId"),
        promptId=((local or {}).get("queue") or {}).get("prompt_id"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_server_routes.py -k offload_finalize -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add civitai_comfy_nodes/server_routes.py tests/test_server_routes.py
git commit -m "feat(offload): add _offload_finalize background orchestration"
```

---

### Task 4: Rewire the route to submit-then-spawn; remove `_offload_run`

**Files:**
- Modify: `civitai_comfy_nodes/server_routes.py` (`_civitai_offload_run` route body; delete `_offload_run`)

**Interfaces:**
- Consumes: `_offload_submit` (Task 2), `_offload_finalize` (Task 3), `_extract_trace_url`, `threading` (already imported).
- Produces: route returns `{"workflow", "offload", "traceUrl"?}` immediately; spawns `_offload_finalize` in a daemon thread when `runLocalTail and not whatif`.

- [ ] **Step 1: Replace the route body**

In `civitai_comfy_nodes/server_routes.py`, replace the entire `_civitai_offload_run` handler (currently the `body.get("wait")` clamp through the final `return web.json_response(result)`) with:

```python
    @_server.routes.post("/civitai/offload/run")
    async def _civitai_offload_run(request):
        body = await request.json()
        prompt = body.get("prompt") or body.get("output")
        if not isinstance(prompt, dict):
            return web.json_response({"error": "prompt must be a ComfyUI API prompt object"}, status=400)
        selected = body.get("selectedNodeIds") or body.get("selected_node_ids") or None
        if selected is not None and not isinstance(selected, list):
            return web.json_response({"error": "selectedNodeIds must be an array"}, status=400)
        workflow = body.get("workflow")
        if workflow is not None and not isinstance(workflow, dict):
            return web.json_response({"error": "workflow must be a serialized ComfyUI workflow object"}, status=400)
        whatif = bool(body.get("whatif", False))
        run_local_tail = bool(body.get("runLocalTail", False))
        live_progress = bool(body.get("liveProgress", True))
        client_id = body.get("clientId")
        if not isinstance(client_id, str):
            client_id = None
        comfy_base_url = f"{request.scheme}://{request.host}"
        selected_ids = [str(node_id) for node_id in selected] if selected else None
        run_background = run_local_tail and not whatif
        do_tail = run_background and live_progress
        loop = asyncio.get_event_loop()
        try:
            submit = await loop.run_in_executor(
                None,
                lambda: _offload_submit(prompt, selected_ids, workflow, whatif=whatif, do_tail=do_tail),
            )
        except CivitaiAuthError:
            return web.json_response({"error": "auth_required"}, status=401)
        except CivitaiNodeError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=502)

        submitted_workflow = submit["workflow"]
        response = {"workflow": submitted_workflow, "offload": submit["build"].as_dict()}
        trace_url = _extract_trace_url(submitted_workflow)
        if trace_url:
            response["traceUrl"] = trace_url
        if run_background:
            threading.Thread(
                target=_offload_finalize,
                args=(prompt, submit["build"], submit["config"], submitted_workflow, comfy_base_url),
                kwargs={"sid": client_id, "do_tail": do_tail},
                name="civitai-offload-finalize",
                daemon=True,
            ).start()
        return web.json_response(response)
```

- [ ] **Step 2: Delete the now-unused `_offload_run`**

Remove the whole `def _offload_run(...)` function (the block starting `def _offload_run(` and ending at its final `return result`). `_offload_submit` + `_offload_finalize` replace it. Leave `_run_local_tail`, `_poll_workflow_to_terminal`, `_start_trace_tail`, and `_TraceTailHandle` untouched.

- [ ] **Step 3: Verify import + no stale references**

Run: `python -c "import civitai_comfy_nodes.server_routes as sr; assert not hasattr(sr, '_offload_run'); assert hasattr(sr, '_offload_submit') and hasattr(sr, '_offload_finalize')"`
Expected: no output, exit 0.

Run: `git grep -n "_offload_run" -- '*.py'`
Expected: no matches.

- [ ] **Step 4: Run the full backend suite**

Run: `python -m pytest tests -q`
Expected: all pass (the route itself is not unit-tested — it needs aiohttp/ComfyUI — but the extracted helpers are covered).

- [ ] **Step 5: Commit**

```bash
git add civitai_comfy_nodes/server_routes.py
git commit -m "feat(offload): return workflowId on submit; finalize in background thread"
```

---

### Task 5: Frontend — fast submit + status toast listener

**Files:**
- Modify: `web/civitai-offload.js` (`submitOffload`; add a `civitai.offload.status` listener in `setup()`)

**Interfaces:**
- Consumes: server route returns `{workflow, offload, traceUrl?}` quickly; server pushes `civitai.offload.status` (`{state, ...}`) when the background finishes.
- Produces: button re-enables right after submit; a success/failure toast fires when the background completes.

- [ ] **Step 1: Stop long-polling on submit**

In `web/civitai-offload.js`, in `submitOffload`, delete the `payload.wait = 5;` line so the body no longer requests a submit long-poll:

Before:
```js
async function submitOffload(payload) {
  payload.wait = 5;
  payload.runLocalTail = true;
  // Replay the remote run's /ws frames (progress + previews) onto this tab's canvas.
  payload.liveProgress = true;
  payload.clientId = api.clientId;
```

After:
```js
async function submitOffload(payload) {
  payload.runLocalTail = true;
  // Replay the remote run's /ws frames (progress + previews) onto this tab's canvas.
  payload.liveProgress = true;
  payload.clientId = api.clientId;
```

- [ ] **Step 2: Register the status listener in `setup()`**

In `web/civitai-offload.js`, inside `app.registerExtension({ ... setup() { ... } })`, immediately after `installQueuePromptOverride();`, add:

```js
    api.addCustomEventListener("civitai.offload.status", (event) => {
      const detail = event.detail || {};
      if (detail.state === "error") {
        toast("error", "Civitai offload failed", String(detail.message || "Unknown error"));
      } else if (detail.state === "done") {
        const wf = detail.workflowId ? ` (${detail.workflowId})` : "";
        toast("success", "Civitai offload complete", `Results downloaded${wf}.`);
      }
    });
```

(`addCustomEventListener` — not `addEventListener` — is required: the ComfyUI ws client only dispatches non-builtin event types that were registered through it.)

- [ ] **Step 3: Lint check the JS (syntax sanity)**

Run: `node --check web/civitai-offload.js`
Expected: no output, exit 0. (If `node` is unavailable, open the file and confirm the listener sits inside `setup()` and braces balance.)

- [ ] **Step 4: Commit**

```bash
git add web/civitai-offload.js
git commit -m "feat(offload): re-enable button on submit; toast background completion"
```

---

### Task 6: Verification (suite + manual live check)

**Files:** none (verification only)

- [ ] **Step 1: Full suite + import-without-comfy guard**

Run: `python -m pytest tests -q`
Expected: all pass.

Run: `python -c "import civitai_comfy_nodes"`
Expected: imports cleanly with no ComfyUI installed.

- [ ] **Step 2: Manual live check in real ComfyUI (record evidence)**

This is the true success criterion and cannot be unit-tested. With the repo symlinked into `ComfyUI/custom_nodes/`, `CIVITAI_API_TOKEN` exported, and the orchestrator reachable:

1. Open a graph, click **Run on Civitai**.
2. Confirm the queue button **re-enables within ~1s** and the "Submitted to Civitai (workflow X)" toast shows the workflowId immediately (not at the end).
3. Confirm the canvas shows **live node progress/highlighting** during the remote run (replayed `executing`/`progress`/`executed`).
4. Confirm a **"Civitai offload complete"** toast at the end, and the output node shows the downloaded result.
5. Capture the browser devtools **WS frames** to confirm `executing`/`progress`/`executed`/`logs` arrive during the run (proves the events reach the stream — the original bug).

If step 3 shows node highlighting but no aggregate progress bar, that's the known `progress_state` prompt_id gap noted in the spec — a separate follow-up, not a blocker for this plan.

- [ ] **Step 3: Final commit (if any manual-fix tweaks were needed)**

```bash
git add -A
git commit -m "chore(offload): verification fixups"
```

---

## Self-Review

**Spec coverage:**
- Submit `wait=0`, return workflowId fast → Task 2 + Task 4. ✓
- Background subscribe-to-trace + poll(`wait=10`) + local finalize → Task 3 (reuses existing `_start_trace_tail`/`_poll_workflow_to_terminal`/`_run_local_tail`). ✓
- `_push_offload_status` custom ws event + frontend toast for the error-handling shift → Task 1 + Task 5. ✓
- Frontend re-enables on submit → Task 5 (drop `wait`, existing `finally` re-enables). ✓
- sid-targeted replay default; sid=None broadcast fallback → preserved (route passes `client_id`/None straight through to `_start_trace_tail`/`send_sync`). ✓
- whatif stays synchronous → Task 4 (`run_background = run_local_tail and not whatif`). ✓
- Tests pass without ComfyUI/network → all new tests monkeypatch the `server`/client/offload deps. ✓
- Live-progress rendering verified, not pre-solved → Task 6 Step 2. ✓

**Placeholder scan:** none — every code/test step shows full content.

**Type consistency:** `_offload_submit` returns `{"config","build","workflow"}`, consumed verbatim by the route and passed positionally into `_offload_finalize(prompt, build, config, workflow, comfy_base_url, *, sid, do_tail)`. `_push_offload_status(sid, state, **fields)` call sites use `message=`/`workflowId=`/`promptId=`, matching the helper. `_start_trace_tail(config, workflow, *, sid)` and `_poll_workflow_to_terminal(client, workflow, timeout_minutes)` and `_run_local_tail(prompt, offload_result, comfy_base_url, *, client_id)` match their existing definitions. Custom event name `civitai.offload.status` identical in helper, tests, and frontend. ✓
