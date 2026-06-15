"""Shared execution engine for all generated recipe nodes.

Generated classes are declarative: they define RECIPE/STEP_TYPE/DISCRIMINATOR,
FIELDS (widget -> wire field), OUTPUTS (wire field -> Comfy type) and INPUT_TYPES.
Everything behavioral — payload building, submit, poll, blob conversion — lives here.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass

from . import comfy_compat, conversions
from .client import TERMINAL_STATUSES, OrchestrationClient
from .config import resolve_config
from .errors import CivitaiNodeError, workflow_failure_message

# Propagates to ComfyUI's root logger, so these show in the same console as "[INFO] got prompt".
logger = logging.getLogger("civitai_comfy_nodes")

# Long-poll the status GET so it returns the moment the workflow finishes. The fetch runs in a
# background thread and is checked every INTERRUPT_TICK seconds, so ComfyUI Cancel stays responsive
# even while the request blocks server-side. MIN_POLL_INTERVAL throttles the loop if the server
# doesn't honor `wait` (returns immediately) so we never tight-loop the API.
LONGPOLL_WAIT = 10
MIN_POLL_INTERVAL = 3
INTERRUPT_TICK = 0.5
PROGRESS_BY_STATUS = {"unassigned": 5, "preparing": 8, "scheduled": 10, "processing": 40}


@dataclass(frozen=True)
class F:
    """Input field: widget value -> wire payload."""

    api: str
    kind: str = "value"  # value | json | image_inline | image_list | image_url | video_url | audio_url


@dataclass(frozen=True)
class O:
    """Output field: step output -> return slot(s)."""

    api: str
    kind: str  # image | image_list | video | audio | audio_or_video | string | json


class CivitaiRecipeNodeBase:
    RECIPE = ""
    STEP_TYPE = ""
    DISCRIMINATOR: dict = {}
    FIELDS: dict = {}
    OUTPUTS: tuple = ()
    FUNCTION = "run"
    CATEGORY = "Civitai"
    # Mark every recipe node as a graph output so it's runnable standalone — many recipes
    # (echo, chat, analysis) return only STRING/JSON with no downstream sink, which would
    # otherwise trip ComfyUI's "Prompt has no outputs" guard.
    OUTPUT_NODE = True

    def run(self, api_config=None, **widgets):
        config = resolve_config(api_config)
        client = OrchestrationClient(config)
        payload = self._build_payload(client, widgets)
        payload.update(self.DISCRIMINATOR)

        workflow = client.submit_workflow(self.STEP_TYPE, payload, wait=5)
        workflow_id = workflow.get("id", "?")
        logger.info("Civitai %s: submitted workflow %s (status: %s)", self.RECIPE, workflow_id, workflow.get("status"))
        workflow = self._poll(client, workflow, timeout_minutes=config.timeout_minutes)
        if workflow.get("status") != "succeeded":
            raise CivitaiNodeError(workflow_failure_message(workflow))

        logger.info("Civitai workflow %s succeeded%s", workflow_id, self._cost_summary(workflow))
        step = (workflow.get("steps") or [{}])[0]
        output = step.get("output") or {}
        results = self._convert_outputs(client, output)
        wid = workflow.get("id", "")
        # `ui.civitai_status` is read by web/civitai-status.js to show the id + per-currency cost
        # on the node itself; `result` carries the actual output slots.
        return {
            "ui": {"civitai_status": [{"workflow_id": wid, "cost": self._cost_text(workflow)}]},
            "result": (*results, wid, json.dumps(workflow)),
        }

    def _build_payload(self, client: OrchestrationClient, widgets: dict) -> dict:
        payload = {}
        for widget_name, value in widgets.items():
            field = self.FIELDS.get(widget_name)
            if field is None or value is None:
                continue
            if field.kind == "value":
                if value == "":
                    continue
                payload[field.api] = value
            elif field.kind == "air":
                # AIR string from a Civitai Model Selector socket
                if value:
                    payload[field.api] = value
            elif field.kind == "air_list":
                # list of AIR strings from a Civitai Embedding Selector socket
                airs = [a for a in value if a]
                if airs:
                    payload[field.api] = airs
            elif field.kind == "json":
                if not str(value).strip():
                    continue
                try:
                    payload[field.api] = json.loads(value)
                except json.JSONDecodeError as e:
                    raise CivitaiNodeError(f"Input '{widget_name}' is not valid JSON: {e}") from e
            elif field.kind == "image_inline":
                payload[field.api] = conversions.image_tensor_to_data_url(value)
            elif field.kind == "image_list":
                payload[field.api] = conversions.image_tensor_to_data_urls(value)
            elif field.kind == "image_url":
                payload[field.api] = client.upload_media(conversions.image_tensor_to_png_bytes(value), "image/png")
            elif field.kind == "video_url":
                payload[field.api] = client.upload_media(conversions.video_to_bytes(value), "video/mp4")
            elif field.kind == "audio_url":
                payload[field.api] = client.upload_media(conversions.audio_to_flac_bytes(value), "audio/flac")
            elif field.kind == "lora_array":
                # CIVITAI_LORAS list -> [{air, strength}] (the array-shaped `loras` field)
                entries = [{"air": x["air"], "strength": x.get("strength", 1.0)} for x in value if x.get("air")]
                if entries:
                    payload[field.api] = entries
            elif field.kind == "lora_strength_map":
                # CIVITAI_LORAS list -> {air: strength} (sdcpp ecosystems' dict-of-number loras)
                networks = {x["air"]: x.get("strength", 1.0) for x in value if x.get("air")}
                if networks:
                    payload[field.api] = networks
            elif field.kind == "network_map":
                # CIVITAI_LORAS list -> {air: {strength, triggerWord}} (the `additionalNetworks` map)
                networks = {}
                for x in value:
                    if not x.get("air"):
                        continue
                    params = {"strength": x.get("strength", 1.0)}
                    if x.get("triggerWord"):
                        params["triggerWord"] = x["triggerWord"]
                    networks[x["air"]] = params
                if networks:
                    payload[field.api] = networks
            elif field.kind == "controlnet_array":
                if value:
                    payload[field.api] = list(value)
            else:
                raise CivitaiNodeError(f"Unknown field kind '{field.kind}' for input '{widget_name}'")
        return payload

    def _poll(self, client: OrchestrationClient, workflow: dict, *, timeout_minutes: float) -> dict:
        bar = comfy_compat.progress_bar(100)
        workflow_id = workflow.get("id", "?")
        deadline = time.time() + timeout_minutes * 60
        last_marker = None
        while workflow.get("status") not in TERMINAL_STATUSES:
            if time.time() > deadline:
                self._try_cancel(client, workflow)
                raise CivitaiNodeError(
                    f"Civitai workflow {workflow_id} timed out after {timeout_minutes:g} minutes "
                    f"(status: {workflow.get('status')}). Increase timeout via the Civitai Auth node or "
                    "CIVITAI_COMFY_TIMEOUT."
                )
            progress = self._report_progress(bar, workflow)
            status = workflow.get("status", "")
            preceding = self._preceding_jobs(workflow)
            marker = (status, progress, preceding)
            if marker != last_marker:
                ahead = f", {preceding} job{'' if preceding == 1 else 's'} ahead" if preceding is not None else ""
                logger.info("Civitai workflow %s: %s, %d%%%s", workflow_id, status, progress, ahead)
                last_marker = marker
            try:
                started = time.time()
                workflow = self._interruptible_get(client, workflow_id, wait=LONGPOLL_WAIT)
                # Throttle only if the server returned early without long-polling (wait unsupported).
                if workflow.get("status") not in TERMINAL_STATUSES:
                    self._interruptible_sleep(MIN_POLL_INTERVAL - (time.time() - started))
            except BaseException:
                self._try_cancel(client, workflow)
                raise
        bar.update_absolute(100)
        return workflow

    @staticmethod
    def _interruptible_get(client: OrchestrationClient, workflow_id: str, *, wait: int) -> dict:
        """Long-poll get_workflow in a daemon thread, polling the ComfyUI interrupt flag every tick
        so Cancel is honored within INTERRUPT_TICK even while the request blocks server-side."""
        box: dict = {}

        def fetch():
            try:
                box["result"] = client.get_workflow(workflow_id, wait=wait)
            except BaseException as exc:  # re-raised on the calling thread
                box["error"] = exc

        thread = threading.Thread(target=fetch, daemon=True)
        thread.start()
        while thread.is_alive():
            comfy_compat.check_interrupted()
            thread.join(INTERRUPT_TICK)
        if "error" in box:
            raise box["error"]
        return box["result"]

    @staticmethod
    def _interruptible_sleep(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            comfy_compat.check_interrupted()
            time.sleep(min(INTERRUPT_TICK, max(0.0, end - time.time())))

    @staticmethod
    def _try_cancel(client: OrchestrationClient, workflow: dict) -> None:
        try:
            if workflow.get("id"):
                client.cancel_workflow(workflow["id"])
        except CivitaiNodeError:
            pass

    @staticmethod
    def _report_progress(bar, workflow: dict) -> int:
        status = workflow.get("status", "")
        progress = PROGRESS_BY_STATUS.get(status, 5)
        if status == "processing":
            steps = workflow.get("steps") or [{}]
            rates = [j.get("estimatedProgressRate") or 0 for j in (steps[0].get("jobs") or [])]
            progress += int(55 * max(rates, default=0))
        bar.update_absolute(progress)
        return progress

    # Buzz wallet (accountType) -> display name; the colour IS the Buzz currency.
    _BUZZ_WALLETS = {"yellow": "Yellow", "blue": "Blue", "green": "Green", "fakeRed": "Red"}

    @classmethod
    def _cost_text(cls, workflow: dict) -> str:
        """'16 Blue Buzz' / '11 Blue Buzz, 5 Green Buzz' — one entry per transaction, so a cost
        split across currencies lists each wallet and amount (refunds flagged). Falls back to the
        workflow cost total when no transactions are present (e.g. whatif)."""
        transactions = ((workflow.get("transactions") or {}).get("list")) or []
        parts = []
        for t in transactions:
            wallet = cls._BUZZ_WALLETS.get(t.get("accountType"), str(t.get("accountType") or "?"))
            suffix = " refunded" if t.get("type") == "credit" else ""
            parts.append(f"{t.get('amount')} {wallet} Buzz{suffix}")
        if parts:
            return ", ".join(parts)
        total = (workflow.get("cost") or {}).get("total")
        return f"{total} Buzz" if total is not None else ""

    @classmethod
    def _cost_summary(cls, workflow: dict) -> str:
        """The cost text as a ' — ...' suffix for the success log line."""
        text = cls._cost_text(workflow)
        return f" — {text}" if text else ""

    @staticmethod
    def _preceding_jobs(workflow: dict):
        """How many jobs are ahead in the queue (queuePosition is an object, not a number)."""
        jobs = ((workflow.get("steps") or [{}])[0].get("jobs")) or []
        counts = [
            j["queuePosition"]["precedingJobs"]
            for j in jobs
            if isinstance(j.get("queuePosition"), dict) and j["queuePosition"].get("precedingJobs") is not None
        ]
        return min(counts) if counts else None

    def _convert_outputs(self, client: OrchestrationClient, output: dict) -> list:
        results = []
        for spec in self.OUTPUTS:
            value = output.get(spec.api)
            if spec.kind == "string":
                results.append("" if value is None else str(value))
            elif spec.kind == "json":
                results.append(json.dumps(value))
            elif spec.kind == "image":
                results.append(conversions.bytes_to_image_tensor(self._download(client, value, spec.api)))
            elif spec.kind == "image_list":
                blobs = value or []
                tensors = [conversions.bytes_to_image_tensor(self._download(client, b, spec.api)) for b in blobs]
                results.append(conversions.stack_image_tensors(tensors))
            elif spec.kind == "video":
                results.append(conversions.bytes_to_video_output(self._download(client, value, spec.api)))
            elif spec.kind == "audio":
                results.append(conversions.bytes_to_audio_output(self._download(client, value, spec.api)))
            elif spec.kind == "audio_or_video":
                data = self._download(client, value, spec.api)
                if (value or {}).get("type") == "video":
                    results.extend([None, conversions.bytes_to_video_output(data)])
                else:
                    results.extend([conversions.bytes_to_audio_output(data), None])
            else:
                raise CivitaiNodeError(f"Unknown output kind '{spec.kind}' for output '{spec.api}'")
        return results

    @staticmethod
    def _download(client: OrchestrationClient, blob: dict | None, name: str) -> bytes:
        if not blob:
            raise CivitaiNodeError(f"Workflow succeeded but output '{name}' is missing")
        if blob.get("blockedReason"):
            raise CivitaiNodeError(f"Output '{name}' was blocked: {blob['blockedReason']}")
        if blob.get("available") is False:
            raise CivitaiNodeError(f"Output '{name}' is not available (blob {blob.get('id', '?')})")
        return client.download_blob(blob)
