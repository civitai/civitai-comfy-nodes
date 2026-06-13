"""Shared execution engine for all generated recipe nodes.

Generated classes are declarative: they define RECIPE/STEP_TYPE/DISCRIMINATOR,
FIELDS (widget -> wire field), OUTPUTS (wire field -> Comfy type) and INPUT_TYPES.
Everything behavioral — payload building, submit, poll, blob conversion — lives here.
"""

import json
import logging
import time
from dataclasses import dataclass

from . import comfy_compat, conversions
from .client import TERMINAL_STATUSES, OrchestrationClient
from .config import resolve_config
from .errors import CivitaiNodeError, workflow_failure_message

# Propagates to ComfyUI's root logger, so these show in the same console as "[INFO] got prompt".
logger = logging.getLogger("civitai_comfy_nodes")

POLL_SCHEDULE = (2, 2, 5, 5, 10, 15)
POLL_MAX_INTERVAL = 30
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

        cost = (workflow.get("cost") or {}).get("total")
        logger.info("Civitai workflow %s succeeded%s", workflow_id, f" · {cost} Buzz" if cost is not None else "")
        step = (workflow.get("steps") or [{}])[0]
        output = step.get("output") or {}
        results = self._convert_outputs(client, output)
        return (*results, workflow.get("id", ""), json.dumps(workflow))

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
        intervals = iter(POLL_SCHEDULE)
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
            queue_pos = self._queue_position(workflow)
            marker = (status, progress, queue_pos)
            if marker != last_marker:
                queue = f", queue position {queue_pos}" if queue_pos else ""
                logger.info("Civitai workflow %s: %s (%d%%%s)", workflow_id, status, progress, queue)
                last_marker = marker
            interval = next(intervals, POLL_MAX_INTERVAL)
            try:
                self._interruptible_sleep(interval)
            except BaseException:
                self._try_cancel(client, workflow)
                raise
            workflow = client.get_workflow(workflow_id)
        bar.update_absolute(100)
        return workflow

    @staticmethod
    def _interruptible_sleep(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            comfy_compat.check_interrupted()
            time.sleep(min(0.5, max(0.0, end - time.time())))

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

    @staticmethod
    def _queue_position(workflow: dict):
        jobs = ((workflow.get("steps") or [{}])[0].get("jobs")) or []
        positions = [j.get("queuePosition") for j in jobs if j.get("queuePosition") is not None]
        return min(positions) if positions else None

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
