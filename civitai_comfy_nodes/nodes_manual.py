"""Hand-written nodes: auth/config and convenience wrappers around awkward recipes."""

import json

from .base import CivitaiRecipeNodeBase, F
from .errors import CivitaiNodeError


class CivitaiAuth:
    """Explicit auth/endpoint configuration. Optional — without it, nodes use
    CIVITAI_API_TOKEN or the stored OAuth login against the production API."""

    CATEGORY = "Civitai"
    FUNCTION = "configure"
    RETURN_TYPES = ("CIVITAI_CONFIG",)
    RETURN_NAMES = ("api_config",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["auto", "api_key", "oauth"],
                    {
                        "default": "auto",
                        "tooltip": "auto: api_token field > CIVITAI_API_TOKEN env > stored OAuth (no browser). "
                        "api_key: same, never any browser. oauth: sign in via the browser if no token is found.",
                    },
                ),
            },
            "optional": {
                "api_token": ("STRING", {"default": "", "tooltip": "Civitai API token (leave empty to use env/OAuth)"}),
                "base_url": (
                    "STRING",
                    {"default": "", "tooltip": "Orchestration base URL override (e.g. a local dev stack)"},
                ),
                "allow_mature_content": ("BOOLEAN", {"default": False}),
                "timeout_minutes": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 720.0, "step": 1.0}),
            },
        }

    def configure(self, mode, api_token="", base_url="", allow_mature_content=False, timeout_minutes=30.0):
        config = {
            "mode": mode,
            "allow_mature_content": allow_mature_content,
            "timeout_minutes": timeout_minutes,
        }
        if api_token and mode != "oauth":
            config["api_token"] = api_token
        if base_url:
            config["base_url"] = base_url
        return (config,)


class CivitaiChatSimple(CivitaiRecipeNodeBase):
    """Single-turn chat completion without hand-writing the messages JSON."""

    RECIPE = "chatCompletion"
    STEP_TYPE = "chatCompletion"
    CATEGORY = "Civitai/Text"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "workflow_id", "raw_json")
    FIELDS = {
        "model": F("model"),
        "temperature": F("temperature"),
        "max_tokens": F("maxTokens"),
        "messages_json": F("messages", "json"),
    }
    OUTPUTS = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "openai/gpt-oss-120b"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "image": ("IMAGE", {"tooltip": "Optional image to include in the user message"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072, "step": 1}),
                "api_config": ("CIVITAI_CONFIG", {}),
            },
        }

    def run(self, api_config=None, **widgets):
        from . import conversions

        user_prompt = widgets.pop("user_prompt", "")
        system_prompt = widgets.pop("system_prompt", "")
        image = widgets.pop("image", None)

        content: list | str
        if image is not None:
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "imageUrl": {"url": conversions.image_tensor_to_data_url(image)}},
            ]
        else:
            content = user_prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        widgets["messages_json"] = json.dumps(messages)
        output = super().run(api_config=api_config, **widgets)

        result = output["result"]
        workflow_id, raw_json = result[-2], result[-1]
        workflow = json.loads(raw_json)
        step_output = (workflow.get("steps") or [{}])[0].get("output") or {}
        choices = step_output.get("choices") or []
        if not choices:
            raise CivitaiNodeError("Chat completion returned no choices")
        text = (choices[0].get("message") or {}).get("content") or ""
        return {"ui": output.get("ui", {}), "result": (text, workflow_id, raw_json)}


CONTROLNET_PREPROCESSORS = [
    "canny",
    "mlsd",
    "depthZoe",
    "depthAnything",
    "depthAnythingV2",
    "zoeDepthAnything",
    "zoeDepth",
    "midasDepth",
    "leresDepth",
    "metric3dDepth",
    "softedgePidinet",
    "hed",
    "teed",
    "midasNormal",
    "baeNormal",
    "dsineNormal",
    "metric3dNormal",
    "lineartRealistic",
    "lineartStandard",
    "lineartAnime",
    "lineartManga",
    "anyline",
    "scribble",
    "scribbleXdog",
    "scribblePidinet",
    "fakeScribble",
    "openpose",
    "dwpose",
    "oneformerCoco",
    "oneformerAde20k",
    "uniformer",
    "shuffle",
    "tile",
    "gray",
    "rembg",
]


class CivitaiLoraLoader:
    """Civitai multi-LoRA selector. One node holds any number of LoRAs — each with an enable toggle,
    AIR (picked via Browse Civitai), strength and optional trigger word — managed by the rows UI in
    `web/civitai-catalog.js`, which serializes them into the hidden `loras_json` widget. Wire the
    `loras` output into a recipe node's `loras` / `additional_networks` input (chain another selector
    via the `loras` input to combine). Connect MODEL + CLIP to also download every enabled LoRA and
    apply it locally for KSampler etc.; the download only happens when model+clip are wired, so
    cloud-only use stays free."""

    CATEGORY = "Civitai/Loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("CIVITAI_LORAS", "MODEL", "CLIP")
    RETURN_NAMES = ("loras", "model", "clip")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Managed by the rows UI; hidden on the node. JSON: [{air, strength, triggerWord, on}, ...]
                "loras_json": ("STRING", {"default": "[]"}),
            },
            "optional": {
                "loras": ("CIVITAI_LORAS", {"tooltip": "Chain from another Civitai LoRA selector"}),
                "model": ("MODEL", {"tooltip": "Connect a MODEL to apply the stack locally (downloads each LoRA)"}),
                "clip": ("CLIP", {"tooltip": "Connect a CLIP to apply the stack locally"}),
                "api_config": ("CIVITAI_CONFIG", {}),
            },
        }

    @staticmethod
    def _parse_rows(loras_json):
        try:
            rows = json.loads(loras_json) if loras_json else []
        except (TypeError, ValueError):
            return []
        return rows if isinstance(rows, list) else []

    def load(self, loras_json="[]", loras=None, model=None, clip=None, api_config=None):
        stack = list(loras or [])
        for row in self._parse_rows(loras_json):
            if not isinstance(row, dict) or row.get("on") is False:
                continue
            air = (row.get("air") or "").strip()
            if not air:
                continue
            entry = {"air": air, "strength": float(row.get("strength", 1.0))}
            trigger = (row.get("triggerWord") or "").strip()
            if trigger:
                entry["triggerWord"] = trigger
            stack.append(entry)

        # Local mode: model + clip wired in -> download and apply the whole accumulated stack.
        if model is not None and clip is not None and stack:
            import os

            from . import local_models, oauth

            token = (api_config or {}).get("api_token") or os.environ.get("CIVITAI_API_TOKEN")
            if not token:
                token = oauth.get_valid_access_token()
            for item in stack:
                path = local_models.download_model(item["air"], folder="loras", token=token)
                model, clip = local_models.apply_lora(model, clip, path, item.get("strength", 1.0))
        return (stack, model, clip)


class CivitaiControlNet:
    """Build a Civitai ControlNet stack. Chain several (control_nets → control_nets), then wire the
    final `control_nets` output into a recipe node's `control_nets` input."""

    CATEGORY = "Civitai/Loaders"
    FUNCTION = "append"
    RETURN_TYPES = ("CIVITAI_CONTROLNETS",)
    RETURN_NAMES = ("control_nets",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preprocessor": (CONTROLNET_PREPROCESSORS,),
                "image": ("IMAGE", {"tooltip": "Control image (required by the orchestrator)"}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "start_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_step": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "control_nets": ("CIVITAI_CONTROLNETS", {"tooltip": "Chain from another Civitai ControlNet"}),
            },
        }

    def append(self, preprocessor, image, weight, start_step, end_step, control_nets=None):
        from . import conversions

        stack = list(control_nets or [])
        stack.append(
            {
                "preprocessor": preprocessor,
                "weight": weight,
                "startStep": start_step,
                "endStep": end_step,
                "image": conversions.image_tensor_to_data_url(image),
            }
        )
        return (stack,)


class _AnyType(str):
    """The classic '*' Any type: its __ne__ override makes ComfyUI's validate_node_input treat it as
    matching every input type, so the `path` output can wire into any standard loader's file widget
    (ckpt_name, lora_name, vae_name, control_net_name, …)."""

    def __ne__(self, other):
        return False


ANY_TYPE = _AnyType("*")


class CivitaiModelSelector:
    """Pick a Civitai model and drop this *in front of* an existing loader instead of replacing it.
    Outputs the model's `air` (wire into a Civitai recipe node to run in the cloud) and a local
    `path` (wire into a standard loader's file widget, e.g. Load Checkpoint's `ckpt_name`). The
    model is downloaded into the matching ComfyUI model folder only when `path` is connected, so
    AIR/cloud-only use stays free."""

    CATEGORY = "Civitai/Loaders"
    FUNCTION = "select"
    RETURN_TYPES = ("CIVITAI_AIR", ANY_TYPE)
    RETURN_NAMES = ("air", "path")
    _PATH_SLOT = 1  # downloading is driven by the `path` output being wired

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "air": ("STRING", {"default": "", "tooltip": "Model AIR (use the Browse Civitai button)"}),
            },
            "optional": {"api_config": ("CIVITAI_CONFIG", {})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def _path_consumed(cls, prompt, unique_id) -> bool:
        """True if the `path` output is wired downstream (so we should download the file)."""
        if not prompt or unique_id is None:
            return False
        uid = str(unique_id)
        for node in prompt.values():
            for value in (node.get("inputs") or {}).values():
                if not (isinstance(value, list) and len(value) == 2):
                    continue
                if str(value[0]) == uid and value[1] == cls._PATH_SLOT:
                    return True
        return False

    @classmethod
    def IS_CHANGED(cls, air, api_config=None, prompt=None, unique_id=None):
        # Re-run when `path` gets (dis)connected, not just when the AIR changes.
        return f"{(air or '').strip()}|{cls._path_consumed(prompt, unique_id)}"

    def select(self, air, api_config=None, prompt=None, unique_id=None):
        air = (air or "").strip()
        if not air:
            raise CivitaiNodeError("No model AIR set — use the Browse Civitai button to pick one.")
        path = ""
        if self._path_consumed(prompt, unique_id):
            import os

            from . import local_models, oauth

            token = (api_config or {}).get("api_token") or os.environ.get("CIVITAI_API_TOKEN")
            if not token:
                token = oauth.get_valid_access_token()  # best-effort; public models need no token
            folder = local_models.folder_for_air(air)
            full = local_models.download_model(air, folder=folder, token=token)
            path = os.path.basename(full)  # folder-relative name the loader's combo resolves
        return (air, path)

class CivitaiEmbeddingSelector:
    """Pick Civitai textual-inversion embeddings. Chain several (embeddings → embeddings) and wire
    the final `embeddings` output into a recipe node's `embeddings` input."""

    CATEGORY = "Civitai/Loaders"
    FUNCTION = "append"
    RETURN_TYPES = ("CIVITAI_EMBEDDINGS",)
    RETURN_NAMES = ("embeddings",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "air": ("STRING", {"default": "", "tooltip": "Embedding AIR (use the Browse Civitai button)"}),
            },
            "optional": {
                "embeddings": ("CIVITAI_EMBEDDINGS", {"tooltip": "Chain from another Civitai Embedding Selector"}),
            },
        }

    def append(self, air, embeddings=None):
        stack = list(embeddings or [])
        air = (air or "").strip()
        if air:
            stack.append(air)
        return (stack,)


class _CivitaiOffloadPassthrough:
    CATEGORY = "Civitai/Offload"
    FUNCTION = "mark"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("value",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region_id": (
                    "STRING",
                    {
                        "default": "default",
                        "tooltip": "Matching start/end markers delimit one Civitai offload region.",
                    },
                ),
            },
            "optional": {
                "value": (
                    ANY_TYPE,
                    {"tooltip": "Passthrough value. The offload transformer removes this marker before submission."},
                ),
            },
        }

    def mark(self, region_id="default", value=None):
        return (value,)


class CivitaiOffloadStart(_CivitaiOffloadPassthrough):
    """Passthrough marker that starts a graph region intended for Civitai customComfy offload."""


class CivitaiOffloadEnd(_CivitaiOffloadPassthrough):
    """Passthrough marker that ends a graph region intended for Civitai customComfy offload."""


NODE_CLASS_MAPPINGS = {
    "CivitaiAuth": CivitaiAuth,
    "CivitaiChatSimple": CivitaiChatSimple,
    "CivitaiLoraLoader": CivitaiLoraLoader,
    "CivitaiControlNet": CivitaiControlNet,
    "CivitaiModelSelector": CivitaiModelSelector,
    "CivitaiEmbeddingSelector": CivitaiEmbeddingSelector,
    "CivitaiOffloadStart": CivitaiOffloadStart,
    "CivitaiOffloadEnd": CivitaiOffloadEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiAuth": "Civitai Auth",
    "CivitaiChatSimple": "Civitai Chat (Simple)",
    "CivitaiLoraLoader": "Civitai LoRA Selector",
    "CivitaiControlNet": "Civitai ControlNet",
    "CivitaiModelSelector": "Civitai Model Selector",
    "CivitaiEmbeddingSelector": "Civitai Embedding Selector",
    "CivitaiOffloadStart": "Civitai Offload Start",
    "CivitaiOffloadEnd": "Civitai Offload End",
}
