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
                        "tooltip": "auto: api_token > env > stored OAuth > browser login. "
                        "api_key: never trigger a browser login. oauth: ignore api_token input.",
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

        workflow_id, raw_json = output[-2], output[-1]
        workflow = json.loads(raw_json)
        step_output = (workflow.get("steps") or [{}])[0].get("output") or {}
        choices = step_output.get("choices") or []
        if not choices:
            raise CivitaiNodeError("Chat completion returned no choices")
        text = (choices[0].get("message") or {}).get("content") or ""
        return (text, workflow_id, raw_json)


AIR_EXAMPLE = "urn:air:sdxl:lora:civitai:328553@368189"

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
    """Build a Civitai LoRA stack by AIR. Chain several together (loras → loras), then wire the
    final `loras` output into a recipe node's `loras` / `additional_networks` input."""

    CATEGORY = "Civitai/Loaders"
    FUNCTION = "append"
    RETURN_TYPES = ("CIVITAI_LORAS",)
    RETURN_NAMES = ("loras",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "air": ("STRING", {"default": "", "tooltip": f"LoRA AIR, e.g. {AIR_EXAMPLE}"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "trigger_word": ("STRING", {"default": "", "tooltip": "Optional trigger word / embedding token"}),
                "loras": ("CIVITAI_LORAS", {"tooltip": "Chain from another Civitai LoRA Loader"}),
            },
        }

    def append(self, air, strength, trigger_word="", loras=None):
        stack = list(loras or [])
        air = air.strip()
        if air:
            entry = {"air": air, "strength": strength}
            if trigger_word.strip():
                entry["triggerWord"] = trigger_word.strip()
            stack.append(entry)
        return (stack,)


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


class CivitaiCheckpointLoader:
    """Civitai checkpoint loader. Wire `air` into a Civitai recipe node to run in the cloud, OR
    wire model/clip/vae into local nodes (KSampler, CLIP Text Encode, …) — in that case the
    checkpoint is downloaded from Civitai and loaded locally. The download only happens when the
    model/clip/vae outputs are actually connected, so cloud-only use stays free."""

    CATEGORY = "Civitai/Loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("air", "model", "clip", "vae")
    _LOCAL_SLOTS = (1, 2, 3)  # model/clip/vae output indices

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "air": (
                    "STRING",
                    {"default": "", "tooltip": "Checkpoint AIR (use the Browse Civitai button)"},
                ),
            },
            "optional": {"api_config": ("CIVITAI_CONFIG", {})},
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def _local_outputs_consumed(cls, prompt, unique_id) -> bool:
        """True if model/clip/vae are wired downstream (so we should download + load locally)."""
        if not prompt or unique_id is None:
            return False
        uid = str(unique_id)
        for node in prompt.values():
            for value in (node.get("inputs") or {}).values():
                if not (isinstance(value, list) and len(value) == 2):
                    continue
                if str(value[0]) == uid and value[1] in cls._LOCAL_SLOTS:
                    return True
        return False

    @classmethod
    def IS_CHANGED(cls, air, api_config=None, prompt=None, unique_id=None):
        # Re-run when the local outputs get (dis)connected, not just when the AIR changes.
        return f"{(air or '').strip()}|{cls._local_outputs_consumed(prompt, unique_id)}"

    def load(self, air, api_config=None, prompt=None, unique_id=None):
        air = (air or "").strip()
        if not air:
            raise CivitaiNodeError("No checkpoint AIR set — use the Browse Civitai button to pick one.")
        model = clip = vae = None
        if self._local_outputs_consumed(prompt, unique_id):
            import os

            from . import local_models, oauth

            token = (api_config or {}).get("api_token") or os.environ.get("CIVITAI_API_TOKEN")
            if not token:
                token = oauth.get_valid_access_token()  # best-effort; public models need no token
            path = local_models.download_model(air, folder="checkpoints", token=token)
            model, clip, vae = local_models.load_checkpoint(path)
        return (air, model, clip, vae)


NODE_CLASS_MAPPINGS = {
    "CivitaiAuth": CivitaiAuth,
    "CivitaiChatSimple": CivitaiChatSimple,
    "CivitaiLoraLoader": CivitaiLoraLoader,
    "CivitaiControlNet": CivitaiControlNet,
    "CivitaiCheckpointLoader": CivitaiCheckpointLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiAuth": "Civitai Auth",
    "CivitaiChatSimple": "Civitai Chat (Simple)",
    "CivitaiLoraLoader": "Civitai LoRA Loader",
    "CivitaiControlNet": "Civitai ControlNet",
    "CivitaiCheckpointLoader": "Civitai Checkpoint Loader",
}
