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


NODE_CLASS_MAPPINGS = {
    "CivitaiAuth": CivitaiAuth,
    "CivitaiChatSimple": CivitaiChatSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiAuth": "Civitai Auth",
    "CivitaiChatSimple": "Civitai Chat (Simple)",
}
