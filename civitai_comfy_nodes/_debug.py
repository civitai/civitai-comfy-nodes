"""Debug logging for civitai-comfy-nodes HTTP requests.

Set DEBUG = True in config.py or set this module-level flag to enable
detailed request/response logging to the console.
"""

DEBUG: bool = True


def debug_log(msg: str) -> None:
    if DEBUG:
        print(f"[civitai-comfy-nodes][DEBUG] {msg}")