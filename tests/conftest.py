import os
import tempfile

# Make the package's import-time node registration deterministic: point the settings store at a path
# that won't exist, so load_pack_settings() returns {} and all nodes register regardless of the
# developer's real ~/.civitai/comfy-settings.json. Per-test fixtures still override this via monkeypatch.
os.environ.setdefault(
    "CIVITAI_COMFY_SETTINGS_STORE",
    os.path.join(tempfile.gettempdir(), "civitai-comfy-nodes-tests", "nonexistent-settings.json"),
)
# Keep the model-AIR resolution cache out of the developer's real ~/.civitai during tests.
os.environ.setdefault(
    "CIVITAI_COMFY_MODEL_CACHE",
    os.path.join(tempfile.gettempdir(), "civitai-comfy-nodes-tests", "model-air-cache.json"),
)
