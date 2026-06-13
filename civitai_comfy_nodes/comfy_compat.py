"""Guarded access to the ComfyUI runtime so the package also imports under plain pytest."""

import tempfile

try:
    import comfy.model_management
    import comfy.utils

    IN_COMFY = True
except ImportError:
    IN_COMFY = False


def check_interrupted() -> None:
    """Raise ComfyUI's interrupt exception if the user pressed cancel; no-op outside ComfyUI."""
    if IN_COMFY:
        comfy.model_management.throw_exception_if_processing_interrupted()


class _NullProgressBar:
    def update_absolute(self, value: int, total: int | None = None) -> None:
        pass


def progress_bar(total: int = 100):
    if IN_COMFY:
        return comfy.utils.ProgressBar(total)
    return _NullProgressBar()


def get_temp_dir() -> str:
    if IN_COMFY:
        try:
            import folder_paths

            return folder_paths.get_temp_directory()
        except ImportError:
            pass
    return tempfile.gettempdir()


def video_from_file(path: str):
    """Wrap a video file path in ComfyUI's VIDEO type."""
    try:
        from comfy_api.latest import InputImpl
    except ImportError as e:
        raise RuntimeError(
            "VIDEO outputs require the ComfyUI runtime (comfy_api is not importable in this environment)."
        ) from e
    return InputImpl.VideoFromFile(path)
