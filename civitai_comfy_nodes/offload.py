"""Helpers for submitting local ComfyUI workflows as Civitai customComfy jobs.

The code in here is deliberately import-safe outside ComfyUI. Runtime-only modules like
folder_paths are imported lazily so pytest can exercise the inventory and transform logic.
"""

from __future__ import annotations

import copy
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib import parse

import requests

from .errors import CivitaiNodeError
from .model_resolve import MODEL_EXTENSIONS, LocalModelRecord, resolve_model_air, scan_local_model_files

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only hit on Python 3.10 runtimes
    tomllib = None


_log = logging.getLogger("civitai_comfy_nodes.offload")

OFFLOAD_START_CLASS = "CivitaiOffloadStart"
OFFLOAD_END_CLASS = "CivitaiOffloadEnd"
OFFLOAD_MARKER_CLASSES = {OFFLOAD_START_CLASS, OFFLOAD_END_CLASS}
MODEL_SELECTOR_CLASS = "CivitaiModelSelector"
OUTPUT_NODE_CLASSES = {
    "PreviewImage",
    "SaveAnimatedPNG",
    "SaveAnimatedWEBP",
    "SaveAudio",
    "SaveAudioMP3",
    "SaveAudioOpus",
    "SaveGLB",
    "SaveImage",
    "SaveImageAdvanced",
    "SaveVideo",
    "SaveWEBM",
}
UPLOAD_MEDIA_INPUTS = {
    "LoadImage": {"image": {"image/png", "image/jpeg", "image/webp"}},
    "LoadImageMask": {"image": {"image/png", "image/jpeg", "image/webp"}},
    "LoadImageOutput": {"image": {"image/png", "image/jpeg", "image/webp"}},
    "LoadAudio": {"audio": {"audio/mpeg", "audio/webm", "video/mp4", "video/webm"}},
    "LoadVideo": {"file": {"video/mp4", "video/webm"}},
    "VHS_LoadAudioUpload": {"audio": {"audio/mpeg", "audio/webm"}},
    "VHS_LoadVideo": {"video": {"video/mp4", "video/webm"}},
    "VHS_LoadVideoFFmpeg": {"video": {"video/mp4", "video/webm"}},
}

MODEL_WIDGET_FOLDERS = {
    "ckpt_name": ("checkpoints",),
    "lora_name": ("loras",),
    "vae_name": ("vae",),
    "control_net_name": ("controlnet",),
    "controlnet_name": ("controlnet",),
    "unet_name": ("diffusion_models",),
    "clip_name": ("text_encoders", "clip"),
    "clip_name1": ("text_encoders", "clip"),
    "clip_name2": ("text_encoders", "clip"),
    "clip_vision_name": ("clip_vision",),
}
AIR_RE = re.compile(r"^(?:urn:)?air:", re.IGNORECASE)


@dataclass
class InstalledNodepack:
    folder: str
    registry_id: str | None
    version: str | None
    air: str | None
    package_name: str | None = None
    git_remote: str | None = None
    git_commit: str | None = None
    version_source: str | None = None
    loaded: bool | None = None
    loaded_node_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UploadedInputBlob:
    node_id: str
    input_name: str
    original_name: str
    path: str
    content_type: str
    air: str
    blob_id: str | None = None
    url: str | None = None
    size: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OffloadBuildResult:
    steps: list[dict[str, Any]]
    workflow: dict[str, Any]
    resources: list[str]
    warnings: list[str]
    selected_node_ids: list[str]
    included_node_ids: list[str]
    model_resources: list[dict[str, Any]]
    nodepack_resources: list[dict[str, Any]]
    input_blobs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LocalContinuationBuildResult:
    prompt: dict[str, Any]
    bridge_node_id: str
    tail_node_ids: list[str]
    output_node_ids: list[str]
    remote_source_node_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_git(path: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _read_pyproject_metadata(path: Path) -> dict[str, str]:
    pyproject = path / "pyproject.toml"
    if not pyproject.exists() or tomllib is None:
        return {}
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return {}
    project = data.get("project") if isinstance(data, dict) else None
    if not isinstance(project, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("name", "version"):
        value = str(project.get(key) or "").strip()
        if value:
            result[key] = value
    urls = project.get("urls")
    if isinstance(urls, dict):
        for key in ("Repository", "Source", "Homepage", "repository", "source", "homepage"):
            value = str(urls.get(key) or "").strip()
            if value:
                result["repository"] = value
                break
    return result


def _read_package_json_metadata(path: Path) -> dict[str, str]:
    package_json = path / "package.json"
    if not package_json.exists():
        return {}
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("name", "version", "repository"):
        value = data.get(key)
        if isinstance(value, dict):
            value = value.get("url")
        value = str(value or "").strip()
        if value:
            result[key] = value
    return result


def _github_registry_id_from_url(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("github:"):
        path = text.split(":", 1)[1]
    elif text.startswith("git@github.com:"):
        path = text.split(":", 1)[1]
    else:
        parsed = parse.urlparse(text)
        host = (parsed.hostname or parsed.netloc).lower()
        if host not in {"github.com", "www.github.com"}:
            return None
        path = parsed.path
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) < 2:
        return None
    owner = parts[0].strip().lower()
    repo = parts[1].strip()
    if repo.endswith(".git"):
        repo = repo[:-4]
    repo = repo.lower()
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def _clean_nodepack_version(value: str | None) -> str | None:
    version = str(value or "").strip()
    if not version:
        return None
    if version.startswith("v") and len(version) > 1 and version[1].isdigit():
        version = version[1:]
    if not any(ch.isdigit() for ch in version):
        return None
    if any(ch.isspace() for ch in version) or "@" in version:
        return None
    return version


def _git_tag_version(path: Path) -> str | None:
    output = _run_git(path, ["tag", "--points-at", "HEAD"])
    if not output:
        return None
    for tag in output.splitlines():
        version = _clean_nodepack_version(tag)
        if version:
            return version
    return None


def _infer_nodepack_air(registry_id: str | None, version: str | None) -> str | None:
    if not registry_id or not version:
        return None
    return f"urn:air:comfy:nodepack:comfyregistry:{registry_id}@{version}"


def _loaded_custom_node_state() -> dict[Path, int] | None:
    nodes_module = sys.modules.get("nodes")
    if nodes_module is None:
        return None
    loaded_dirs = getattr(nodes_module, "LOADED_MODULE_DIRS", None)
    node_classes = getattr(nodes_module, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(loaded_dirs, dict):
        return None
    module_node_counts: dict[str, int] = {}
    if isinstance(node_classes, dict):
        for node_cls in node_classes.values():
            relative_module = str(getattr(node_cls, "RELATIVE_PYTHON_MODULE", "") or "")
            if not relative_module.startswith("custom_nodes."):
                continue
            module_name = relative_module.split(".", 2)[1]
            module_node_counts[module_name] = module_node_counts.get(module_name, 0) + 1
    state: dict[Path, int] = {}
    for module_name, module_dir in loaded_dirs.items():
        try:
            resolved = Path(str(module_dir)).resolve()
        except OSError:
            continue
        state[resolved] = module_node_counts.get(str(module_name), 0)
    return state


def _workflow_nodepack_folders(workflow: dict[str, Any]) -> set[str] | None:
    """Return custom nodepack folders used by workflow class types when running inside ComfyUI."""
    nodes_module = sys.modules.get("nodes")
    if nodes_module is None:
        return None
    node_classes = getattr(nodes_module, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(node_classes, dict):
        return None

    folders: set[str] = set()
    for node in workflow.values():
        class_type = node.get("class_type")
        if not class_type:
            continue
        node_class = node_classes.get(class_type)
        if node_class is None:
            continue
        module_names = [
            str(getattr(node_class, "RELATIVE_PYTHON_MODULE", "") or ""),
            str(getattr(node_class, "__module__", "") or ""),
        ]
        for module_name in module_names:
            if not module_name.startswith("custom_nodes."):
                continue
            parts = module_name.split(".")
            if len(parts) >= 2 and parts[1]:
                folders.add(parts[1])
                break
    return folders


def custom_nodes_roots() -> list[Path]:
    package_root = Path(__file__).resolve().parents[1]
    candidates = [Path.cwd() / "custom_nodes"]
    if package_root.parent.name == "custom_nodes":
        candidates.append(package_root.parent)
    candidates.append(package_root.parent / "custom_nodes")
    candidates.append(package_root.parents[1] / "custom_nodes")
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def scan_installed_nodepacks(root: Path | None = None) -> list[InstalledNodepack]:
    roots = [root] if root is not None else custom_nodes_roots()
    package_root = Path(__file__).resolve().parents[1]
    nodepacks: list[InstalledNodepack] = []
    seen: set[Path] = set()
    loaded_state = _loaded_custom_node_state()
    for root_path in roots:
        if not root_path.exists():
            continue
        for path in sorted(root_path.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            if path.name in {"__pycache__", "civitai_p2p_worker", "civitai-comfy-nodes"}:
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen or resolved == package_root.resolve():
                continue
            seen.add(resolved)

            pyproject = _read_pyproject_metadata(path)
            package_json = _read_package_json_metadata(path)
            git_remote = _run_git(path, ["remote", "get-url", "origin"])

            version_source = None
            version = _clean_nodepack_version(pyproject.get("version"))
            if version:
                version_source = "pyproject"
            if not version:
                version = _clean_nodepack_version(package_json.get("version"))
                if version:
                    version_source = "packageJson"
            if not version:
                version = _git_tag_version(path)
                if version:
                    version_source = "gitTag"

            if version_source == "pyproject":
                repository_url = pyproject.get("repository") or git_remote or package_json.get("repository")
            elif version_source == "packageJson":
                repository_url = package_json.get("repository") or git_remote or pyproject.get("repository")
            else:
                repository_url = git_remote or pyproject.get("repository") or package_json.get("repository")
            registry_id = _github_registry_id_from_url(repository_url)
            air = _infer_nodepack_air(registry_id, version)
            loaded_node_count = loaded_state.get(resolved, 0) if loaded_state is not None else 0
            loaded = loaded_state is None or (resolved in loaded_state and loaded_node_count > 0)
            nodepacks.append(
                InstalledNodepack(
                    folder=path.name,
                    registry_id=registry_id,
                    version=version,
                    air=air,
                    package_name=pyproject.get("name") or package_json.get("name"),
                    git_remote=repository_url,
                    git_commit=_run_git(path, ["rev-parse", "HEAD"]),
                    version_source=version_source,
                    loaded=None if loaded_state is None else loaded,
                    loaded_node_count=loaded_node_count,
                )
            )
    return nodepacks


def _node_inputs(node: dict[str, Any]) -> dict[str, Any]:
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _input_links(node: dict[str, Any]) -> list[tuple[str, int]]:
    links: list[tuple[str, int]] = []
    for value in _node_inputs(node).values():
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
            try:
                links.append((str(value[0]), int(value[1])))
            except (TypeError, ValueError):
                continue
    return links


def _ancestors(prompt: dict[str, Any], node_id: str) -> set[str]:
    result: set[str] = set()

    def visit(current: str) -> None:
        if current in result or current not in prompt:
            return
        result.add(current)
        for source_id, _slot in _input_links(prompt[current]):
            visit(source_id)

    visit(str(node_id))
    return result


def _downstream(prompt: dict[str, Any]) -> dict[str, set[str]]:
    downstream: dict[str, set[str]] = {str(node_id): set() for node_id in prompt}
    for node_id, node in prompt.items():
        for source_id, _slot in _input_links(node):
            downstream.setdefault(source_id, set()).add(str(node_id))
    return downstream


def _descendants(prompt: dict[str, Any], node_id: str) -> set[str]:
    edges = _downstream(prompt)
    result: set[str] = set()

    def visit(current: str) -> None:
        if current in result:
            return
        result.add(current)
        for target in edges.get(current, set()):
            visit(target)

    visit(str(node_id))
    return result


def _region_node_ids(prompt: dict[str, Any]) -> set[str]:
    starts: dict[str, list[str]] = {}
    ends: dict[str, list[str]] = {}
    for node_id, node in prompt.items():
        class_type = node.get("class_type")
        region_id = str(_node_inputs(node).get("region_id") or "default")
        if class_type == OFFLOAD_START_CLASS:
            starts.setdefault(region_id, []).append(str(node_id))
        elif class_type == OFFLOAD_END_CLASS:
            ends.setdefault(region_id, []).append(str(node_id))

    selected: set[str] = set()
    for region_id, start_ids in starts.items():
        for start_id in start_ids:
            for end_id in ends.get(region_id, []):
                selected |= _descendants(prompt, start_id) & _ancestors(prompt, end_id)
    return selected


def _serialized_workflow_nodes(workflow: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(workflow, dict):
        return []
    nodes = workflow.get("nodes")
    if isinstance(nodes, list):
        return [node for node in nodes if isinstance(node, dict)]
    nested = workflow.get("workflow")
    if isinstance(nested, dict):
        return _serialized_workflow_nodes(nested)
    return []


def _serialized_node_id(node: dict[str, Any]) -> str | None:
    value = node.get("id")
    if value is None:
        return None
    return str(value)


def _serialized_node_class(node: dict[str, Any]) -> str:
    return str(node.get("type") or node.get("class_type") or node.get("comfyClass") or "")


def _serialized_node_pos(node: dict[str, Any]) -> tuple[float, float] | None:
    pos = node.get("pos") or node.get("position")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, ValueError):
            return None
    if isinstance(pos, dict):
        try:
            return float(pos.get("x")), float(pos.get("y"))
        except (TypeError, ValueError):
            return None
    return None


def _serialized_region_id(node: dict[str, Any]) -> str:
    widgets = node.get("widgets_values")
    if isinstance(widgets, list) and widgets:
        value = str(widgets[0] or "").strip()
        if value:
            return value
    widgets_by_name = node.get("widgets")
    if isinstance(widgets_by_name, list):
        for widget in widgets_by_name:
            if not isinstance(widget, dict) or widget.get("name") != "region_id":
                continue
            value = str(widget.get("value") or "").strip()
            if value:
                return value
    properties = node.get("properties")
    if isinstance(properties, dict):
        value = str(properties.get("region_id") or "").strip()
        if value:
            return value
    return "default"


CIVITAI_GROUP_TITLE_RE = re.compile(r"^\s*(run on )?civitai\b", re.IGNORECASE)


def is_civitai_group_title(title: Any) -> bool:
    return bool(CIVITAI_GROUP_TITLE_RE.match(str(title or "")))


def _serialized_workflow_groups(workflow: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(workflow, dict):
        return []
    groups = workflow.get("groups")
    if isinstance(groups, list):
        return [group for group in groups if isinstance(group, dict)]
    nested = workflow.get("workflow")
    return _serialized_workflow_groups(nested) if isinstance(nested, dict) else []


def _serialized_node_size(node: dict[str, Any]) -> tuple[float, float]:
    size = node.get("size")
    try:
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return float(size[0]), float(size[1])
        if isinstance(size, dict):
            return float(size.get("0", size.get(0, 0)) or 0), float(size.get("1", size.get(1, 0)) or 0)
    except (TypeError, ValueError):
        pass
    return 0.0, 0.0


def _group_region_node_ids(prompt: dict[str, Any], workflow: dict[str, Any] | None) -> set[str]:
    """Select API prompt nodes whose centre lies inside a canvas group titled "Civitai…"."""
    bounds = []
    for group in _serialized_workflow_groups(workflow):
        if not is_civitai_group_title(group.get("title")):
            continue
        bounding = group.get("bounding")
        if not (isinstance(bounding, (list, tuple)) and len(bounding) >= 4):
            continue
        try:
            x, y, w, h = (float(v) for v in bounding[:4])
        except (TypeError, ValueError):
            continue
        bounds.append((x, y, x + w, y + h))
    if not bounds:
        return set()

    selected: set[str] = set()
    for node in _serialized_workflow_nodes(workflow):
        node_id = _serialized_node_id(node)
        if node_id is None or node_id not in prompt or _serialized_node_class(node) in OFFLOAD_MARKER_CLASSES:
            continue
        pos = _serialized_node_pos(node)
        if pos is None:
            continue
        width, height = _serialized_node_size(node)
        cx, cy = pos[0] + width / 2, pos[1] + height / 2
        if any(left <= cx <= right and top <= cy <= bottom for left, top, right, bottom in bounds):
            selected.add(node_id)
    return selected


def _visual_region_node_ids(prompt: dict[str, Any], workflow: dict[str, Any] | None) -> set[str]:
    """Select API prompt nodes visually placed between matching Start/End nodes.

    This supports the UX contract users expect on the canvas: put a Start marker to the left of the
    offloadable subgraph, put the matching End marker to the right, and everything between those
    markers becomes the submitted customComfy region. The markers do not need to be wired into the
    Comfy execution graph, so local marker placement stays separate from model/data edges.
    """
    graph_nodes = _serialized_workflow_nodes(workflow)
    if not graph_nodes:
        return set()

    starts: dict[str, list[dict[str, Any]]] = {}
    ends: dict[str, list[dict[str, Any]]] = {}
    nodes_by_id: dict[str, dict[str, Any]] = {}
    for node in graph_nodes:
        node_id = _serialized_node_id(node)
        if node_id:
            nodes_by_id[node_id] = node
        class_type = _serialized_node_class(node)
        if class_type == OFFLOAD_START_CLASS:
            starts.setdefault(_serialized_region_id(node), []).append(node)
        elif class_type == OFFLOAD_END_CLASS:
            ends.setdefault(_serialized_region_id(node), []).append(node)

    selected: set[str] = set()
    for region_id, start_nodes in starts.items():
        for start in start_nodes:
            start_pos = _serialized_node_pos(start)
            if start_pos is None:
                continue
            for end in ends.get(region_id, []):
                end_pos = _serialized_node_pos(end)
                if end_pos is None:
                    continue
                left, right = sorted((start_pos[0], end_pos[0]))
                if left == right:
                    continue
                for node_id, node in nodes_by_id.items():
                    if node_id not in prompt:
                        continue
                    class_type = _serialized_node_class(node)
                    if class_type in OFFLOAD_MARKER_CLASSES:
                        continue
                    pos = _serialized_node_pos(node)
                    if pos is not None and left < pos[0] < right:
                        selected.add(node_id)
    return selected


def _normalize_prompt(prompt: dict[str, Any]) -> dict[str, Any]:
    return {str(node_id): copy.deepcopy(node) for node_id, node in (prompt or {}).items() if isinstance(node, dict)}


def _node_sort_key(node_id: str) -> tuple[int, int | str]:
    return (0, int(node_id)) if str(node_id).isdigit() else (1, str(node_id))


def _dependency_closure(prompt: dict[str, Any], node_ids: set[str]) -> set[str]:
    included: set[str] = set()
    for node_id in node_ids:
        included |= _ancestors(prompt, node_id)
    return included


def _is_output_node(class_type: str) -> bool:
    try:
        nodes_module = sys.modules.get("nodes")
        if nodes_module is None:
            import nodes as nodes_module  # type: ignore[no-redef]
        node_classes = getattr(nodes_module, "NODE_CLASS_MAPPINGS", None)
        cls = node_classes.get(class_type) if isinstance(node_classes, dict) else None
        if cls is not None:
            return bool(getattr(cls, "OUTPUT_NODE", False))
    except Exception:
        pass
    return class_type in OUTPUT_NODE_CLASSES


def _user_output_nodes_within_region(prompt: dict[str, Any], included: set[str]) -> set[str]:
    output_nodes: set[str] = set()
    for node_id, node in prompt.items():
        node_id = str(node_id)
        if node_id in included or not _is_output_node(str(node.get("class_type") or "")):
            continue
        dependencies = _ancestors(prompt, node_id) - {node_id}
        if any(prompt.get(dependency, {}).get("class_type") == OFFLOAD_END_CLASS for dependency in dependencies):
            continue
        if dependencies and dependencies <= included:
            output_nodes.add(node_id)
    return output_nodes


def _replace_link_references(prompt: dict[str, Any], old_node_id: str, replacement: list[Any] | None) -> None:
    for node in prompt.values():
        for input_name, value in list(_node_inputs(node).items()):
            if not (isinstance(value, list) and len(value) == 2 and str(value[0]) == old_node_id):
                continue
            if replacement is None:
                del node["inputs"][input_name]
            else:
                node["inputs"][input_name] = copy.deepcopy(replacement)


def strip_offload_markers(prompt: dict[str, Any]) -> dict[str, Any]:
    """Remove passthrough marker nodes and rewire their output slot 0 to their `value` input."""
    workflow = copy.deepcopy(prompt)
    changed = True
    while changed:
        changed = False
        for node_id, node in list(workflow.items()):
            if node.get("class_type") not in OFFLOAD_MARKER_CLASSES:
                continue
            value = _node_inputs(node).get("value")
            replacement = value if isinstance(value, list) and len(value) == 2 else None
            _replace_link_references(workflow, str(node_id), replacement)
            del workflow[node_id]
            changed = True
    return workflow


def _dangling_links(prompt: dict[str, Any]) -> list[tuple[str, str]]:
    dangling: list[tuple[str, str]] = []
    ids = set(prompt)
    for node_id, node in prompt.items():
        for source_id, _slot in _input_links(node):
            if source_id not in ids:
                dangling.append((str(node_id), source_id))
    return dangling


def _unique_node_id(prompt: dict[str, Any], preferred: str) -> str:
    if preferred not in prompt:
        return preferred
    numeric_ids = [int(node_id) for node_id in prompt if str(node_id).isdigit()]
    return str((max(numeric_ids) if numeric_ids else 0) + 1)


def build_local_continuation_prompt(
    prompt: dict[str, Any],
    *,
    remote_node_ids: list[str],
    imported_image_name: str,
    bridge_node_id: str = "civitai_remote_asset",
) -> LocalContinuationBuildResult | None:
    """Build a local Comfy prompt for nodes downstream of the offloaded region.

    Links that crossed from the offloaded subgraph into the local tail are rewritten to a LoadImage
    bridge loaded from the remote customComfy asset. This is intentionally image-first because the
    current customComfy asset contract only exposes file URLs, not typed socket values.
    """
    normalized = _normalize_prompt(prompt)
    remote_ids = {str(node_id) for node_id in remote_node_ids if str(node_id) in normalized}
    if not normalized or not remote_ids:
        return None

    downstream = _downstream(normalized)
    tail_seed: set[str] = set()
    remote_source_ids: set[str] = set()
    for remote_id in remote_ids:
        for target_id in downstream.get(remote_id, set()):
            if target_id in remote_ids:
                continue
            tail_seed.add(target_id)
            remote_source_ids.add(remote_id)
    if not tail_seed:
        return None

    tail_descendants: set[str] = set()
    for node_id in tail_seed:
        tail_descendants |= _descendants(normalized, node_id)
    tail_descendants -= remote_ids
    if not tail_descendants:
        return None

    output_node_ids = {
        node_id
        for node_id in tail_descendants
        if _is_output_node(str(normalized.get(node_id, {}).get("class_type") or ""))
    }
    target_ids = output_node_ids or tail_descendants
    tail_ids = _dependency_closure(normalized, target_ids) - remote_ids
    if not tail_ids:
        return None

    local_prompt = {
        node_id: copy.deepcopy(normalized[node_id])
        for node_id in sorted(tail_ids, key=_node_sort_key)
    }
    bridge_id = _unique_node_id({**normalized, **local_prompt}, bridge_node_id)
    local_prompt = {
        bridge_id: {"class_type": "LoadImage", "inputs": {"image": imported_image_name}},
        **local_prompt,
    }

    for node in local_prompt.values():
        for input_name, value in list(_node_inputs(node).items()):
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in remote_ids:
                remote_source_ids.add(str(value[0]))
                node["inputs"][input_name] = [bridge_id, 0]

    dangling = _dangling_links(local_prompt)
    if dangling:
        refs = ", ".join(f"{node_id}->{source_id}" for node_id, source_id in dangling[:8])
        raise CivitaiNodeError(f"Local continuation has inputs from unavailable nodes: {refs}")

    return LocalContinuationBuildResult(
        prompt=local_prompt,
        bridge_node_id=bridge_id,
        tail_node_ids=sorted(tail_ids, key=_node_sort_key),
        output_node_ids=sorted(output_node_ids, key=_node_sort_key),
        remote_source_node_ids=sorted(remote_source_ids, key=_node_sort_key),
    )


def _value_contains_air(value: Any, resources: set[str]) -> None:
    if isinstance(value, str):
        if AIR_RE.match(value.strip()):
            resources.add(value.strip())
    elif isinstance(value, list):
        for item in value:
            _value_contains_air(item, resources)
    elif isinstance(value, dict):
        for item in value.values():
            _value_contains_air(item, resources)


def _model_record_index(records: list[LocalModelRecord]) -> dict[tuple[str | None, str], LocalModelRecord]:
    index: dict[tuple[str | None, str], LocalModelRecord] = {}
    for record in records:
        names = {record.name, Path(record.name).name, Path(record.path).name}
        for name in names:
            key_name = name.replace("\\", "/").lower()
            index[(record.folder, key_name)] = record
            index[(None, key_name)] = record
    return index


def _find_model_record(
    value: str,
    input_name: str,
    records_by_name: dict[tuple[str | None, str], LocalModelRecord],
) -> LocalModelRecord | None:
    name = value.replace("\\", "/").lower()
    folders = MODEL_WIDGET_FOLDERS.get(input_name, ())
    for folder in folders:
        record = records_by_name.get((folder, name)) or records_by_name.get((folder, Path(name).name))
        if record:
            return record
    if Path(name).suffix.lower() in MODEL_EXTENSIONS or input_name in MODEL_WIDGET_FOLDERS:
        return records_by_name.get((None, name)) or records_by_name.get((None, Path(name).name))
    return None


def _resolve_record_air(
    record: LocalModelRecord,
    *,
    token: str | None,
    session: requests.Session | None,
    civitai_base_url: str | None,
) -> LocalModelRecord | None:
    if record.air:
        return record
    resolved = resolve_model_air(record.path, token=token, session=session, civitai_base_url=civitai_base_url)
    if not resolved:
        return None
    resolved.folder = record.folder
    resolved.name = record.name
    return resolved


def replace_local_models_with_airs(
    workflow: dict[str, Any],
    *,
    model_records: list[LocalModelRecord],
    resources: set[str],
    token: str | None = None,
    session: requests.Session | None = None,
    civitai_base_url: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    rewritten = copy.deepcopy(workflow)
    index = _model_record_index(model_records)
    resolved_models: list[dict[str, Any]] = []
    warnings: list[str] = []
    cache: dict[str, LocalModelRecord | None] = {}

    for node in rewritten.values():
        for input_name, value in list(_node_inputs(node).items()):
            _value_contains_air(value, resources)
            if not isinstance(value, str):
                continue
            if AIR_RE.match(value.strip()):
                continue
            record = _find_model_record(value, input_name, index)
            if not record:
                if input_name in MODEL_WIDGET_FOLDERS:
                    warnings.append(
                        f"Local model '{value}' on input '{input_name}' was not found in ComfyUI model dirs"
                    )
                continue
            cache_key = record.path
            resolved = cache.get(cache_key)
            if cache_key not in cache:
                resolved = _resolve_record_air(
                    record, token=token, session=session, civitai_base_url=civitai_base_url
                )
                cache[cache_key] = resolved
            if not resolved or not resolved.air:
                warnings.append(f"Local model '{value}' could not be resolved to a Civitai AIR by hash")
                continue
            node["inputs"][input_name] = resolved.air
            resources.add(resolved.air)
            resolved_models.append(resolved.as_dict())
    return rewritten, resolved_models, warnings


def _is_remote_or_air_media_value(value: str) -> bool:
    cleaned = value.strip()
    return bool(
        AIR_RE.match(cleaned)
        or cleaned.startswith("http://")
        or cleaned.startswith("https://")
        or cleaned.startswith("data:")
    )


def _resolve_comfy_input_path(name: str) -> Path:
    try:
        import folder_paths  # type: ignore[import-not-found]
    except Exception as e:
        raise CivitaiNodeError(
            f"Cannot resolve local media input '{name}' outside a running ComfyUI environment"
        ) from e

    try:
        exists = folder_paths.exists_annotated_filepath(name)
        path = folder_paths.get_annotated_filepath(name)
    except Exception as e:
        raise CivitaiNodeError(f"Could not resolve local media input '{name}': {e}") from e

    if not exists:
        raise CivitaiNodeError(f"Local media input '{name}' does not exist in ComfyUI input storage")
    return Path(path)


def _media_content_type(path: str | os.PathLike[str], allowed: set[str]) -> str:
    path = str(path)
    try:
        with open(path, "rb") as handle:
            header = handle.read(16)
    except OSError as e:
        raise CivitaiNodeError(f"Could not read local media input '{path}': {e}") from e

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = "image/png"
    elif header.startswith(b"\xff\xd8\xff"):
        detected = "image/jpeg"
    elif header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        detected = "image/webp"
    elif header[4:8] == b"ftyp":
        detected = "video/mp4"
    elif header.startswith(b"\x1a\x45\xdf\xa3"):
        detected = "audio/webm" if "audio/webm" in allowed else "video/webm"
    elif header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        detected = "audio/mpeg"
    else:
        detected = mimetypes.guess_type(path)[0]

    if detected in allowed:
        return detected
    if detected == "video/webm" and "audio/webm" in allowed and "video/webm" not in allowed:
        return "audio/webm"
    if detected == "audio/webm" and "video/webm" in allowed and "audio/webm" not in allowed:
        return "video/webm"

    guessed = mimetypes.guess_type(path)[0]
    if guessed in allowed:
        return guessed
    if guessed == "video/webm" and "audio/webm" in allowed:
        return "audio/webm"
    if guessed == "audio/webm" and "video/webm" in allowed:
        return "video/webm"

    supported = ", ".join(sorted(allowed))
    raise CivitaiNodeError(
        f"Local media input '{path}' has unsupported content type '{detected or guessed or 'unknown'}'. "
        f"Civitai blob upload supports {supported} for this node input."
    )


def _blob_air_from_upload(blob: dict[str, Any]) -> str:
    blob_id = blob.get("id")
    if not blob_id and blob.get("url"):
        parsed = parse.urlparse(str(blob["url"]))
        blob_id = parse.unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    if not blob_id:
        raise CivitaiNodeError("Blob upload response did not include a blob id or usable URL")
    return f"urn:air:other:other:orchestrator:blob@{blob_id}"


def replace_local_media_inputs_with_blob_airs(
    workflow: dict[str, Any],
    *,
    resources: set[str],
    upload_blob_file: Callable[[Path, str], dict[str, Any]] | None,
    path_resolver: Callable[[str], str | os.PathLike[str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Upload local media loader inputs and replace widget filenames with blob AIRs."""
    rewritten = copy.deepcopy(workflow)
    if upload_blob_file is None:
        return rewritten, []

    resolver = path_resolver or _resolve_comfy_input_path
    uploaded: list[UploadedInputBlob] = []
    cache: dict[Path, dict[str, Any]] = {}

    for node_id, node in rewritten.items():
        class_type = node.get("class_type")
        upload_inputs = UPLOAD_MEDIA_INPUTS.get(class_type)
        if not upload_inputs:
            continue
        inputs = _node_inputs(node)
        for input_name, allowed_content_types in upload_inputs.items():
            value = inputs.get(input_name)
            if not isinstance(value, str) or not value.strip() or _is_remote_or_air_media_value(value):
                continue

            path = Path(resolver(value)).expanduser().resolve()
            if not path.exists():
                raise CivitaiNodeError(f"Local media input '{value}' resolved to missing file '{path}'")
            if not path.is_file():
                raise CivitaiNodeError(f"Local media input '{value}' resolved to non-file path '{path}'")

            content_type = _media_content_type(path, allowed_content_types)
            blob = cache.get(path)
            if blob is None:
                blob = upload_blob_file(path, content_type)
                cache[path] = blob
            air = _blob_air_from_upload(blob)
            inputs[input_name] = air
            resources.add(air)
            uploaded.append(
                UploadedInputBlob(
                    node_id=str(node_id),
                    input_name=input_name,
                    original_name=value,
                    path=str(path),
                    content_type=content_type,
                    air=air,
                    blob_id=blob.get("id"),
                    url=blob.get("url"),
                    size=path.stat().st_size,
                )
            )

    return rewritten, [item.as_dict() for item in uploaded]


def _selector_resources_by_slot(node: dict[str, Any]) -> dict[int, str]:
    """Parse a CivitaiModelSelector's file-pinned resource AIRs from its `resources_json` widget
    ({"bySlot": {"1": air, ...}, "all": [...]}) into {slot: air}. Empty when absent/corrupt."""
    raw = _node_inputs(node).get("resources_json")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    by_slot = parsed.get("bySlot") if isinstance(parsed, dict) else None
    result: dict[int, str] = {}
    if isinstance(by_slot, dict):
        for key, value in by_slot.items():
            try:
                slot = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value.strip():
                result[slot] = value.strip()
    return result


def civitai_api_node_ids(prompt: dict[str, Any]) -> list[str]:
    """Nodes of this pack that call the orchestrator themselves (recipe nodes, CivitaiAuth). The
    local selector/loader helpers are not included: they only produce AIRs or download files."""
    from . import generated, nodes_manual
    from .base import CivitaiRecipeNodeBase

    classes = {**generated.NODE_CLASS_MAPPINGS, **nodes_manual.NODE_CLASS_MAPPINGS}
    api_classes = {
        key for key, cls in classes.items() if key == "CivitaiAuth" or issubclass(cls, CivitaiRecipeNodeBase)
    }
    return sorted(
        (node_id for node_id, node in prompt.items() if node.get("class_type") in api_classes), key=_node_sort_key
    )


def bake_model_selectors(prompt: dict[str, Any], resources: set[str]) -> list[str]:
    """Replace every downstream link consuming a CivitaiModelSelector output slot with that slot's
    file-pinned AIR, add the AIR to `resources`, and remove the selector so the worker never
    re-downloads it — mirroring comfy-cloud's PromptInterceptor.BakeModelSelectors. A selector whose
    only wired output is `air` (slot 0, no file-pinned AIR) is left in place and returned, so the
    caller can warn that it will run on the worker."""
    selectors = {
        node_id: _selector_resources_by_slot(node)
        for node_id, node in prompt.items()
        if node.get("class_type") == MODEL_SELECTOR_CLASS
    }
    if not selectors:
        return []
    left_in_place: set[str] = set()
    for node_id, node in prompt.items():
        if node_id in selectors:
            continue
        for input_name, value in list(_node_inputs(node).items()):
            if not (isinstance(value, list) and len(value) == 2):
                continue
            source_id = str(value[0])
            by_slot = selectors.get(source_id)
            if by_slot is None:
                continue
            try:
                slot = int(value[1])
            except (TypeError, ValueError):
                continue
            air = by_slot.get(slot)
            if air:
                node["inputs"][input_name] = air
                resources.add(air)
            else:
                left_in_place.add(source_id)
    for node_id in list(selectors):
        if node_id not in left_in_place:
            del prompt[node_id]
    return sorted(left_in_place, key=_node_sort_key)


_HF_HOSTS = {"huggingface.co", "hf.co"}
# HF file urls route through a "resolve" (download) or "blob" (viewer) segment.
_HF_ROUTER_SEGMENTS = {"resolve", "blob"}
_HF_DIRECTORY_TYPES = {
    "checkpoints": "checkpoint",
    "diffusion_models": "diffusion_model",
    "loras": "lora",
    "upscale_models": "upscaler",
    "hypernetworks": "hypernetwork",
    "embeddings": "embedding",
    "repository": "other",  # never "repository": that routes to the whole-repo tar, not a file
}
_HF_AIR_SAFE_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def _hf_type_for_directory(directory: str | None) -> str:
    d = (directory or "").strip().lower()
    if not d:
        return "other"
    return _HF_DIRECTORY_TYPES.get(d, d)


def _hf_air_from_url(url: str | None, directory: str | None) -> str | None:
    """Build a downloadable HuggingFace *file* AIR from a template loader's `properties.models` url
    (`urn:air:other:{type}:huggingface:{repo}@{rev}/{path}`), or None for a null/non-HF/malformed
    url. Mirrors comfy-cloud HuggingFaceModelAir.TryBuild — the orchestrator's huggingface resource
    provider resolves the AIR to the presigned HF download, so a loaded template runs without a
    manual Civitai pick."""
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return None
    if (parsed.hostname or "").lower() not in _HF_HOSTS:
        return None
    segments = [s for s in parsed.path.split("/") if s]
    router = next((i for i, s in enumerate(segments) if s in _HF_ROUTER_SEGMENTS), -1)
    # Need >=1 repo segment before the router, a revision after it, and >=1 path segment.
    if router < 1 or router + 2 >= len(segments):
        return None
    repo = parse.unquote("/".join(segments[:router]))
    revision = parse.unquote(segments[router + 1])
    path = parse.unquote("/".join(segments[router + 2 :]))
    if not all(_HF_AIR_SAFE_RE.match(part) for part in (repo, revision, path)):
        return None
    return f"urn:air:other:{_hf_type_for_directory(directory)}:huggingface:{repo}@{revision}/{path}"


def _hf_value_matches_name(value: str, name: str) -> bool:
    """The widget value and the metadata name match when equal, or when either carries the folder
    prefix the other omits (mirrors comfy-cloud ValueMatchesName / the JS findReferencingWidgets)."""
    return (
        value == name
        or value.endswith("/" + name)
        or value.endswith("\\" + name)
        or name.endswith("/" + value)
        or name.endswith("\\" + value)
    )


def _hf_models_from_nodes(
    nodes: Any,
    correlate_by_node_id: bool,
    by_node: dict[str, list[tuple[str, str]]],
    airs_by_name: dict[str, set[str]],
) -> None:
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        properties = node.get("properties")
        models = properties.get("models") if isinstance(properties, dict) else None
        if not isinstance(models, list) or not models:
            continue
        node_id = _serialized_node_id(node) if correlate_by_node_id else None
        for model in models:
            if not isinstance(model, dict):
                continue
            name = model.get("name")
            if not isinstance(name, str) or not name:
                continue
            air = _hf_air_from_url(model.get("url"), model.get("directory"))
            if not air:
                continue
            if node_id is not None:
                by_node.setdefault(node_id, []).append((name, air))
            airs_by_name.setdefault(name, set()).add(air)


def collect_huggingface_model_airs(
    workflow: dict[str, Any] | None,
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, str]]:
    """Build (by_node, global) HuggingFace AIR maps from a serialized workflow's loader metadata
    (`nodes[].properties.models` = `{name, url, directory}` triples), covering both top-level nodes
    and each subgraph definition. Mirrors comfy-cloud CollectHuggingFaceModelAirs. The API-format
    prompt flattens subgraphs into composite ids, so subgraph models resolve via the global map
    (name unique across the graph) rather than node-id correlation."""
    by_node: dict[str, list[tuple[str, str]]] = {}
    airs_by_name: dict[str, set[str]] = {}
    root = workflow if isinstance(workflow, dict) else {}
    if isinstance(root.get("workflow"), dict):
        root = root["workflow"]
    _hf_models_from_nodes(root.get("nodes"), True, by_node, airs_by_name)
    definitions = root.get("definitions")
    subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
    for subgraph in subgraphs or []:
        if isinstance(subgraph, dict):
            _hf_models_from_nodes(subgraph.get("nodes"), False, by_node, airs_by_name)
    global_map = {name: next(iter(airs)) for name, airs in airs_by_name.items() if len(airs) == 1}
    return by_node, global_map


def apply_huggingface_model_airs(
    prompt: dict[str, Any],
    by_node: dict[str, list[tuple[str, str]]],
    global_map: dict[str, str],
    resources: set[str],
) -> int:
    """Rewrite every bare-filename loader input matching a template HuggingFace model to its AIR and
    declare it as a resource, so the worker downloads it from HuggingFace. Skips values a Civitai pin
    already owns (the ` — ` DisplayValue marker) or that are already AIRs. Returns the rewrite count.
    Mirrors comfy-cloud ApplyHuggingFaceModelAirs."""
    if not by_node and not global_map:
        return 0
    rewritten = 0
    for node_id, node in prompt.items():
        for input_name, value in list(_node_inputs(node).items()):
            if not isinstance(value, str) or not value:
                continue
            if " — " in value or value.startswith("urn:air:"):
                continue
            air = next(
                (a for name, a in by_node.get(str(node_id), []) if _hf_value_matches_name(value, name)),
                None,
            )
            if air is None:
                air = next((a for name, a in global_map.items() if _hf_value_matches_name(value, name)), None)
            if air is None:
                continue
            node["inputs"][input_name] = air
            resources.add(air)
            rewritten += 1
    return rewritten


def build_custom_comfy_offload(
    prompt: dict[str, Any],
    *,
    selected_node_ids: list[str] | None = None,
    workflow: dict[str, Any] | None = None,
    model_records: list[LocalModelRecord] | None = None,
    nodepacks: list[InstalledNodepack] | None = None,
    token: str | None = None,
    session: requests.Session | None = None,
    civitai_base_url: str | None = None,
    trace: str | None = None,
    min_vram_gb: int | None = None,
    use_sage_attention: bool | None = None,
    upload_blob_file: Callable[[Path, str], dict[str, Any]] | None = None,
    input_path_resolver: Callable[[str], str | os.PathLike[str]] | None = None,
) -> OffloadBuildResult:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        raise CivitaiNodeError("No ComfyUI prompt graph was provided")

    explicit_selection = {str(node_id) for node_id in selected_node_ids or [] if str(node_id) in normalized}
    region_selection = _region_node_ids(normalized)
    group_selection = set() if explicit_selection else _group_region_node_ids(normalized, workflow)
    visual_region_selection = (
        set() if explicit_selection or group_selection else _visual_region_node_ids(normalized, workflow)
    )
    selected = explicit_selection or group_selection or visual_region_selection or region_selection or set(normalized)
    included = _dependency_closure(normalized, selected)
    if region_selection and not explicit_selection:
        included |= _user_output_nodes_within_region(normalized, included)
    subset = {
        node_id: copy.deepcopy(normalized[node_id])
        for node_id in sorted(included, key=_node_sort_key)
    }
    stripped = strip_offload_markers(subset)

    # Bake before the dangling-link check: consuming links become AIR strings and the selector nodes
    # are removed, so they neither dangle nor re-download on the worker.
    resources: set[str] = set()
    selectors_left = bake_model_selectors(stripped, resources)

    api_nodes = civitai_api_node_ids(stripped)
    if api_nodes:
        listed = ", ".join(f"{node_id} ({stripped[node_id].get('class_type')})" for node_id in api_nodes[:8])
        raise CivitaiNodeError(
            "Civitai API nodes already run on Civitai and would be billed twice on a worker: "
            f"{listed}. Move them outside the Civitai group (they run locally) or use a native ComfyUI "
            "node instead."
        )

    # Resolve template HuggingFace models (built-in ComfyUI templates reference HF files by bare
    # filename + node metadata) to AIRs before the local-hash pass, which then skips the AIR values.
    hf_by_node, hf_global = collect_huggingface_model_airs(workflow)
    hf_rewritten = apply_huggingface_model_airs(stripped, hf_by_node, hf_global, resources)
    if hf_rewritten:
        _log.info("offload build: resolved %d HuggingFace template model AIRs", hf_rewritten)

    dangling = _dangling_links(stripped)
    if dangling:
        refs = ", ".join(f"{node_id}->{source_id}" for node_id, source_id in dangling[:8])
        raise CivitaiNodeError(f"Offload selection has inputs from nodes outside the submitted graph: {refs}")

    for node in stripped.values():
        _value_contains_air(_node_inputs(node), resources)

    _t = time.monotonic()
    model_records = model_records if model_records is not None else scan_local_model_files()
    _log.info("offload build: scanned %d local models in %.2fs", len(model_records), time.monotonic() - _t)
    _t = time.monotonic()
    rewritten, resolved_models, model_warnings = replace_local_models_with_airs(
        stripped,
        model_records=model_records,
        resources=resources,
        token=token,
        session=session,
        civitai_base_url=civitai_base_url,
    )
    _log.info(
        "offload build: resolved %d model AIRs in %.2fs (hash sources: %s)",
        len(resolved_models),
        time.monotonic() - _t,
        ", ".join(sorted({(m.get("hash_source") or "?") for m in resolved_models})) or "-",
    )
    _t = time.monotonic()
    rewritten, input_blobs = replace_local_media_inputs_with_blob_airs(
        rewritten,
        resources=resources,
        upload_blob_file=upload_blob_file,
        path_resolver=input_path_resolver,
    )
    _log.info("offload build: uploaded %d media blobs in %.2fs", len(input_blobs), time.monotonic() - _t)

    _t = time.monotonic()
    nodepacks = nodepacks if nodepacks is not None else scan_installed_nodepacks()
    _log.info("offload build: scanned %d nodepacks in %.2fs", len(nodepacks), time.monotonic() - _t)
    used_nodepack_folders = _workflow_nodepack_folders(rewritten)
    nodepack_resources = []
    used_packs: list[InstalledNodepack] = []
    unresolved_pack_folders: list[str] = []
    for nodepack in nodepacks:
        if nodepack.loaded is False:
            continue
        if used_nodepack_folders is not None and nodepack.folder not in used_nodepack_folders:
            continue
        if not nodepack.air:
            unresolved_pack_folders.append(nodepack.folder)
            continue
        used_packs.append(nodepack)
        nodepack_resources.append(nodepack.as_dict())
    # Stable order so the snapshot's results[] align with the $ref indices below.
    used_packs.sort(key=lambda p: p.air or "")

    warnings = list(dict.fromkeys(model_warnings))
    if selected != included:
        warnings.append("Included upstream dependencies required to make the offloaded Comfy graph runnable")
    if (visual_region_selection or region_selection) and not explicit_selection:
        warnings.append("Using Civitai Offload Start/End markers to select the submitted graph")
    if selectors_left:
        warnings.append(
            "A Civitai Model Selector's `air` output is wired directly, so the selector node will run "
            "on the worker instead of being baked away: " + ", ".join(selectors_left)
        )
    if unresolved_pack_folders:
        warnings.append(
            "Custom nodes used by this workflow aren't published as a Civitai registry pack, so they "
            "can't be installed on the worker: " + ", ".join(sorted(set(unresolved_pack_folders)))
        )

    # Declare each used custom nodepack as its complete install-layer AIR (the (pack, comfy image)
    # pair), produced by a comfyNodepackSnapshot step that runs first and dedupes server-side. The
    # comfy image is omitted — a local install has none of its own, so orchestration captures the
    # layer against the workers' current image and pins the run to it. customComfy references each
    # layer via $ref into the snapshot output; results[] order matches the nodepacks[] order.
    custom_resources: list[Any] = sorted(resources)
    steps: list[dict[str, Any]] = []
    if used_packs:
        steps.append(
            {
                "$type": "comfyNodepackSnapshot",
                "name": "snapshot",
                "input": {"nodepacks": [pack.air for pack in used_packs]},
            }
        )
        for index in range(len(used_packs)):
            custom_resources.append({"$ref": "snapshot", "path": f"output.results[{index}].layerAir"})

    custom_input: dict[str, Any] = {"resources": custom_resources, "workflow": rewritten}
    if trace:
        custom_input["trace"] = trace
    if min_vram_gb is not None:
        custom_input["minVramGb"] = min_vram_gb
    if use_sage_attention:
        custom_input["useSageAttention"] = True
    # The worker exposes this as CIVITAI_API_TOKEN so any civitai-comfy-nodes recipe nodes inside the
    # offloaded graph run as — and bill — the submitting user. Never logged.
    steps.append({"$type": "customComfy", "input": custom_input})
    return OffloadBuildResult(
        steps=steps,
        workflow=rewritten,
        resources=sorted(resources),
        warnings=warnings,
        selected_node_ids=sorted(selected, key=_node_sort_key),
        included_node_ids=sorted(included, key=_node_sort_key),
        model_resources=resolved_models,
        nodepack_resources=nodepack_resources,
        input_blobs=input_blobs,
    )
