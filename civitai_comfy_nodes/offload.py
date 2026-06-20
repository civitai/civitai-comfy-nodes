"""Helpers for submitting local ComfyUI workflows as Civitai customComfy jobs.

The code in here is deliberately import-safe outside ComfyUI. Runtime-only modules like
folder_paths are imported lazily so pytest can exercise the inventory and transform logic.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib import parse

import requests

from . import oauth
from .errors import CivitaiNodeError
from .local_models import AIR_TYPE_FOLDERS, USER_AGENT

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only hit on Python 3.10 runtimes
    tomllib = None


CIVITAI_BASE_URL = os.environ.get("CIVITAI_BASE_URL") or oauth.OAUTH_BASE
MODEL_EXTENSIONS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin"}
SAFETENSORS_MAX_HEADER = 16 * 1024 * 1024
OFFLOAD_START_CLASS = "CivitaiOffloadStart"
OFFLOAD_END_CLASS = "CivitaiOffloadEnd"
OFFLOAD_MARKER_CLASSES = {OFFLOAD_START_CLASS, OFFLOAD_END_CLASS}
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
MODEL_FOLDERS = tuple(dict.fromkeys(AIR_TYPE_FOLDERS.values()))
AIR_RE = re.compile(r"^(?:urn:)?air:", re.IGNORECASE)
HEX_RE = re.compile(r"[0-9a-fA-F]{8,128}")


@dataclass
class LocalModelRecord:
    folder: str
    name: str
    path: str
    hashes: dict[str, str] | None = None
    hash_source: str | None = None
    air: str | None = None
    model_version_id: int | None = None
    lookup_hash_type: str | None = None
    lookup_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _clean_hash(value: Any) -> str | None:
    if value is None:
        return None
    match = HEX_RE.search(str(value).strip())
    if not match:
        return None
    return match.group(0).upper()


def _hash_type_for_metadata_key(key: str, value: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    if normalized in {"sha256", "sha256hash", "filesha256", "fullsha256"}:
        return "SHA256"
    if normalized in {"autov1", "autov1hash"}:
        return "AutoV1"
    if normalized in {"autov2", "autov2hash", "sshslegacyhash"}:
        return "AutoV2"
    if normalized in {"autov3", "autov3hash", "sshsmodelhash"}:
        return "AutoV3"
    if "sha256" in normalized and len(value) == 64:
        return "SHA256"
    if "autov3" in normalized and len(value) == 64:
        return "AutoV3"
    if "autov2" in normalized and len(value) == 10:
        return "AutoV2"
    if "autov1" in normalized and len(value) == 8:
        return "AutoV1"
    if "hash" in normalized:
        return "Hash"
    return None


def _read_safetensors_header(path: str | os.PathLike[str]) -> tuple[dict[str, Any] | None, int]:
    try:
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None, 0
            header_length = int.from_bytes(raw_length, "little")
            if header_length <= 0 or header_length > SAFETENSORS_MAX_HEADER:
                return None, 0
            header_bytes = handle.read(header_length)
    except OSError:
        return None, 0
    if len(header_bytes) != header_length:
        return None, 0
    try:
        header = json.loads(header_bytes.rstrip(b"\0").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 0
    return header if isinstance(header, dict) else None, 8 + header_length


def read_model_hashes_from_metadata(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read embedded hashes from safetensors metadata without hashing the model bytes."""
    header, _payload_offset = _read_safetensors_header(path)
    if not header:
        return {}

    items: list[tuple[str, Any]] = []
    metadata = header.get("__metadata__")
    if isinstance(metadata, dict):
        items.extend(metadata.items())
    items.extend((key, value) for key, value in header.items() if key != "__metadata__")

    hashes: dict[str, str] = {}
    for key, value in items:
        cleaned = _clean_hash(value)
        if not cleaned:
            continue
        hash_type = _hash_type_for_metadata_key(str(key), cleaned)
        if hash_type and hash_type not in hashes:
            hashes[hash_type] = cleaned
    return hashes


def _sha256_file(path: str | os.PathLike[str]) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    crc = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return sha256.hexdigest().upper(), f"{crc & 0xFFFFFFFF:08X}"


def _autov1_file(path: str | os.PathLike[str]) -> str | None:
    try:
        size = os.path.getsize(path)
        if size < 0x100000 * 2:
            return None
        with open(path, "rb") as handle:
            handle.seek(0x100000)
            block = handle.read(0x10000)
    except OSError:
        return None
    if len(block) != 0x10000:
        return None
    return hashlib.sha256(block).hexdigest().upper()[:8]


def _autov3_safetensors_payload(path: str | os.PathLike[str]) -> str | None:
    _header, payload_offset = _read_safetensors_header(path)
    if not payload_offset:
        return None
    sha256 = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(payload_offset)
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest().upper()


def compute_model_hashes(path: str | os.PathLike[str]) -> dict[str, str]:
    """Compute the Civitai hash subset needed for local model AIR lookup.

    Mirrors the scanner behavior for SHA256, AutoV1, AutoV2, AutoV3, and CRC32. Blake3 is omitted
    because this package intentionally has no native hashing dependency.
    """
    sha256, crc32 = _sha256_file(path)
    hashes = {"SHA256": sha256, "AutoV2": sha256[:10], "CRC32": crc32}
    autov1 = _autov1_file(path)
    if autov1:
        hashes["AutoV1"] = autov1
    autov3 = _autov3_safetensors_payload(path)
    if autov3:
        hashes["AutoV3"] = autov3
    return hashes


def get_model_hashes(path: str | os.PathLike[str], *, prefer_metadata: bool = True) -> tuple[dict[str, str], str]:
    if prefer_metadata:
        metadata_hashes = read_model_hashes_from_metadata(path)
        if metadata_hashes:
            return metadata_hashes, "metadata"
    return compute_model_hashes(path), "computed"


def _lookup_candidates(hashes: dict[str, str]) -> list[tuple[str, str]]:
    order = ("SHA256", "AutoV3", "AutoV2", "AutoV1", "CRC32", "Hash")
    candidates: list[tuple[str, str]] = []
    for key in order:
        value = hashes.get(key)
        if value:
            candidates.append((key, value))
    for key, value in hashes.items():
        if (key, value) not in candidates:
            candidates.append((key, value))
    return candidates


def lookup_model_version_by_hash(
    hash_value: str,
    *,
    token: str | None = None,
    session: requests.Session | None = None,
    civitai_base_url: str | None = None,
) -> dict[str, Any] | None:
    base = (civitai_base_url or CIVITAI_BASE_URL).rstrip("/")
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    http = session or requests.Session()
    response = http.get(
        f"{base}/api/v1/model-versions/by-hash/{hash_value.upper()}",
        headers=headers,
        timeout=30,
    )
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        raise CivitaiNodeError(f"Civitai hash lookup failed ({response.status_code}): {response.text}")
    data = response.json()
    return data if isinstance(data, dict) and data.get("air") else None


def resolve_model_air(
    path: str | os.PathLike[str],
    *,
    token: str | None = None,
    session: requests.Session | None = None,
    civitai_base_url: str | None = None,
) -> LocalModelRecord | None:
    """Resolve a local file to AIR using metadata hashes first, then computed hashes as fallback."""
    metadata_hashes = read_model_hashes_from_metadata(path)
    for hashes, source in ((metadata_hashes, "metadata"),):
        if not hashes:
            continue
        for hash_type, hash_value in _lookup_candidates(hashes):
            version = lookup_model_version_by_hash(
                hash_value, token=token, session=session, civitai_base_url=civitai_base_url
            )
            if version:
                return LocalModelRecord(
                    folder="",
                    name=Path(path).name,
                    path=str(path),
                    hashes=hashes,
                    hash_source=source,
                    air=version.get("air"),
                    model_version_id=version.get("id"),
                    lookup_hash_type=hash_type,
                    lookup_hash=hash_value,
                )

    computed_hashes = compute_model_hashes(path)
    for hash_type, hash_value in _lookup_candidates(computed_hashes):
        version = lookup_model_version_by_hash(
            hash_value, token=token, session=session, civitai_base_url=civitai_base_url
        )
        if version:
            return LocalModelRecord(
                folder="",
                name=Path(path).name,
                path=str(path),
                hashes=computed_hashes,
                hash_source="computed",
                air=version.get("air"),
                model_version_id=version.get("id"),
                lookup_hash_type=hash_type,
                lookup_hash=hash_value,
            )
    return None


def _folder_paths_for(folder: str) -> list[str]:
    try:
        import folder_paths

        return list(folder_paths.get_folder_paths(folder) or [])
    except Exception:
        return []


def model_roots_by_folder(folders: tuple[str, ...] = MODEL_FOLDERS) -> dict[str, list[Path]]:
    roots: dict[str, list[Path]] = {}
    for folder in folders:
        paths = [Path(path) for path in _folder_paths_for(folder)]
        if paths:
            roots[folder] = paths
    return roots


def scan_local_model_files(roots: dict[str, list[Path]] | None = None) -> list[LocalModelRecord]:
    roots = roots if roots is not None else model_roots_by_folder()
    records: list[LocalModelRecord] = []
    seen: set[Path] = set()
    for folder, paths in roots.items():
        for root in paths:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
                if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    name = str(path.relative_to(root))
                except ValueError:
                    name = path.name
                records.append(LocalModelRecord(folder=folder, name=name, path=str(path)))
    return records


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
    upload_blob_file: Callable[[Path, str], dict[str, Any]] | None = None,
    input_path_resolver: Callable[[str], str | os.PathLike[str]] | None = None,
) -> OffloadBuildResult:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        raise CivitaiNodeError("No ComfyUI prompt graph was provided")

    explicit_selection = {str(node_id) for node_id in selected_node_ids or [] if str(node_id) in normalized}
    region_selection = _region_node_ids(normalized)
    visual_region_selection = set() if explicit_selection else _visual_region_node_ids(normalized, workflow)
    selected = explicit_selection or visual_region_selection or region_selection or set(normalized)
    included = _dependency_closure(normalized, selected)
    if region_selection and not explicit_selection:
        included |= _user_output_nodes_within_region(normalized, included)
    subset = {
        node_id: copy.deepcopy(normalized[node_id])
        for node_id in sorted(included, key=_node_sort_key)
    }
    stripped = strip_offload_markers(subset)

    dangling = _dangling_links(stripped)
    if dangling:
        refs = ", ".join(f"{node_id}->{source_id}" for node_id, source_id in dangling[:8])
        raise CivitaiNodeError(f"Offload selection has inputs from nodes outside the submitted graph: {refs}")

    resources: set[str] = set()
    for node in stripped.values():
        _value_contains_air(_node_inputs(node), resources)

    model_records = model_records if model_records is not None else scan_local_model_files()
    rewritten, resolved_models, model_warnings = replace_local_models_with_airs(
        stripped,
        model_records=model_records,
        resources=resources,
        token=token,
        session=session,
        civitai_base_url=civitai_base_url,
    )
    rewritten, input_blobs = replace_local_media_inputs_with_blob_airs(
        rewritten,
        resources=resources,
        upload_blob_file=upload_blob_file,
        path_resolver=input_path_resolver,
    )

    nodepacks = nodepacks if nodepacks is not None else scan_installed_nodepacks()
    used_nodepack_folders = _workflow_nodepack_folders(rewritten)
    nodepack_resources = []
    for nodepack in nodepacks:
        if not nodepack.air or nodepack.loaded is False:
            continue
        if used_nodepack_folders is not None and nodepack.folder not in used_nodepack_folders:
            continue
        resources.add(nodepack.air)
        nodepack_resources.append(nodepack.as_dict())

    warnings = list(dict.fromkeys(model_warnings))
    if selected != included:
        warnings.append("Included upstream dependencies required to make the offloaded Comfy graph runnable")
    if (visual_region_selection or region_selection) and not explicit_selection:
        warnings.append("Using Civitai Offload Start/End markers to select the submitted graph")

    custom_input: dict[str, Any] = {"resources": sorted(resources), "workflow": rewritten}
    if trace:
        custom_input["trace"] = trace
    return OffloadBuildResult(
        steps=[{"$type": "customComfy", "input": custom_input}],
        workflow=rewritten,
        resources=sorted(resources),
        warnings=warnings,
        selected_node_ids=sorted(selected, key=_node_sort_key),
        included_node_ids=sorted(included, key=_node_sort_key),
        model_resources=resolved_models,
        nodepack_resources=nodepack_resources,
        input_blobs=input_blobs,
    )
