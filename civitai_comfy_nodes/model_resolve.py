"""Resolve local model files to Civitai versions: safetensors header hashes, computed hashes, the
by-hash API and a persistent cache. Import-safe outside ComfyUI (folder_paths is imported lazily)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from . import model_cache, oauth
from .errors import CivitaiNodeError
from .local_models import AIR_TYPE_FOLDERS, USER_AGENT

CIVITAI_BASE_URL = os.environ.get("CIVITAI_BASE_URL") or oauth.OAUTH_BASE
MODEL_EXTENSIONS = {".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin"}
SAFETENSORS_MAX_HEADER = 16 * 1024 * 1024
MODEL_FOLDERS = tuple(dict.fromkeys(AIR_TYPE_FOLDERS.values()))
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


def compute_model_hashes(path: str | os.PathLike[str], *, include_autov3: bool = True) -> dict[str, str]:
    """Compute the Civitai hash subset needed for local model AIR lookup.

    Mirrors the scanner behavior for SHA256, AutoV1, AutoV2, AutoV3, and CRC32. Blake3 is omitted
    because this package intentionally has no native hashing dependency. ``include_autov3=False``
    skips the AutoV3 payload digest, which is a *second* full-file pass — callers that resolve via
    SHA256 first can defer it until that lookup misses.
    """
    sha256, crc32 = _sha256_file(path)
    hashes = {"SHA256": sha256, "AutoV2": sha256[:10], "CRC32": crc32}
    autov1 = _autov1_file(path)
    if autov1:
        hashes["AutoV1"] = autov1
    if include_autov3:
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


def version_file_for_hash(version: dict[str, Any] | None, hash_value: str) -> dict[str, Any] | None:
    """The file entry of a model version whose `hashes` (any type) contain `hash_value`."""
    wanted = (hash_value or "").strip().upper()
    if not version or not wanted:
        return None
    for file in version.get("files") or []:
        file_hashes = file.get("hashes") if isinstance(file, dict) else None
        if isinstance(file_hashes, dict) and any(str(v).upper() == wanted for v in file_hashes.values()):
            return file
    return None


def _with_canonical_sha256(hashes: dict[str, str], source: str, version: dict[str, Any], hash_value: str) -> dict:
    # A file can't embed its own whole-file digest, so a "SHA256" read from safetensors metadata is
    # some tensor-payload hash; only the version file record knows the real one.
    merged = dict(hashes)
    if source == "metadata":
        merged.pop("SHA256", None)
    file = version_file_for_hash(version, hash_value)
    sha256 = (file or {}).get("hashes", {}).get("SHA256") if file else None
    if sha256:
        merged["SHA256"] = str(sha256).upper()
    return merged


def _lookup_record(
    path: str | os.PathLike[str],
    hashes: dict[str, str],
    source: str,
    *,
    token: str | None,
    session: requests.Session | None,
    civitai_base_url: str | None,
) -> LocalModelRecord | None:
    for hash_type, hash_value in _lookup_candidates(hashes):
        version = lookup_model_version_by_hash(
            hash_value, token=token, session=session, civitai_base_url=civitai_base_url
        )
        if version:
            return LocalModelRecord(
                folder="",
                name=Path(path).name,
                path=str(path),
                hashes=_with_canonical_sha256(hashes, source, version, hash_value),
                hash_source=source,
                air=version.get("air"),
                model_version_id=version.get("id"),
                lookup_hash_type=hash_type,
                lookup_hash=hash_value,
            )
    return None


def resolve_model_air(
    path: str | os.PathLike[str],
    *,
    token: str | None = None,
    session: requests.Session | None = None,
    civitai_base_url: str | None = None,
) -> LocalModelRecord | None:
    """Resolve a local file to AIR: persistent cache, then safetensors metadata hashes, then computed
    hashes. Computing the SHA256 of a multi-GB file is the expensive step, so a cache keyed on file
    identity short-circuits repeat resolutions, and the AutoV3 payload hash (a second full-file pass)
    is only computed when SHA256 and friends miss."""
    cached = model_cache.get(path)
    if cached and cached.get("air"):
        return LocalModelRecord(
            folder="",
            name=Path(path).name,
            path=str(path),
            hashes=cached.get("hashes") or {},
            hash_source="cache",
            air=cached["air"],
            model_version_id=cached.get("model_version_id"),
        )

    metadata_hashes = read_model_hashes_from_metadata(path)
    if metadata_hashes:
        record = _lookup_record(
            path, metadata_hashes, "metadata", token=token, session=session, civitai_base_url=civitai_base_url
        )
        if record:
            model_cache.put(path, hashes=record.hashes, air=record.air, model_version_id=record.model_version_id)
            return record

    computed_hashes = dict((cached or {}).get("hashes") or {})
    if not computed_hashes:
        computed_hashes = compute_model_hashes(path, include_autov3=False)
    record = _lookup_record(
        path, computed_hashes, "computed", token=token, session=session, civitai_base_url=civitai_base_url
    )
    if not record and "AutoV3" not in computed_hashes:
        autov3 = _autov3_safetensors_payload(path)
        if autov3:
            computed_hashes["AutoV3"] = autov3
            record = _lookup_record(
                path, {"AutoV3": autov3}, "computed", token=token, session=session, civitai_base_url=civitai_base_url
            )
    model_cache.put(
        path,
        hashes=computed_hashes,
        air=record.air if record else None,
        model_version_id=record.model_version_id if record else None,
    )
    return record


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
