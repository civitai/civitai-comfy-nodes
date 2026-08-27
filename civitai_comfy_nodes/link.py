"""Civitai Link client: pairs this ComfyUI with civitai.com through the Link relay so the site's
"download via Civitai Link" button drops models straight into the local model folders and the site
sees which models are installed. Optional dependency: python-socketio[client]; without it the
package still imports and Link reports itself unavailable."""

import logging
import os
import threading
import time
import uuid

from . import config, local_models, model_cache, model_resolve
from . import link_protocol as proto

try:
    import socketio

    HAVE_SOCKETIO = True
except ImportError:
    socketio = None
    HAVE_SOCKETIO = False

_log = logging.getLogger("civitai_comfy_nodes.link")

SOCKETIO_PATH = "/api/socketio"
MAX_CONCURRENT_DOWNLOADS = 2
CONNECT_BACKOFF_MIN = 5.0
CONNECT_BACKOFF_MAX = 60.0
LIST_PUSH_EVERY = 25
STATUS_EVENT = "civitai.link.status"
ACTIVITY_EVENT = "civitai.link.activity"


def _default_sio():
    return socketio.Client(reconnection=True, reconnection_delay=2, reconnection_delay_max=60)


def _scan_local_models():
    return model_resolve.scan_local_model_files(model_resolve.model_roots_by_folder(proto.LIST_FOLDERS))


class LinkClient:
    """One relay connection. Socket handlers run on python-socketio's read-loop thread, so they only
    flip state and emit; every command is handled on its own thread."""

    def __init__(
        self,
        *,
        url: str,
        key: str | None,
        activated: bool = False,
        notify=None,
        sio_factory=None,
        downloader=None,
        hash_lookup=None,
        resolver=None,
        token_provider=None,
        scan=None,
        folder_roots=None,
    ):
        self.url = url.rstrip("/")
        self._key = key
        self._activated = activated
        self._notify = notify
        self._sio_factory = sio_factory or _default_sio
        self._downloader = downloader or local_models.stream_download
        self._hash_lookup = hash_lookup or model_resolve.lookup_model_version_by_hash
        self._resolver = resolver or model_resolve.resolve_model_air
        self._token_provider = token_provider or (lambda: config.auth_state()[0])
        self._scan = scan or _scan_local_models
        self._folder_roots = folder_roots or model_resolve._folder_paths_for
        self._sio = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._lifecycle = threading.Lock()
        self.connected = False
        self.joined = False
        self.room_ready = False
        self.last_error: str | None = None
        self.activities = proto.Activities()
        self._cancels: dict[str, threading.Event] = {}
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_DOWNLOADS)
        self._dest_locks: dict[str, threading.Lock] = {}
        self._dest_guard = threading.Lock()
        self._active: dict[str, dict] = {}
        self._resolver_lock = threading.Lock()
        self._resolver_running = False
        self._resolver_pending: list = []

    # ── lifecycle ────────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="civitai-link", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        for event in list(self._cancels.values()):
            event.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)

    def set_key(self, key: str, *, activated: bool) -> None:
        self._key = key
        self._activated = activated
        self._wake.set()

    def leave(self) -> None:
        sio = self._sio
        if sio is not None and self.connected:
            try:
                sio.emit("leave")
            except Exception:
                _log.debug("leave emit failed", exc_info=True)
        self.stop()

    def _run(self) -> None:
        backoff = CONNECT_BACKOFF_MIN
        while not self._stop.is_set():
            self._wake.clear()
            if not self._key:
                self._wake.wait()
                continue
            sio = self._sio_factory()
            self._sio = sio
            self._bind(sio)
            try:
                sio.connect(self.url, transports=["websocket"], socketio_path=SOCKETIO_PATH, wait_timeout=10)
            except Exception as e:
                self._set_error(f"Cannot reach {self.url}: {e}")
                self._sio = None
                self._wake.wait(backoff)
                backoff = min(backoff * 2, CONNECT_BACKOFF_MAX)
                continue
            backoff = CONNECT_BACKOFF_MIN
            self._wake.wait()
            self._disconnect(sio)
        self._sio = None

    def _disconnect(self, sio) -> None:
        try:
            sio.disconnect()
        except Exception:
            _log.debug("disconnect failed", exc_info=True)
        self.connected = False
        self.joined = False
        self.room_ready = False
        self._changed()

    def _bind(self, sio) -> None:
        sio.on("connect", self._on_connect)
        sio.on("disconnect", self._on_disconnect)
        sio.on("connect_error", self._on_connect_error)
        sio.on("upgradeKey", self._on_upgrade_key)
        sio.on("kicked", self._on_kicked)
        sio.on("error", self._on_error)
        sio.on("roomPresence", self._on_presence)
        sio.on("command", self._on_command)

    # ── socket handlers (read-loop thread: never block, never call()) ────────────────────────

    def _on_connect(self, *args) -> None:
        self.connected = True
        self.joined = False
        self.last_error = None
        sio = self._sio
        sio.emit("iam", {"type": "sd"})
        if self._key:
            sio.emit("join", self._key, callback=self._on_join_ack)
        self._changed()

    def _on_join_ack(self, *args) -> None:
        result = args[0] if args and isinstance(args[0], dict) else {}
        self.joined = bool(result.get("success"))
        if not self.joined:
            self.last_error = result.get("msg") or "civitai.com rejected the join"
        self._changed()

    def _on_upgrade_key(self, data=None) -> None:
        key = data.get("key") if isinstance(data, dict) else None
        if not key:
            return
        self._key = key
        self._activated = True
        try:
            config.save_link_key(key, activated=True)
        except OSError:
            _log.warning("Could not persist the upgraded Civitai Link key", exc_info=True)
        self.joined = True
        self._changed()

    def _on_kicked(self, *args) -> None:
        self._key = None
        self._activated = False
        self.joined = False
        self.room_ready = False
        config.clear_link_key()
        self.last_error = "civitai.com rejected this pairing — generate a new code and pair again"
        self._wake.set()
        self._changed()

    def _on_error(self, data=None) -> None:
        msg = data.get("msg") if isinstance(data, dict) else str(data or "")
        self._set_error(msg or "Link relay error")

    def _on_connect_error(self, data=None) -> None:
        self._set_error(f"Connection error: {data}")

    def _on_presence(self, data=None) -> None:
        self.room_ready = bool(isinstance(data, dict) and (data.get("client") or 0) > 0)
        self._changed()

    def _on_disconnect(self, *args) -> None:
        self.connected = False
        self.joined = False
        self.room_ready = False
        self._changed()

    def _on_command(self, payload=None) -> None:
        if not isinstance(payload, dict) or not payload.get("type"):
            return
        _log.info("Civitai Link command %s (%s)", payload.get("type"), payload.get("id"))
        threading.Thread(target=self._dispatch, args=(payload,), name="civitai-link-cmd", daemon=True).start()

    # ── command dispatch ─────────────────────────────────────────────────────────────────────

    def _dispatch(self, command: dict) -> None:
        handlers = {
            "resources:list": self._handle_list,
            "resources:add": self._handle_add,
            "resources:remove": self._handle_remove,
            "activities:cancel": self._handle_cancel,
            "activities:list": self._handle_activities,
            "activities:clear": self._handle_activities_clear,
        }
        handler = handlers.get(command["type"])
        try:
            if handler is None:
                self._send(proto.make_response(command, "error", error=f"Unsupported command '{command['type']}'"))
            else:
                handler(command)
        except Exception as e:
            _log.exception("Civitai Link command %s failed", command.get("type"))
            self._send(proto.make_response(command, "error", error=str(e)))

    def _send(self, response: dict) -> None:
        if response.get("type") in proto.ACTIVITY_TYPES:
            self.activities.upsert(response)
            self._emit_local(ACTIVITY_EVENT, response)
        sio = self._sio
        if sio is None or not self.joined:
            return
        try:
            sio.emit("commandStatus", response)
        except Exception:
            _log.debug("commandStatus emit failed", exc_info=True)

    def _handle_list(self, command: dict) -> None:
        entries, pending = self._current_resources()
        _log.info("Civitai Link resources:list → %d reported, %d still to resolve", len(entries), len(pending))
        self._send(proto.make_response(command, "success", resources=entries))
        if pending:
            self._start_resolver(pending)

    def _current_resources(self) -> tuple[list[dict], list]:
        records = self._scan()
        cache = model_cache.bulk_get([record.path for record in records])
        entries, pending = proto.build_resource_list(records, cache)
        for active in list(self._active.values()):
            resource = active["resource"]
            entries.insert(
                0,
                proto.resource_entry(
                    active["folder"], os.path.basename(active["dest"]), resource["hash"], downloading=True
                ),
            )
        return proto.trim_resource_list(entries), pending

    def _push_resource_list(self) -> None:
        entries, _pending = self._current_resources()
        command = {"id": uuid.uuid4().hex, "type": "resources:list", "createdAt": proto.now_iso()}
        self._send(proto.make_response(command, "success", resources=entries))

    def _start_resolver(self, records: list) -> None:
        with self._resolver_lock:
            self._resolver_pending = records
            if self._resolver_running:
                return
            self._resolver_running = True
        threading.Thread(target=self._resolve_loop, name="civitai-link-resolver", daemon=True).start()

    def _resolve_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._resolver_lock:
                    batch, self._resolver_pending = self._resolver_pending, []
                if not batch:
                    return
                token = self._token()
                for index, record in enumerate(batch, 1):
                    if self._stop.is_set():
                        return
                    try:
                        self._resolver(record.path, token=token)
                    except Exception:
                        _log.debug("Could not resolve %s", record.path, exc_info=True)
                    if index % LIST_PUSH_EVERY == 0:
                        self._push_resource_list()
                self._push_resource_list()
        finally:
            with self._resolver_lock:
                self._resolver_running = False

    def cancel(self, command_id: str) -> bool:
        event = self._cancels.get(str(command_id or ""))
        if event is None:
            return False
        event.set()
        return True

    def _handle_add(self, command: dict) -> None:
        command_id = str(command.get("id") or "")
        resource = command.get("resource")
        if isinstance(resource, dict) and isinstance(resource.get("hash"), str):
            # The site filters progress by its own lowercase hashes against what we echo back.
            command = {**command, "resource": {**resource, "hash": resource["hash"].lower()}}
        cancel = threading.Event()
        self._cancels[command_id] = cancel
        self._send(proto.make_response(command, "pending", progress=0))
        try:
            with self._slots:
                if cancel.is_set():
                    self._send(proto.make_response(command, "canceled"))
                    return
                self._run_add(command, cancel)
        finally:
            self._cancels.pop(command_id, None)

    def _run_add(self, command: dict, cancel: threading.Event) -> None:
        resource = command.get("resource") or {}
        try:
            name = proto.safe_filename(resource.get("name"))
            expected = proto.normalize_sha256(resource.get("hash"))
        except ValueError as e:
            self._send(proto.make_response(command, "error", error=str(e)))
            return
        resource_type = resource.get("type")
        if proto.folder_for_resource(resource_type) is None:
            self._send(proto.make_response(command, "error", error=f"Unsupported resource type '{resource_type}'"))
            return
        version = self._lookup_version(expected)
        file = model_resolve.version_file_for_hash(version, expected)
        folder = proto.folder_for_resource(resource_type, (file or {}).get("type"))
        if model_cache.find_by_hash(expected):
            self._send(proto.make_response(command, "success", progress=100))
            self._push_resource_list()
            return
        dest_dir = local_models._model_dir(folder)
        dest = proto.unique_destination(dest_dir, name, expected)
        with self._dest_lock(dest):
            self._active[str(command.get("id") or "")] = {"resource": resource, "dest": dest, "folder": folder}
            try:
                self._send(proto.make_response(command, "processing", progress=0))
                self._push_resource_list()
                sha256 = self._download(command, resource.get("url"), dest, cancel)
            except local_models.DownloadCanceledError:
                self._send(proto.make_response(command, "canceled"))
                self._active.pop(str(command.get("id") or ""), None)
                self._push_resource_list()
                return
            except Exception as e:
                self._send(proto.make_response(command, "error", error=str(e)))
                self._active.pop(str(command.get("id") or ""), None)
                self._push_resource_list()
                return
            finally:
                self._active.pop(str(command.get("id") or ""), None)
        if sha256.lower() != expected:
            try:
                os.remove(dest)
            except OSError:
                pass
            self._send(proto.make_response(command, "error", error="Downloaded file failed its checksum"))
            return
        upper = sha256.upper()
        model_cache.put(
            dest,
            hashes={"SHA256": upper, "AutoV2": upper[:10]},
            air=(version or {}).get("air"),
            model_version_id=(version or {}).get("id"),
        )
        self._send(proto.make_response(command, "success", progress=100))
        self._push_resource_list()

    def _download(self, command: dict, url: str | None, dest: str, cancel: threading.Event) -> str:
        if not url:
            raise ValueError("Resource has no download url")
        started = time.monotonic()
        last_report = [started]

        def on_progress(written: int, total: int) -> None:
            now = time.monotonic()
            if now - last_report[0] < proto.PROGRESS_INTERVAL:
                return
            last_report[0] = now
            self._send(
                proto.make_response(command, "processing", **proto.progress_fields(written, total, started, now))
            )

        # The site hands over a signed URL; an Authorization header on a presigned URL breaks its
        # signature, so the token is only tried when the unsigned request is refused.
        try:
            return self._downloader(url, dest, token=None, on_progress=on_progress, cancel=cancel)
        except local_models.DownloadHttpError as e:
            token = self._token() if e.status_code in (401, 403) else None
            if not token:
                raise
            return self._downloader(url, dest, token=token, on_progress=on_progress, cancel=cancel)

    def _handle_remove(self, command: dict) -> None:
        resource = command.get("resource") or {}
        expected = proto.normalize_sha256(resource.get("hash"))
        path = self._locate(resource.get("path"), expected)
        if not path:
            self._send(proto.make_response(command, "error", error="Resource not found locally"))
            return
        cached = str(((model_cache.get(path) or {}).get("hashes") or {}).get("SHA256") or "").lower()
        if cached != expected:
            self._send(proto.make_response(command, "error", error="Local file no longer matches the resource hash"))
            return
        for candidate in (path, path + ".part"):
            try:
                os.remove(candidate)
            except FileNotFoundError:
                pass
        model_cache.remove(path)
        self._send(proto.make_response(command, "success"))
        self._push_resource_list()

    def _locate(self, resource_path: str | None, sha256: str) -> str | None:
        parsed = proto.parse_resource_path(resource_path)
        if parsed:
            folder, relative = parsed
            for root in self._folder_roots(folder):
                candidate = os.path.join(root, *relative.split("/"))
                real_root = os.path.realpath(root)
                if os.path.realpath(candidate).startswith(real_root + os.sep) and os.path.isfile(candidate):
                    return candidate
        return model_cache.find_by_hash(sha256)

    def _handle_cancel(self, command: dict) -> None:
        if not self.cancel(command.get("activityId")):
            self._send(proto.make_response(command, "error", error="No such download"))
            return
        self._send(proto.make_response(command, "success"))

    def _handle_activities(self, command: dict) -> None:
        self._send(proto.make_response(command, "success", activities=self.activities.list()))

    def _handle_activities_clear(self, command: dict) -> None:
        self.activities.clear()
        self._send(proto.make_response(command, "success", activities=[]))

    # ── helpers ──────────────────────────────────────────────────────────────────────────────

    def _lookup_version(self, sha256: str) -> dict | None:
        try:
            return self._hash_lookup(sha256, token=self._token())
        except Exception:
            _log.debug("by-hash lookup failed for %s", sha256, exc_info=True)
            return None

    def _token(self) -> str | None:
        try:
            return self._token_provider()
        except Exception:
            return None

    def _dest_lock(self, dest: str) -> threading.Lock:
        with self._dest_guard:
            return self._dest_locks.setdefault(dest, threading.Lock())

    def _set_error(self, message: str) -> None:
        self.last_error = message
        _log.warning("Civitai Link: %s", message)
        self._changed()

    def _changed(self) -> None:
        self._emit_local(STATUS_EVENT, self.status())

    def _emit_local(self, event: str, data: dict) -> None:
        if self._notify is None:
            return
        try:
            self._notify(event, data)
        except Exception:
            _log.debug("notify %s failed", event, exc_info=True)

    def status(self) -> dict:
        return {
            "paired": bool(self._key),
            "activated": self._activated,
            "keyHint": (self._key or "")[-4:],
            "connected": self.connected,
            "joined": self.joined,
            "roomReady": self.room_ready,
            "lastError": self.last_error,
            "downloads": len(self._active),
        }


# ── module singleton ─────────────────────────────────────────────────────────────────────────

_client: LinkClient | None = None
_lock = threading.Lock()
_notify = None
_registered = False


def available() -> bool:
    return HAVE_SOCKETIO


def disabled_reason() -> str | None:
    if not HAVE_SOCKETIO:
        return "python-socketio[client] is not installed"
    if config.is_hosted_session():
        return "not available in hosted sessions"
    if not config.stored_enable_link():
        return "disabled in settings"
    return None


def register(notify=None) -> None:
    global _notify, _registered
    if notify is not None:
        _notify = notify
    if _registered:
        return
    _registered = True
    reconfigure()


def reconfigure() -> None:
    """Apply the current settings + stored key: stop any running client and start a fresh one."""
    global _client
    with _lock:
        previous, _client = _client, None
    if previous is not None:
        previous.stop()
    if disabled_reason():
        return
    stored = config.load_link_key()
    if not stored:
        return
    client = LinkClient(url=config.link_url(), key=stored["key"], activated=stored["activated"], notify=_notify)
    with _lock:
        _client = client
    client.start()


def pair(code: str) -> dict:
    key = proto.normalize_key(code)
    reason = disabled_reason()
    if reason:
        raise ValueError(f"Civitai Link is {reason}")
    config.save_link_key(key, activated=bool(proto.UPGRADED_KEY_RE.match(key)))
    reconfigure()
    return status()


def unpair() -> dict:
    global _client
    with _lock:
        client, _client = _client, None
    if client is not None:
        client.leave()
    config.clear_link_key()
    return status()


def cancel(command_id: str) -> dict:
    client = _client
    if client is None or not client.cancel(command_id):
        raise ValueError("No such download")
    return status()


def status() -> dict:
    stored = config.load_link_key()
    payload = {
        "available": HAVE_SOCKETIO,
        "enabled": config.stored_enable_link(),
        "hosted": config.is_hosted_session(),
        "disabledReason": disabled_reason(),
        "paired": bool(stored),
        "activated": bool(stored and stored["activated"]),
        "keyHint": stored["key"][-4:] if stored else "",
        "url": config.link_url(),
        "urlSource": "env"
        if os.environ.get("CIVITAI_LINK_URL")
        else "stored"
        if config.stored_link_url()
        else "default",
        "connected": False,
        "joined": False,
        "roomReady": False,
        "lastError": None,
        "downloads": 0,
        "activities": [],
    }
    client = _client
    if client is not None:
        payload.update(client.status())
        payload["activities"] = client.activities.list()
    return payload
