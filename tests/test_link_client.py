import hashlib
import importlib
import os
import stat
import sys
import threading
import time

import pytest

from civitai_comfy_nodes import config, link, local_models, model_cache
from civitai_comfy_nodes.link import LinkClient
from civitai_comfy_nodes.model_resolve import LocalModelRecord

DATA = b"lora-weights"
SHA = hashlib.sha256(DATA).hexdigest()
KEY = "abc123"


def wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeSio:
    def __init__(self):
        self.handlers = {}
        self.emitted = []
        self.connect_kwargs = None
        self.disconnected = False

    def on(self, event, handler=None):
        self.handlers[event] = handler

    def emit(self, event, data=None, callback=None):
        self.emitted.append((event, data, callback))

    def connect(self, url, **kwargs):
        self.connect_kwargs = {"url": url, **kwargs}
        self.handlers["connect"]()

    def disconnect(self):
        self.disconnected = True
        self.handlers["disconnect"]()

    def statuses(self, event_type=None):
        return [
            d["status"]
            for e, d, _ in self.emitted
            if e == "commandStatus" and (event_type is None or d.get("type") == event_type)
        ]

    def last(self, event_type):
        return [d for e, d, _ in self.emitted if e == "commandStatus" and d.get("type") == event_type][-1]


def fake_downloader(calls, data=DATA, fail_first=None):
    def download(url, dest, *, token=None, on_progress=None, cancel=None, in_execution=False):
        calls.append({"url": url, "dest": dest, "token": token})
        if fail_first and len(calls) == 1:
            raise fail_first
        if cancel is not None and cancel.is_set():
            raise local_models.DownloadCanceledError("canceled")
        with open(dest, "wb") as out:
            out.write(data)
        if on_progress:
            on_progress(len(data), len(data))
        return hashlib.sha256(data).hexdigest()

    return download


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_MODEL_CACHE", str(tmp_path / "cache.json"))
    monkeypatch.setenv("CIVITAI_COMFY_LINK_STORE", str(tmp_path / "link.json"))
    monkeypatch.setenv("CIVITAI_COMFY_SETTINGS_STORE", str(tmp_path / "settings.json"))
    monkeypatch.delenv("CIVITAI_COMFY_SESSION_ID", raising=False)
    monkeypatch.delenv("CIVITAI_LINK_URL", raising=False)
    models = tmp_path / "models"

    def model_dir(folder):
        path = models / folder
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setattr(local_models, "_model_dir", model_dir)
    return models


def make_client(env, **overrides):
    fake = FakeSio()
    events = []
    calls = []
    kwargs = dict(
        url="http://relay/",
        key=KEY,
        notify=lambda event, data: events.append((event, data)),
        sio_factory=lambda: fake,
        downloader=fake_downloader(calls),
        hash_lookup=lambda sha256, token=None: None,
        resolver=lambda path, token=None: None,
        token_provider=lambda: None,
        scan=lambda: [],
        folder_roots=lambda folder: [str(env / folder)],
    )
    kwargs.update(overrides)
    client = LinkClient(**kwargs)
    client._sio = fake
    client.joined = True
    return client, fake, events, calls


def add_command(command_id="cmd-1", **resource):
    base = {
        "type": "LORA",
        "hash": SHA.upper(),
        "name": "my lora.safetensors",
        "modelName": "My LoRA",
        "url": "http://dl/x",
    }
    base.update(resource)
    return {"id": command_id, "type": "resources:add", "createdAt": "2026-01-01T00:00:00Z", "resource": base}


# ── socket lifecycle ─────────────────────────────────────────────────────────────────────────────


def test_connect_emits_iam_then_join_over_websocket(env):
    client, fake, events, _ = make_client(env)
    client._sio = None
    client.joined = False
    client.start()
    assert wait_for(lambda: fake.connect_kwargs is not None)
    assert fake.connect_kwargs["url"] == "http://relay"
    assert fake.connect_kwargs["transports"] == ["websocket"]
    assert fake.connect_kwargs["socketio_path"] == "/api/socketio"
    assert fake.emitted[0] == ("iam", {"type": "sd"}, None)
    assert fake.emitted[1][:2] == ("join", KEY)
    assert not client.joined
    fake.emitted[1][2]({"success": True, "msg": "Joined"})
    assert client.joined and client.connected
    assert any(e == "civitai.link.status" and d["joined"] for e, d in events)
    client.stop()
    assert fake.disconnected
    assert not client.connected


def test_join_ack_failure_records_error(env):
    client, fake, _, _ = make_client(env)
    client._bind(fake)
    fake.handlers["connect"]()
    fake.emitted[1][2]({"success": False, "msg": "You must identify yourself first"})
    assert not client.joined
    assert client.last_error == "You must identify yourself first"


def test_upgrade_key_is_persisted_activated_and_private(env, tmp_path):
    client, fake, _, _ = make_client(env)
    client._bind(fake)
    fake.handlers["upgradeKey"]({"key": "f" * 128})
    stored = config.load_link_key()
    assert stored == {"key": "f" * 128, "activated": True, "instance_id": None, "paired_at": stored["paired_at"]}
    assert stat.S_IMODE(os.stat(tmp_path / "link.json").st_mode) == 0o600
    assert client.status()["keyHint"] == "ffff" and client.status()["activated"]


def test_kicked_forgets_the_pairing(env):
    config.save_link_key(KEY, activated=False)
    client, fake, _, _ = make_client(env)
    client._bind(fake)
    fake.handlers["kicked"]()
    assert config.load_link_key() is None
    assert client.status()["paired"] is False
    assert "pair again" in client.last_error


def test_presence_marks_room_ready_only_with_a_browser(env):
    client, fake, _, _ = make_client(env)
    client._bind(fake)
    fake.handlers["roomPresence"]({"client": 0, "sd": 1})
    assert not client.room_ready
    fake.handlers["roomPresence"]({"client": 1, "sd": 1})
    assert client.room_ready


def test_command_before_join_is_answered_locally_but_not_sent(env):
    client, fake, events, _ = make_client(env)
    client.joined = False
    client._dispatch({"id": "1", "type": "activities:list"})
    assert fake.statuses() == []
    client._dispatch(add_command("2", type="Poses"))
    assert fake.statuses() == []
    assert client.activities.get("2")["status"] == "error"
    assert any(e == "civitai.link.activity" for e, _ in events)


# ── resources:add ────────────────────────────────────────────────────────────────────────────────


def test_add_downloads_verifies_and_reports(env):
    client, fake, events, calls = make_client(env)
    client._dispatch(add_command())
    assert fake.statuses("resources:add") == ["pending", "processing", "success"]
    dest = env / "loras" / "my lora.safetensors"
    assert dest.read_bytes() == DATA
    assert calls[0]["token"] is None
    assert fake.last("resources:add")["progress"] == 100
    assert fake.last("resources:add")["resource"]["modelName"] == "My LoRA"
    assert model_cache.get(dest)["hashes"]["SHA256"] == SHA.upper()
    started, finished = [d for e, d, _ in fake.emitted if d.get("type") == "resources:list"]
    assert started["resources"][0]["downloading"] is True and started["resources"][0]["hash"] == SHA
    assert finished["resources"] == []  # fake scan is empty; the push itself is what matters
    assert fake.last("resources:add")["resource"]["hash"] == SHA  # echoed lowercase for the site's filters
    assert [d["status"] for e, d in events if e == "civitai.link.activity"] == ["pending", "processing", "success"]


def test_add_folder_follows_the_file_type_from_the_hash_lookup(env):
    version = {
        "id": 9,
        "air": "urn:air:x:checkpoint:civitai:1@9",
        "files": [{"type": "Diffusion Model", "hashes": {"SHA256": SHA.upper()}}],
    }
    client, fake, _, _ = make_client(env, hash_lookup=lambda sha256, token=None: version)
    client._dispatch(add_command(type="Checkpoint", name="m.safetensors"))
    assert (env / "diffusion_models" / "m.safetensors").exists()
    assert model_cache.get(env / "diffusion_models" / "m.safetensors")["air"] == version["air"]


def test_add_lookup_failure_is_not_fatal(env):
    def boom(sha256, token=None):
        raise RuntimeError("civitai down")

    client, fake, _, _ = make_client(env, hash_lookup=boom)
    client._dispatch(add_command())
    assert fake.statuses("resources:add")[-1] == "success"


def test_add_checksum_mismatch_deletes_the_file(env):
    client, fake, _, _ = make_client(env)
    client._dispatch(add_command(hash="b" * 64))
    assert fake.statuses("resources:add")[-1] == "error"
    assert "checksum" in fake.last("resources:add")["error"]
    assert not (env / "loras" / "my lora.safetensors").exists()


def test_add_rejects_unsupported_types_and_bad_names(env):
    client, fake, _, calls = make_client(env)
    client._dispatch(add_command("1", type="Poses"))
    client._dispatch(add_command("2", name=".."))
    client._dispatch(add_command("3", hash="nope"))
    assert fake.statuses("resources:add") == ["pending", "error"] * 3
    assert calls == []


def test_add_keeps_traversal_names_inside_the_model_folder(env):
    client, fake, _, _ = make_client(env)
    client._dispatch(add_command(name="../../evil.safetensors"))
    assert fake.statuses("resources:add")[-1] == "success"
    assert (env / "loras" / "evil.safetensors").exists()
    assert not (env.parent / "evil.safetensors").exists()


def test_add_already_present_skips_download(env):
    existing = env / "loras" / "existing.safetensors"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(DATA)
    model_cache.put(existing, hashes={"SHA256": SHA.upper()})
    client, fake, _, calls = make_client(env)
    client._dispatch(add_command())
    assert fake.statuses("resources:add") == ["pending", "success"]
    assert calls == []


def test_add_retries_with_token_only_after_401(env):
    calls = []
    client, fake, _, _ = make_client(
        env,
        downloader=fake_downloader(calls, fail_first=local_models.DownloadHttpError("gated", 401)),
        token_provider=lambda: "tok",
    )
    client._dispatch(add_command())
    assert [c["token"] for c in calls] == [None, "tok"]
    assert fake.statuses("resources:add")[-1] == "success"


def test_add_without_token_reports_the_http_error(env):
    calls = []
    client, fake, _, _ = make_client(
        env, downloader=fake_downloader(calls, fail_first=local_models.DownloadHttpError("gated", 403))
    )
    client._dispatch(add_command())
    assert fake.statuses("resources:add")[-1] == "error" and "gated" in fake.last("resources:add")["error"]
    assert len(calls) == 1


def test_add_never_clobbers_a_different_existing_file(env):
    taken = env / "loras" / "my lora.safetensors"
    taken.parent.mkdir(parents=True)
    taken.write_bytes(b"something else")
    client, fake, _, calls = make_client(env)
    client._dispatch(add_command())
    assert taken.read_bytes() == b"something else"
    assert calls[0]["dest"] == str(env / "loras" / f"my lora_{SHA[:8]}.safetensors")
    assert fake.statuses("resources:add")[-1] == "success"


def test_add_cancel_mid_download_reports_canceled(env):
    calls = []
    client, fake, _, _ = make_client(env)

    def download(url, dest, *, token=None, on_progress=None, cancel=None, in_execution=False):
        calls.append(dest)
        client._dispatch({"id": "c", "type": "activities:cancel", "activityId": "cmd-1"})
        assert cancel.is_set()
        raise local_models.DownloadCanceledError("canceled")

    client._downloader = download
    client._dispatch(add_command())
    assert fake.statuses("resources:add") == ["pending", "processing", "canceled"]
    assert fake.last("activities:cancel")["status"] == "success"
    assert not (env / "loras" / "my lora.safetensors").exists()
    assert fake.last("resources:list")["resources"] == []  # the in-flight entry is withdrawn
    assert client._active == {}
    client._dispatch({"id": "c2", "type": "activities:cancel", "activityId": "gone"})
    assert fake.last("activities:cancel")["status"] == "error"


def test_third_concurrent_add_waits_for_a_slot(env):
    release = threading.Event()
    client, fake, _, _ = make_client(env)
    started = []

    def download(url, dest, *, token=None, on_progress=None, cancel=None, in_execution=False):
        started.append(dest)
        release.wait(5)
        data = os.path.basename(dest).encode()
        with open(dest, "wb") as out:
            out.write(data)
        return hashlib.sha256(data).hexdigest()

    def command(i):
        name = f"m{i}.safetensors"
        return add_command(f"cmd-{i}", name=name, hash=hashlib.sha256(name.encode()).hexdigest())

    client._downloader = download
    threads = [threading.Thread(target=client._dispatch, args=(command(i),)) for i in range(3)]
    for t in threads:
        t.start()
    assert wait_for(lambda: len(started) == 2)
    time.sleep(0.1)
    assert len(started) == 2  # the third holds at "pending" until a slot frees
    assert fake.statuses("resources:add").count("pending") == 3
    release.set()
    for t in threads:
        t.join(5)
    assert len(started) == 3
    assert fake.statuses("resources:add").count("success") == 3


def test_stop_cancels_in_flight_downloads(env):
    client, fake, _, _ = make_client(env)
    seen = {}

    def download(url, dest, *, token=None, on_progress=None, cancel=None, in_execution=False):
        seen["cancel"] = cancel
        client.stop()
        assert cancel.is_set()
        raise local_models.DownloadCanceledError("stopped")

    client._downloader = download
    client._dispatch(add_command())
    assert fake.statuses("resources:add")[-1] == "canceled"


# ── resources:list / remove / activities ─────────────────────────────────────────────────────────


def test_list_replies_from_cache_then_resolves_the_rest_in_background(env):
    loras = env / "loras"
    loras.mkdir(parents=True)
    cached = loras / "cached.safetensors"
    cached.write_bytes(DATA)
    model_cache.put(cached, hashes={"SHA256": SHA.upper()})
    fresh = loras / "sub" / "fresh.safetensors"
    fresh.parent.mkdir()
    fresh.write_bytes(b"other")
    fresh_sha = hashlib.sha256(b"other").hexdigest()
    records = [
        LocalModelRecord(folder="loras", name="cached.safetensors", path=str(cached)),
        LocalModelRecord(folder="loras", name=os.path.join("sub", "fresh.safetensors"), path=str(fresh)),
    ]
    resolved = []

    def resolver(path, token=None):
        resolved.append(path)
        model_cache.put(path, hashes={"SHA256": fresh_sha.upper()})

    client, fake, _, _ = make_client(env, scan=lambda: records, resolver=resolver)
    client._dispatch({"id": "L1", "type": "resources:list", "createdAt": "x"})
    reply = [d for e, d, _ in fake.emitted if d.get("id") == "L1"][0]
    assert reply["status"] == "success"
    assert reply["resources"] == [
        {
            "type": "LORA",
            "hash": SHA,
            "name": "cached.safetensors",
            "path": "loras/cached.safetensors",
            "hasPreview": "",
        }
    ]
    assert wait_for(lambda: not client._resolver_running and len(fake.statuses("resources:list")) >= 2)
    assert resolved == [str(fresh)]
    pushed = fake.last("resources:list")
    assert pushed["id"] != "L1"
    assert {r["hash"] for r in pushed["resources"]} == {SHA, fresh_sha}
    assert [r["path"] for r in pushed["resources"] if r["hash"] == fresh_sha] == ["loras/sub/fresh.safetensors"]


def test_list_marks_in_flight_downloads(env):
    client, fake, _, _ = make_client(env)
    client._active["cmd-9"] = {
        "resource": {"hash": SHA.upper()},
        "dest": str(env / "loras" / "x.safetensors"),
        "folder": "loras",
    }
    client._dispatch({"id": "L", "type": "resources:list"})
    entry = fake.last("resources:list")["resources"][0]
    assert entry["downloading"] is True and entry["hash"] == SHA and entry["path"] == "loras/x.safetensors"


def test_cancel_api_sets_the_event_and_rejects_unknown_ids(env):
    client, fake, _, _ = make_client(env)
    seen = {}

    def download(url, dest, *, token=None, on_progress=None, cancel=None, in_execution=False):
        seen["cancelled"] = client.cancel("cmd-1")
        assert cancel.is_set()
        raise local_models.DownloadCanceledError("canceled")

    client._downloader = download
    client._dispatch(add_command())
    assert seen["cancelled"] is True
    assert client.cancel("nope") is False


def test_remove_deletes_file_part_and_cache(env):
    loras = env / "loras"
    loras.mkdir(parents=True)
    path = loras / "gone.safetensors"
    path.write_bytes(DATA)
    (loras / "gone.safetensors.part").write_bytes(b"x")
    model_cache.put(path, hashes={"SHA256": SHA.upper()})
    client, fake, _, _ = make_client(env)
    client._dispatch(
        {
            "id": "R",
            "type": "resources:remove",
            "resource": {
                "type": "LORA",
                "hash": SHA,
                "name": "gone.safetensors",
                "path": "loras/gone.safetensors",
                "modelName": "m",
            },
        }
    )
    assert fake.last("resources:remove")["status"] == "success"
    assert not path.exists() and not (loras / "gone.safetensors.part").exists()
    assert model_cache.get(path) is None
    assert fake.statuses("resources:list") == ["success"]


def test_remove_falls_back_to_hash_when_path_is_unsafe(env):
    loras = env / "loras"
    loras.mkdir(parents=True)
    path = loras / "by-hash.safetensors"
    path.write_bytes(DATA)
    model_cache.put(path, hashes={"SHA256": SHA.upper()})
    client, fake, _, _ = make_client(env)
    client._dispatch(
        {"id": "R", "type": "resources:remove", "resource": {"hash": SHA, "path": "loras/../by-hash.safetensors"}}
    )
    assert fake.last("resources:remove")["status"] == "success"
    assert not path.exists()


def test_remove_refuses_when_local_file_no_longer_matches(env):
    loras = env / "loras"
    loras.mkdir(parents=True)
    path = loras / "changed.safetensors"
    path.write_bytes(DATA)
    model_cache.put(path, hashes={"SHA256": "c" * 64})
    client, fake, _, _ = make_client(env)
    client._dispatch(
        {"id": "R", "type": "resources:remove", "resource": {"hash": SHA, "path": "loras/changed.safetensors"}}
    )
    assert fake.last("resources:remove")["status"] == "error"
    assert path.exists()
    client._dispatch(
        {"id": "R2", "type": "resources:remove", "resource": {"hash": "d" * 64, "path": "loras/missing.safetensors"}}
    )
    assert "not found" in fake.last("resources:remove")["error"]


def test_activities_list_clear_and_unknown_commands(env):
    client, fake, _, _ = make_client(env)
    client._dispatch(add_command("1", type="Poses"))
    client._dispatch({"id": "A", "type": "activities:list"})
    listed = fake.last("activities:list")
    assert listed["status"] == "success" and [a["id"] for a in listed["activities"]] == ["1"]
    client._dispatch({"id": "C", "type": "activities:clear"})
    assert fake.last("activities:clear")["activities"] == []
    assert client.activities.list() == []
    client._dispatch({"id": "T", "type": "image:txt2img", "params": {}})
    assert fake.last("image:txt2img")["status"] == "error"


# ── module singleton ─────────────────────────────────────────────────────────────────────────────


class FakeLinkClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.activities = link.proto.Activities()
        FakeLinkClient.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, timeout=10.0):
        self.stopped = True

    def leave(self):
        self.stopped = True

    def status(self):
        return {"connected": True, "joined": True}


@pytest.fixture()
def singleton(env, monkeypatch, tmp_path):
    FakeLinkClient.instances.clear()
    monkeypatch.setattr(link, "LinkClient", FakeLinkClient)
    monkeypatch.setattr(link, "HAVE_SOCKETIO", True)
    monkeypatch.setattr(link, "_client", None)
    monkeypatch.setattr(link, "_registered", False)
    monkeypatch.setenv("CIVITAI_COMFY_INSTALL_ID_STORE", str(tmp_path / "install-id"))
    monkeypatch.setenv("CIVITAI_COMFY_OAUTH_STORE", str(tmp_path / "oauth.json"))
    monkeypatch.setattr(link, "PAIR_RETRY_DELAYS", (0.0,))
    yield
    link._client = None


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture()
def account(singleton, monkeypatch):
    """Stored OAuth login that already carries LinkConnect, plus a scripted link-service."""
    from civitai_comfy_nodes import oauth

    calls = []
    responses = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"method": "POST", "url": url, "json": json, "headers": headers})
        return responses.pop(0)

    def delete(url, params=None, headers=None, timeout=None):
        calls.append({"method": "DELETE", "url": url, "params": params, "headers": headers})
        return FakeResponse(200, {"success": True})

    monkeypatch.setattr(link.requests, "post", post)
    monkeypatch.setattr(link.requests, "delete", delete)
    monkeypatch.setattr(link.oauth, "interactive_login", lambda scope=None: pytest.fail("unexpected browser login"))
    oauth._save_tokens(
        {
            "access_token": "civitai_tok",
            "refresh_token": "r",
            "expires_at": time.time() + 3600,
            "scope": oauth.LINK_SCOPE,
        }
    )
    return calls, responses


def test_pair_via_account_exchanges_the_token_for_a_full_key(account):
    calls, responses = account
    responses.append(FakeResponse(200, {"id": 834, "key": "A" * 128, "name": "ComfyUI (box)"}))
    status = link.pair_via_oauth(name="ComfyUI (box)")
    sent = calls[0]
    assert sent["url"] == "https://link.civitai.com/api/link/self"
    assert sent["headers"] == {"Authorization": "Bearer civitai_tok"}
    assert sent["json"] == {"installId": config.install_id(), "name": "ComfyUI (box)"}  # no legacyKey: never paired
    stored = config.load_link_key()
    assert stored["key"] == "a" * 128 and stored["activated"] is True and stored["instance_id"] == 834
    assert FakeLinkClient.instances[-1].kwargs["key"] == "a" * 128 and FakeLinkClient.instances[-1].started
    assert status["paired"] and status["viaAccount"] and status["activated"]
    # Re-pairing reuses the persisted install id, which is what keeps the instance count flat.
    responses.append(FakeResponse(200, {"id": 834, "key": "b" * 128, "name": "ComfyUI (box)"}))
    link.pair_via_oauth()
    assert calls[1]["json"]["installId"] == sent["json"]["installId"]
    assert "legacyKey" not in calls[1]["json"]


def test_pair_via_account_adopts_a_code_pairing(account):
    calls, responses = account
    config.save_link_key("a1b2c3", activated=False)
    responses.append(FakeResponse(200, {"id": 12, "key": "c" * 128, "name": None}))
    link.pair_via_oauth()
    assert calls[0]["json"]["legacyKey"] == "a1b2c3"
    assert config.load_link_key()["instance_id"] == 12
    config.save_link_key("d" * 128, activated=True)  # upgraded by the socket, still no instance id
    responses.append(FakeResponse(200, {"id": 12, "key": "e" * 128, "name": None}))
    link.pair_via_oauth()
    assert calls[1]["json"]["legacyKey"] == "d" * 128


def test_pair_via_account_signs_in_when_the_login_lacks_link_scope(account, monkeypatch):
    from civitai_comfy_nodes import oauth

    calls, responses = account
    oauth._save_tokens({"access_token": "old", "refresh_token": "r", "expires_at": time.time() + 3600, "scope": 114689})
    requested = []

    def login(scope=None):
        requested.append(scope)
        oauth._save_tokens({"access_token": "fresh", "expires_at": time.time() + 3600, "scope": oauth.LINK_SCOPE})
        return "fresh"

    monkeypatch.setattr(link.oauth, "interactive_login", login)
    responses.append(FakeResponse(200, {"id": 1, "key": "f" * 128, "name": "x"}))
    link.pair_via_oauth()
    assert requested == [oauth.LINK_SCOPE]
    assert calls[0]["headers"]["Authorization"] == "Bearer fresh"
    # No stored login at all takes the same path.
    oauth.clear_tokens()
    responses.append(FakeResponse(200, {"id": 1, "key": "f" * 128, "name": "x"}))
    link.pair_via_oauth()
    assert requested == [oauth.LINK_SCOPE, oauth.LINK_SCOPE]


def test_pair_via_account_401_drops_the_login(account):
    from civitai_comfy_nodes import oauth

    calls, responses = account
    responses.append(FakeResponse(401, {"error": "unauthorized"}))
    with pytest.raises(ValueError, match="sign in afresh"):
        link.pair_via_oauth()
    assert oauth.get_valid_access_token() is None
    assert config.load_link_key() is None and FakeLinkClient.instances == []


def test_pair_via_account_reports_the_instance_limit_without_retrying(account):
    calls, responses = account
    responses.append(FakeResponse(400, {"error": "Instance limit reached"}))
    with pytest.raises(ValueError, match="instance limit"):
        link.pair_via_oauth()
    assert len(calls) == 1


def test_pair_via_account_retries_503_then_gives_up(account):
    calls, responses = account
    responses.extend([FakeResponse(503, {"error": "hub_unavailable"}), FakeResponse(503, {"error": "hub_unavailable"})])
    with pytest.raises(ValueError, match="temporarily unavailable"):
        link.pair_via_oauth()
    assert len(calls) == 2
    responses.extend(
        [FakeResponse(503, {"error": "hub_unavailable"}), FakeResponse(200, {"id": 2, "key": "9" * 128, "name": "x"})]
    )
    assert link.pair_via_oauth()["paired"]


def test_pair_after_login_never_opens_a_browser_and_never_raises(account, monkeypatch):
    from civitai_comfy_nodes import oauth

    calls, responses = account
    oauth._save_tokens({"access_token": "old", "expires_at": time.time() + 3600, "scope": 114689})
    link.pair_after_login()  # narrow login: nothing to do, no browser
    assert calls == [] and config.load_link_key() is None
    oauth._save_tokens({"access_token": "wide", "expires_at": time.time() + 3600, "scope": oauth.LINK_SCOPE})
    responses.append(FakeResponse(400, {"error": "Instance limit reached"}))
    link.pair_after_login()  # failure is logged, not raised
    assert config.load_link_key() is None
    responses.append(FakeResponse(200, {"id": 5, "key": "5" * 128, "name": "x"}))
    link.pair_after_login()
    assert config.load_link_key()["instance_id"] == 5
    link.pair_after_login()  # already paired: no call
    assert len(calls) == 2


def test_status_reports_the_auth_source(account):
    assert link.status()["auth"] == "oauth"


def test_pair_via_account_respects_disabled(account):
    config.save_pack_settings({"enableLink": False})
    with pytest.raises(ValueError, match="disabled"):
        link.pair_via_oauth()


def test_unpair_removes_an_account_instance_on_the_site_only(account):
    calls, responses = account
    config.save_link_key("a1b2c3", activated=False)
    link.unpair()
    assert calls == []  # a code pairing has no instance id to delete
    responses.append(FakeResponse(200, {"id": 7, "key": "1" * 128, "name": "x"}))
    link.pair_via_oauth()
    assert link.unpair()["paired"] is False
    assert calls[-1]["method"] == "DELETE" and calls[-1]["params"] == {"id": 7}
    assert calls[-1]["url"] == "https://link.civitai.com/api/link"


def test_pair_validates_saves_and_starts_a_client(singleton):
    with pytest.raises(ValueError):
        link.pair("xyz")
    status = link.pair(" A1B2C3 ")
    assert config.load_link_key()["key"] == "a1b2c3" and config.load_link_key()["activated"] is False
    assert FakeLinkClient.instances[-1].kwargs["key"] == "a1b2c3"
    assert FakeLinkClient.instances[-1].kwargs["url"] == "https://link.civitai.com"
    assert FakeLinkClient.instances[-1].started
    assert status["paired"] and status["connected"] and status["keyHint"] == "b2c3"
    link.pair("f" * 128)
    assert config.load_link_key()["activated"] is True
    assert FakeLinkClient.instances[0].stopped  # re-pairing replaces the running client
    assert link.unpair()["paired"] is False
    assert config.load_link_key() is None and FakeLinkClient.instances[-1].stopped


def test_module_cancel_requires_a_running_client(singleton):
    with pytest.raises(ValueError):
        link.cancel("x")
    link.pair("a1b2c3")
    FakeLinkClient.instances[-1].cancel = lambda command_id: command_id == "known"
    with pytest.raises(ValueError):
        link.cancel("unknown")
    assert link.cancel("known")["paired"] is True


def test_reconfigure_respects_disable_and_hosted(singleton, monkeypatch):
    config.save_link_key(KEY, activated=False)
    link.reconfigure()
    assert len(FakeLinkClient.instances) == 1
    config.save_pack_settings({"enableLink": False})
    link.reconfigure()
    assert FakeLinkClient.instances[0].stopped and link._client is None
    assert link.status()["disabledReason"] == "disabled in settings"
    with pytest.raises(ValueError):
        link.pair(KEY)
    config.save_pack_settings({})
    monkeypatch.setenv("CIVITAI_COMFY_SESSION_ID", "hosted")
    link.reconfigure()
    assert len(FakeLinkClient.instances) == 1
    assert link.status()["hosted"] is True


def test_register_is_idempotent_and_needs_a_stored_key(singleton):
    link.register(notify=lambda e, d: None)
    link.register()
    assert FakeLinkClient.instances == []  # no key stored -> nothing to start
    assert link.status()["paired"] is False and link.status()["available"] is True


def test_without_socketio_link_is_unavailable(singleton, monkeypatch):
    monkeypatch.setattr(link, "HAVE_SOCKETIO", False)
    config.save_link_key(KEY, activated=False)
    link.reconfigure()
    assert FakeLinkClient.instances == []
    assert "python-socketio" in link.status()["disabledReason"]
    with pytest.raises(ValueError):
        link.pair(KEY)


def test_link_module_imports_without_socketio(monkeypatch):
    monkeypatch.setitem(sys.modules, "socketio", None)
    reloaded = importlib.reload(link)
    try:
        assert reloaded.HAVE_SOCKETIO is False and reloaded.available() is False
    finally:
        monkeypatch.delitem(sys.modules, "socketio")
        importlib.reload(link)
