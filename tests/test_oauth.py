import json
import time

import pytest

from civitai_comfy_nodes import oauth


@pytest.fixture()
def token_store(tmp_path, monkeypatch):
    path = tmp_path / "comfy-oauth.json"
    monkeypatch.setenv("CIVITAI_COMFY_OAUTH_STORE", str(path))
    return path


def test_no_store_returns_none(token_store):
    assert oauth.get_valid_access_token() is None


def test_fresh_token_returned(token_store):
    token_store.write_text(
        json.dumps({"access_token": "civitai_abc", "refresh_token": "civitai_r", "expires_at": time.time() + 3600})
    )
    assert oauth.get_valid_access_token() == "civitai_abc"


def test_expired_token_refreshes(token_store, monkeypatch):
    token_store.write_text(
        json.dumps({"access_token": "civitai_old", "refresh_token": "civitai_r", "expires_at": time.time() - 10})
    )
    monkeypatch.setattr(oauth, "CLIENT_ID", "test-client")

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"access_token": "civitai_new", "refresh_token": "civitai_r2", "expires_in": 3600}

    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured.update(data)
        return FakeResponse()

    monkeypatch.setattr(oauth.requests, "post", fake_post)
    assert oauth.get_valid_access_token() == "civitai_new"
    assert captured["grant_type"] == "refresh_token"
    assert captured["refresh_token"] == "civitai_r"

    stored = json.loads(token_store.read_text())
    assert stored["access_token"] == "civitai_new"
    assert stored["refresh_token"] == "civitai_r2"


def test_failed_refresh_returns_none(token_store, monkeypatch):
    token_store.write_text(
        json.dumps({"access_token": "civitai_old", "refresh_token": "civitai_r", "expires_at": time.time() - 10})
    )
    monkeypatch.setattr(oauth, "CLIENT_ID", "test-client")

    class FakeResponse:
        status_code = 400
        text = "invalid_grant"

    monkeypatch.setattr(oauth.requests, "post", lambda *a, **k: FakeResponse())
    assert oauth.get_valid_access_token() is None


def test_store_permissions(token_store, monkeypatch):
    oauth._save_tokens({"access_token": "x"})
    assert oauth.token_store_path().stat().st_mode & 0o777 == 0o600


def test_login_without_client_id_raises(token_store, monkeypatch):
    monkeypatch.setattr(oauth, "CLIENT_ID", "")
    with pytest.raises(Exception, match="CIVITAI_API_TOKEN"):
        oauth.interactive_login()


def test_login_wait_is_interruptible(token_store, monkeypatch):
    # While waiting on the browser, the loop must poll ComfyUI's interrupt flag so Cancel works
    # instead of wedging the node for the full timeout.
    monkeypatch.setattr(oauth, "CLIENT_ID", "test-client")
    monkeypatch.setattr(oauth, "_candidate_ports", lambda: [18991])
    monkeypatch.setattr(oauth.webbrowser, "open", lambda url: None)

    class _InterruptError(Exception):
        pass

    calls = {"n": 0}

    def fake_check():
        calls["n"] += 1
        raise _InterruptError()

    monkeypatch.setattr(oauth.comfy_compat, "check_interrupted", fake_check)
    with pytest.raises(_InterruptError):
        oauth.interactive_login()
    assert calls["n"] == 1  # aborted on the first poll, not after the 5-minute timeout

    # the callback server was torn down (finally), so the port is free to rebind immediately
    import http.server

    srv = http.server.HTTPServer(("127.0.0.1", 18991), oauth._CallbackHandler)
    srv.server_close()


def test_bind_tries_next_candidate_when_first_is_taken(monkeypatch):
    import http.server

    blocker = http.server.HTTPServer(("127.0.0.1", 0), oauth._CallbackHandler)
    taken = blocker.server_address[1]
    probe = http.server.HTTPServer(("127.0.0.1", 0), oauth._CallbackHandler)
    free = probe.server_address[1]
    probe.server_close()
    monkeypatch.setattr(oauth, "_candidate_ports", lambda: [taken, free])
    server, port = oauth._bind_callback_server()
    try:
        assert port == free  # skipped the occupied port, bound the next candidate
    finally:
        server.server_close()
        blocker.server_close()


def test_candidate_ports_env_override(monkeypatch):
    monkeypatch.setenv("CIVITAI_OAUTH_REDIRECT_PORTS", "1234, 5678 9012")
    assert oauth._candidate_ports() == [1234, 5678, 9012]
