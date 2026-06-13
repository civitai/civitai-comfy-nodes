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
