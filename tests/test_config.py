import pytest

from civitai_comfy_nodes import config, oauth
from civitai_comfy_nodes.errors import CivitaiAuthError


@pytest.fixture()
def no_creds(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_API_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_ORCHESTRATION_URL", raising=False)
    monkeypatch.setenv("CIVITAI_COMFY_API_KEY_STORE", str(tmp_path / "key"))
    monkeypatch.setenv("CIVITAI_COMFY_OAUTH_STORE", str(tmp_path / "oauth.json"))


def test_auth_state_none(no_creds):
    assert config.auth_state() == (None, None)


def test_auth_state_prefers_env(no_creds, monkeypatch):
    monkeypatch.setenv("CIVITAI_API_TOKEN", "envtok")
    assert config.auth_state() == ("envtok", "env")


def test_auth_state_stored_api_key(no_creds):
    oauth.save_api_key("keytok")
    assert config.auth_state() == ("keytok", "apikey")


def test_resolve_config_non_interactive_raises_when_no_creds(no_creds):
    with pytest.raises(CivitaiAuthError):
        config.resolve_config(interactive=False)


def test_resolve_config_uses_stored_api_key(no_creds):
    oauth.save_api_key("keytok")
    cfg = config.resolve_config(interactive=False)
    assert cfg.token == "keytok"
    assert cfg.base_url == config.DEFAULT_BASE_URL


def test_resolve_config_api_config_token_wins(no_creds):
    cfg = config.resolve_config({"api_token": "nodetok", "base_url": "http://local"}, interactive=False)
    assert cfg.token == "nodetok"
    assert cfg.base_url == "http://local"
