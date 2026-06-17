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


@pytest.fixture()
def session_store(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_COMFY_SESSION_ID", raising=False)
    monkeypatch.setenv("CIVITAI_COMFY_SESSION_STORE", str(tmp_path / "session-id"))


def test_session_id_env_overrides_file(session_store, monkeypatch):
    # comfy-cloud pins the session id; it wins over (and never writes) the local file.
    monkeypatch.setenv("CIVITAI_COMFY_SESSION_ID", "  cloud-session-42  ")
    assert config.resolve_session_id() == "cloud-session-42"
    assert not config.session_id_store_path().exists()


def test_session_id_persists_and_is_stable(session_store):
    minted = config.resolve_session_id()
    assert minted
    assert config.session_id_store_path().read_text().strip() == minted
    assert config.resolve_session_id() == minted  # reused across calls / restarts


def test_session_tag_and_submit_tags(session_store, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_SESSION_ID", "abc")
    assert config.session_tag() == f"{config.SOURCE_TAG}:session:abc"
    assert config.submit_tags() == [config.SOURCE_TAG, f"{config.SOURCE_TAG}:session:abc"]
