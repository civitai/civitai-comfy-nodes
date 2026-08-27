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


@pytest.fixture()
def settings_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_SETTINGS_STORE", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture()
def link_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVITAI_COMFY_LINK_STORE", str(tmp_path / "link.json"))
    monkeypatch.delenv("CIVITAI_LINK_URL", raising=False)
    monkeypatch.delenv("CIVITAI_COMFY_SESSION_ID", raising=False)
    return tmp_path / "link.json"


def test_pack_settings_store_round_trip(settings_store):
    assert config.load_pack_settings() == {}
    config.save_pack_settings({"enableLink": False})
    assert config.load_pack_settings() == {"enableLink": False}
    config.settings_store_path().write_text("{nope")
    assert config.load_pack_settings() == {}


def test_link_key_store_round_trip_is_private(link_store):
    import os
    import stat

    assert config.load_link_key() is None
    config.save_link_key("abc123", activated=False)
    stored = config.load_link_key()
    assert stored["key"] == "abc123" and stored["activated"] is False and stored["paired_at"]
    assert stat.S_IMODE(os.stat(link_store).st_mode) == 0o600
    config.save_link_key("f" * 128, activated=True)
    assert config.load_link_key()["activated"] is True
    config.clear_link_key()
    config.clear_link_key()  # idempotent
    assert config.load_link_key() is None
    link_store.write_text("{not json")
    assert config.load_link_key() is None


def test_link_url_precedence_and_defaults(settings_store, link_store, monkeypatch):
    assert config.link_url() == config.DEFAULT_LINK_URL
    assert config.stored_enable_link() is True
    config.save_pack_settings({"linkUrl": "http://stored/", "enableLink": False})
    assert config.link_url() == "http://stored"
    assert config.stored_enable_link() is False
    monkeypatch.setenv("CIVITAI_LINK_URL", "http://env/")
    assert config.link_url() == "http://env"


def test_is_hosted_session_follows_the_pinned_session_env(link_store, monkeypatch):
    assert config.is_hosted_session() is False
    monkeypatch.setenv("CIVITAI_COMFY_SESSION_ID", " cloud ")
    assert config.is_hosted_session() is True
