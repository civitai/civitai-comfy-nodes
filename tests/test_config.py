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
def settings_store(tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_ORCHESTRATION_URL", raising=False)
    monkeypatch.setenv("CIVITAI_COMFY_SETTINGS_STORE", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture()
def settings_with_key(settings_store, tmp_path, monkeypatch):
    monkeypatch.delenv("CIVITAI_API_TOKEN", raising=False)
    monkeypatch.setenv("CIVITAI_COMFY_API_KEY_STORE", str(tmp_path / "key"))
    monkeypatch.setenv("CIVITAI_COMFY_OAUTH_STORE", str(tmp_path / "oauth.json"))
    oauth.save_api_key("keytok")
    return tmp_path


def test_pack_settings_round_trip(settings_store):
    assert config.load_pack_settings() == {}
    config.save_pack_settings({"orchestratorUrl": "http://dev", "minVramGb": 24})
    assert config.load_pack_settings() == {"orchestratorUrl": "http://dev", "minVramGb": 24}
    assert config.stored_orchestrator_url() == "http://dev"
    assert config.stored_min_vram_gb() == 24


def test_stored_getter_defaults(settings_store):
    assert config.stored_orchestrator_url() is None
    assert config.stored_min_vram_gb() is None
    assert config.stored_mature_content() == "auto"
    assert config.stored_use_sage_attention() is True  # Sage Attention defaults on


def test_base_url_precedence_env_over_stored_over_default(settings_store, monkeypatch):
    assert config.base_url() == config.DEFAULT_BASE_URL
    config.save_pack_settings({"orchestratorUrl": "http://stored"})
    assert config.base_url() == "http://stored"  # stored beats default
    monkeypatch.setenv("CIVITAI_ORCHESTRATION_URL", "http://env/")
    assert config.base_url() == "http://env"  # env beats stored, trailing slash stripped


def test_resolve_config_orchestrator_precedence(settings_with_key, monkeypatch):
    config.save_pack_settings({"orchestratorUrl": "http://stored"})
    assert config.resolve_config(interactive=False).base_url == "http://stored"
    monkeypatch.setenv("CIVITAI_ORCHESTRATION_URL", "http://env")
    assert config.resolve_config(interactive=False).base_url == "http://env"
    cfg = config.resolve_config({"base_url": "http://node"}, interactive=False)
    assert cfg.base_url == "http://node"  # node input beats env


def test_resolve_config_mature_from_settings_and_node_override(settings_with_key):
    assert config.resolve_config(interactive=False).mature_content == "auto"
    config.save_pack_settings({"allowMatureContent": "true"})
    cfg = config.resolve_config(interactive=False)
    assert cfg.mature_content == "true"
    assert cfg.allow_mature_content is True
    # an explicit node value overrides the stored default
    assert config.resolve_config({"allow_mature_content": False}, interactive=False).mature_content == "false"


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
