from civitai_comfy_nodes.client import OrchestrationClient
from civitai_comfy_nodes.config import ClientConfig


class _Resp:
    status_code = 200

    @staticmethod
    def json():
        return {"next": "c2", "items": []}


def _client(monkeypatch):
    client = OrchestrationClient(ClientConfig(base_url="http://x", token="t"))
    captured = {}

    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, params=kwargs.get("params"))
        return _Resp()

    monkeypatch.setattr(client.session, "request", fake_request)
    return client, captured


def test_query_workflows_builds_params(monkeypatch):
    client, captured = _client(monkeypatch)
    out = client.query_workflows(cursor="c1", take=42)
    assert out == {"next": "c2", "items": []}
    assert captured["method"] == "GET"
    assert captured["url"].endswith("/v2/consumer/workflows")
    params = captured["params"]
    assert params["take"] == 42
    assert params["cursor"] == "c1"
    assert params["excludeFailed"] == "true"
    assert "hideMatureContent" not in params  # only sent when explicitly set


def test_query_workflows_defaults_and_optionals(monkeypatch):
    client, captured = _client(monkeypatch)
    client.query_workflows(exclude_failed=False, hide_mature=True)
    params = captured["params"]
    assert params["take"] == 60
    assert "cursor" not in params
    assert "excludeFailed" not in params
    assert params["hideMatureContent"] == "true"


def test_query_workflows_can_request_mature(monkeypatch):
    client, captured = _client(monkeypatch)
    client.query_workflows(hide_mature=False)
    assert captured["params"]["hideMatureContent"] == "false"
