import pytest

from civitai_comfy_nodes import buzz
from civitai_comfy_nodes.errors import CivitaiNodeError


def test_parse_buzz_accounts_reads_the_trpc_envelope():
    payload = {"result": {"data": {"json": {"blue": 7557, "green": 841299, "yellow": 585990}}}}
    assert buzz.parse_buzz_accounts(payload) == {"blue": 7557, "green": 841299, "yellow": 585990}


def test_parse_buzz_accounts_defaults_missing_wallets_to_zero():
    assert buzz.parse_buzz_accounts({"result": {"data": {"json": {"blue": 1}}}}) == {"blue": 1, "green": 0, "yellow": 0}


def test_parse_buzz_accounts_rejects_contract_changes():
    with pytest.raises(CivitaiNodeError):
        buzz.parse_buzz_accounts({"result": {"data": {"json": {"blue": "lots"}}}})
    with pytest.raises(CivitaiNodeError):
        buzz.parse_buzz_accounts({"result": {"data": {"json": [1, 2]}}})


def test_fetch_buzz_accounts_sends_bearer_and_maps_auth_failures(monkeypatch):
    captured = {}

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"data": {"json": {"blue": 5, "green": 6, "yellow": 7}}}}

    def fake_get(url, headers=None, timeout=None):
        captured.update(url=url, headers=headers)
        return Resp()

    monkeypatch.setattr(buzz.requests, "get", fake_get)
    assert buzz.fetch_buzz_accounts("tok") == {"blue": 5, "green": 6, "yellow": 7}
    assert captured["url"] == buzz.CIVITAI_BUZZ_ACCOUNTS_URL
    assert captured["headers"]["Authorization"] == "Bearer tok"

    class Denied(Resp):
        status_code = 401

    monkeypatch.setattr(buzz.requests, "get", lambda *a, **k: Denied())
    with pytest.raises(CivitaiNodeError):
        buzz.fetch_buzz_accounts("tok")
