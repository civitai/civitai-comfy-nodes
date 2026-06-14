import time

import pytest

from civitai_comfy_nodes import base
from civitai_comfy_nodes.base import CivitaiRecipeNodeBase


def test_interruptible_get_passes_wait_and_returns():
    class Client:
        def __init__(self):
            self.waits = []

        def get_workflow(self, workflow_id, wait=0):
            self.waits.append(wait)
            return {"id": workflow_id, "status": "succeeded"}

    client = Client()
    result = CivitaiRecipeNodeBase._interruptible_get(client, "wf", wait=10)
    assert result["status"] == "succeeded"
    assert client.waits == [10]  # long-poll seconds forwarded to the GET


def test_interruptible_get_honors_cancel_while_request_blocks(monkeypatch):
    class SlowClient:
        def get_workflow(self, workflow_id, wait=0):
            time.sleep(2)  # simulate a long-poll holding server-side
            return {"status": "succeeded"}

    class _CancelError(Exception):
        pass

    ticks = {"n": 0}

    def fake_check():
        ticks["n"] += 1
        if ticks["n"] >= 2:  # "user pressed Cancel" on the 2nd 0.5s tick
            raise _CancelError()

    monkeypatch.setattr(base.comfy_compat, "check_interrupted", fake_check)
    started = time.time()
    with pytest.raises(_CancelError):
        CivitaiRecipeNodeBase._interruptible_get(SlowClient(), "wf", wait=5)
    assert time.time() - started < 1.5  # aborted well before the 2s request would finish
