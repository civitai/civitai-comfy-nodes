from civitai_comfy_nodes import base
from civitai_comfy_nodes.config import ClientConfig


class _FakeClient:
    def __init__(self, config):
        self.calls = 0

    def submit_workflow(self, step_type, payload, wait=5):
        return {
            "id": "wf-test",
            "status": "scheduled",
            "steps": [{"jobs": [{"queuePosition": {"precedingJobs": 3, "support": "available"}}]}],
        }

    def get_workflow(self, workflow_id, wait=0):
        self.calls += 1
        if self.calls == 1:
            return {"id": workflow_id, "status": "processing", "steps": [{"jobs": [{"estimatedProgressRate": 0.5}]}]}
        return {
            "id": workflow_id,
            "status": "succeeded",
            "steps": [{"output": {}}],
            "cost": {"total": 7},
            "transactions": {"list": [{"type": "debit", "amount": 7, "accountType": "blue"}]},
        }


class _Node(base.CivitaiRecipeNodeBase):
    RECIPE = "echo"
    STEP_TYPE = "echo"
    FIELDS = {}
    OUTPUTS = ()


def test_run_logs_workflow_id_and_status_transitions(monkeypatch, caplog):
    cfg = ClientConfig("http://x", "t", timeout_minutes=30)
    monkeypatch.setattr(base, "resolve_config", lambda api_config=None: cfg)
    monkeypatch.setattr(base, "OrchestrationClient", _FakeClient)
    monkeypatch.setattr(base.CivitaiRecipeNodeBase, "_interruptible_sleep", staticmethod(lambda seconds: None))

    with caplog.at_level("INFO", logger="civitai_comfy_nodes"):
        ret = _Node().run()

    assert ret["ui"]["civitai_status"][0] == {"workflow_id": "wf-test", "cost": "7 Blue Buzz"}
    result = ret["result"]
    assert result[-2] == "wf-test"  # workflow_id output
    text = "\n".join(caplog.messages)
    assert "submitted workflow wf-test" in text
    assert "wf-test: scheduled, 10%, 3 jobs ahead" in text
    assert "wf-test: processing" in text
    assert "succeeded — 7 Blue Buzz" in text
