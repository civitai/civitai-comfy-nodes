"""End-to-end tests against the production orchestration API.

Run explicitly: CIVITAI_API_TOKEN=... pytest -m e2e tests/test_e2e.py
The textToImage check uses whatif and spends nothing.
"""

import os

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.environ.get("CIVITAI_API_TOKEN"), reason="CIVITAI_API_TOKEN not set"),
]


def test_text_to_image_whatif_costs_without_spending():
    from civitai_comfy_nodes.client import OrchestrationClient
    from civitai_comfy_nodes.config import resolve_config
    from civitai_comfy_nodes.generated.image import CivitaiTextToImage

    node = CivitaiTextToImage()
    client = OrchestrationClient(resolve_config())
    payload = node._build_payload(client, {"prompt": "a lighthouse at dusk", "cfg_scale": 7.5, "seed": 42})
    payload.update(node.DISCRIMINATOR)
    workflow = client.submit_workflow(node.STEP_TYPE, payload, wait=0, whatif=True)
    assert workflow.get("cost", {}).get("total", 0) > 0
    assert workflow.get("status") in ("unassigned", "preparing", "scheduled")
