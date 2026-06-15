# civitai-comfy-nodes

ComfyUI custom node pack exposing Civitai Orchestration consumer recipes as generated nodes.
The pack submits recipe steps via the generic workflows endpoint, polls to completion, and
converts blob outputs to native Comfy types.

## Architecture

- `civitai_comfy_nodes/generated/` — **AUTO-GENERATED, never hand-edit.** One module per
  category (image/video/audio/text/analysis/training/misc), ~160 declarative node classes.
- `civitai_comfy_nodes/nodes_manual.py` — hand-written nodes: `CivitaiAuth`, `CivitaiChatSimple`,
  and the **Civitai/Loaders** helpers (`CivitaiLoraLoader`, `CivitaiControlNet`,
  `CivitaiCheckpointLoader`). The loaders output typed sockets (`CIVITAI_LORAS` /
  `CIVITAI_CONTROLNETS`); codegen emits the `loras`/`additionalNetworks`/`controlNets` recipe
  fields as those socket types (see `ir.classify_input_field` network rules) instead of JSON
  text, and `base._build_payload` serializes them per field shape (array vs AIR-keyed map).
- `civitai_comfy_nodes/base.py` — all runtime behavior: payload building from `FIELDS`,
  submit → poll loop (interrupt-aware, ProgressBar), output conversion per `OUTPUTS`. `run()`
  returns `{"ui": {"civitai_status": [...]}, "result": (...)}`; the `ui` payload (workflow id +
  per-currency Buzz cost) is rendered on the node by `web/civitai-status.js`.
- `civitai_comfy_nodes/client.py` — `OrchestrationClient`: workflows submit/get/cancel,
  blob download with expired-URL refresh, presigned uploads for URL-only media fields.
- `civitai_comfy_nodes/config.py` + `oauth.py` — auth chain: CivitaiAuth node input >
  `CIVITAI_API_TOKEN` env > stored OAuth tokens (auto-refresh) > interactive PKCE login.
- `civitai_comfy_nodes/comfy_compat.py` — guarded comfy imports; the package must always
  import (and pass tests) without ComfyUI installed.
- `codegen/` — `ir.py` (spec parsing, discriminator expansion/flattening), `emit.py`
  (source emission), `overrides.json` (per-recipe corrections), `generate.py` (pipeline).

## Load-bearing API facts (verified against spec + orchestration source)

- **Never use `POST /v2/consumer/recipes/{name}` for execution** — those endpoints are
  synchronous, return output-only, and cancel the workflow when the connection drops
  (~100s gateway timeout in prod kills long jobs). Always submit
  `POST /v2/consumer/workflows?wait=5` with `{"steps":[{"$type":"<recipe>","input":{...}}]}`.
- `GET /v2/consumer/workflows/{id}` `wait` is seconds — it long-polls (returns when the
  workflow finishes or after `wait`s with a 202). The base class runs the long-poll in a
  daemon thread and polls the Comfy interrupt flag every 0.5s, so Cancel stays responsive
  while the request blocks; a min-interval throttle avoids tight-looping if the server
  returns early without honoring `wait`.
- Cancel = `PUT /v2/consumer/workflows/{id}` with `{"status": "canceled"}`.
- Blob signed URLs expire; refresh via `POST /v2/consumer/blobs/{blobId}/refresh`.
- Discriminated inputs are recursive (`engine` → `model`/`ecosystem`/`version` → `operation`).
  Codegen fully expands every discriminator into a separate node, then **collapses a level
  back into a dropdown only when all its sibling subtrees are structurally identical**
  (`generate.py:expand_collapse` + `_subtree_signature`). So each node shows exactly its
  variant's fields (no irrelevant inputs, no image/images overlap), while true duplicates
  like openai gpt-image-1/1.5/2 — where they ARE identical — stay one node with a dropdown.
  `engine` never collapses. When a collapsed group splits a fixed path (e.g. fal/qwen2
  create-ops vs edit-ops) the node is disambiguated by the group's lead operation.
- OAuth: PKCE at `civitai.com/api/auth/oauth/*`, `scope` is a decimal bitmask
  (114689 = UserRead|AIServicesRead|AIServicesWrite|BuzzRead), access 1h / refresh 30d.
  Interactive login needs `CIVITAI_OAUTH_CLIENT_ID` (registered app with
  `http://localhost:18188/civitai/callback` as redirect URI).

## Cross-Repo Data Flow (spec sync)

1. Rebuild the orchestration API to regenerate the spec:
   ```bash
   dotnet build ../../civitai-orchestration/repo/src/Civitai.Orchestration.Api
   ```
2. Sync + regenerate + test: `scripts/sync-spec.sh`
3. Review the generated diff (the audit table printed by codegen flags media-field
   detection changes), adjust `codegen/overrides.json` if a field classified wrong, commit.

New recipes fail generation loudly until assigned: a module in `MODULES`
(codegen/generate.py) or a `_skip` entry in overrides.json. Skipped recipe groups:
locally-trivial media utils, blob plumbing, model scans, preprocessImage, and the four
recipes without WorkflowStep mappings.

## Commands

```bash
.venv/bin/python -m codegen.generate     # regenerate nodes (also ruff-formats output)
.venv/bin/python -m pytest tests -q      # unit tests, no ComfyUI/network required
CIVITAI_API_TOKEN=... .venv/bin/python -m pytest -m e2e -o addopts="" tests/test_e2e.py
UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/test_emit_golden.py  # after intended emit changes
```

E2E note: the textToImage test uses `whatif=true` and spends nothing.

## Testing in a real ComfyUI

Symlink this repo into `ComfyUI/custom_nodes/`, export `CIVITAI_API_TOKEN`, restart ComfyUI.
Nodes appear under the **Civitai** category. `example_workflows/` has drag-in graphs.
