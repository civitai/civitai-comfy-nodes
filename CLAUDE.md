# civitai-comfy-nodes

ComfyUI custom node pack exposing Civitai Orchestration consumer recipes as generated nodes.
The pack submits recipe steps via the generic workflows endpoint, polls to completion, and
converts blob outputs to native Comfy types.

## Architecture

- `civitai_comfy_nodes/generated/` — **AUTO-GENERATED, never hand-edit.** One module per
  category (image/video/audio/text/analysis/training/misc), ~57 declarative node classes.
- `civitai_comfy_nodes/base.py` — all runtime behavior: payload building from `FIELDS`,
  submit → poll loop (interrupt-aware, ProgressBar), output conversion per `OUTPUTS`.
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
- `GET /v2/consumer/workflows/{id}` `wait` is a boolean, not seconds — poll with
  client-side sleeps (the base class slices sleeps for Comfy interrupt handling).
- Cancel = `PUT /v2/consumer/workflows/{id}` with `{"status": "canceled"}`.
- Blob signed URLs expire; refresh via `POST /v2/consumer/blobs/{blobId}/refresh`.
- Discriminated inputs are recursive (`engine` → `version` → `provider` → `operation`).
  Codegen expands the top level into one node per variant and flattens nested levels
  into COMBO widgets (optional combos get a `""` omit choice so the server default wins).
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

E2E note: the echo test costs 1 Buzz; the textToImage test uses `whatif=true` and spends nothing.

## Testing in a real ComfyUI

Symlink this repo into `ComfyUI/custom_nodes/`, export `CIVITAI_API_TOKEN`, restart ComfyUI.
Nodes appear under the **Civitai** category. `example_workflows/` has drag-in graphs.
