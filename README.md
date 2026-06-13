# civitai-comfy-nodes

ComfyUI custom nodes for the [Civitai Orchestration API](https://developer.civitaic.com/orchestration/).
Run Civitai's cloud recipes — image/video/audio generation, upscaling, training, captioning,
moderation — as nodes inside any local ComfyUI graph. No local GPU or model downloads needed;
jobs run on Civitai's fleet and are billed in Buzz.

## Install

Clone (or unzip) into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/civitai/civitai-comfy-nodes.git
pip install -r civitai-comfy-nodes/requirements.txt   # just `requests`
```

## Authentication

Nodes resolve credentials in this order:

1. A connected **Civitai Auth** node (explicit token / base URL / mature-content / timeout overrides)
2. The `CIVITAI_API_TOKEN` environment variable ([create an API key](https://civitai.com/user/account))
3. A stored OAuth login (`~/.civitai/comfy-oauth.json`, auto-refreshed)
4. An interactive browser login (OAuth + PKCE) — requires `CIVITAI_OAUTH_CLIENT_ID` to be configured

Headless/remote ComfyUI installs should use the env var.

## Nodes

~160 nodes under the **Civitai** category, generated from the orchestration OpenAPI spec. Engines
with many variants are split into one node per ecosystem/operation (e.g. `Civitai/Image/sdcpp/zImage
Turbo · Create Image`) so each node shows only the inputs that variant actually uses:

- **Civitai/Image** — Text To Image, Image Gen (one node per engine: Flux2, OpenAI, Google, Seedream, …), Upscaler, Background Removal
- **Civitai/Video** — Video Gen (one node per engine: Wan, Kling, Vidu, Veo3, LTX, Sora, …), Upscaler, Interpolation, Enhancement
- **Civitai/Audio** — Text To Speech, Transcription, Audio Captioning, ACE Step Audio
- **Civitai/Text** — Chat Completion (plus a simple single-turn wrapper), Prompt Enhancement, Media Captioning
- **Civitai/Analysis** — Media Rating, WD Tagging, XGuard Moderation, Age Classification
- **Civitai/Training** — Training, Image Resource Training
- **Civitai/Misc** — Comfy, Custom Comfy (run a raw ComfyUI workflow on Civitai's workers), Echo

Every node returns its media outputs as native Comfy types (IMAGE/VIDEO/AUDIO) plus
`workflow_id` and `raw_json` for debugging and cost inspection. Models and LoRAs are
referenced by [AIR URNs](https://developer.civitaic.com/guide/air) (e.g.
`urn:air:sdxl:checkpoint:civitai:101055@128078`). Complex inputs (LoRA lists, ControlNets,
chat messages) are JSON text inputs in this version.

## Development

Nodes are **generated** — never edit `civitai_comfy_nodes/generated/` by hand. To change
node shapes, edit `codegen/overrides.json` (or the codegen itself) and regenerate:

```bash
python -m codegen.generate        # regenerate from spec/v2-consumers.json
pytest tests -q                   # unit tests (no ComfyUI or network needed)
CIVITAI_API_TOKEN=... pytest -m e2e -o addopts="" tests/test_e2e.py   # prod smoke test
```

To pick up orchestration API changes, see `scripts/sync-spec.sh`.
