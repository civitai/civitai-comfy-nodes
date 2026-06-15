# civitai-comfy-nodes

> ⚠️ **Under active development — not yet available in preview.** This repo is a
> work in progress: nodes, APIs, and behavior may change without notice, and it
> is not yet released for public preview. Use at your own risk.

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
3. A stored API key (`~/.civitai/comfy-api-key`) or OAuth login (`~/.civitai/comfy-oauth.json`,
   auto-refreshed) — both can be set from the **Civitai sidebar's connect panel** (no env var needed)
4. An interactive browser login (OAuth + PKCE) — requires `CIVITAI_OAUTH_CLIENT_ID` to be configured

Headless/remote ComfyUI installs should use the env var.

## Nodes

~160 nodes under the **Civitai** category, generated from the orchestration OpenAPI spec. The menu
is organized **ecosystem-first** — `Civitai/<media>/<ecosystem>[/<engine>]/…` — with the engine
(sdcpp/comfy) shown as a sub-level only when an ecosystem is reachable through more than one engine
(e.g. `Civitai/Image/zImage › zImage / turbo / createImage`, `Civitai/Image/anima/sdcpp`). Each
discriminator variant is its own node so it shows only the inputs that variant actually uses:

- **Civitai/Image** — Text To Image, Image Gen (one node per engine: Flux2, OpenAI, Google, Seedream, …), Upscaler, Background Removal
- **Civitai/Video** — Video Gen (one node per engine: Wan, Kling, Vidu, Veo3, LTX, Sora, …), Upscaler, Interpolation, Enhancement
- **Civitai/Audio** — Text To Speech, Transcription, Audio Captioning, ACE Step Audio
- **Civitai/Text** — Chat Completion (plus a simple single-turn wrapper), Prompt Enhancement, Media Captioning
- **Civitai/Analysis** — Media Rating, WD Tagging, XGuard Moderation
- **Civitai/Training** — Training, Image Resource Training
- **Civitai/Misc** — Poly Gen (3D mesh generation)
- **Civitai/Loaders** — Model Selector, LoRA Selector, Embedding Selector, ControlNet (see below)

Every node returns its media outputs as native Comfy types (IMAGE/VIDEO/AUDIO) plus
`workflow_id` and `raw_json` for debugging and cost inspection. Models and LoRAs are
referenced by [AIR URNs](https://developer.civitaic.com/guide/air) (e.g.
`urn:air:sdxl:checkpoint:civitai:101055@128078`).

### Models, LoRAs, ControlNets & embeddings

Recipe nodes expose their model references as typed **sockets**, not text widgets — `model`/`vae`
are `CIVITAI_AIR`, `loras` is `CIVITAI_LORAS`, `embeddings` is `CIVITAI_EMBEDDINGS`. You fill them by
wiring a **Civitai/Loaders** selector node; each selector has a **🔍 Browse Civitai** button (a
searchable card grid of generation-capable models via a same-origin proxy to
`civitai.com/api/v1/models`) and shows an on-node preview (thumbnail + name) of its current
resource. Recipe nodes themselves have no Browse button — change a model by wiring a selector.

- **Civitai Model Selector** — pick a model; it outputs the `air` (wire into a recipe node's `model`
  / `vae` socket) plus a `path` you wire into a *standard* loader's file widget (e.g. Load
  Checkpoint's `ckpt_name`). The model downloads into the matching ComfyUI folder **only when `path`
  is connected**, so AIR/cloud-only use never downloads. Drop it in front of any loader — no need to
  replace it.
- **Civitai LoRA Selector** — set an AIR + strength (+ optional trigger word) and wire its `loras`
  output into a recipe node's `loras` / `additional_networks` input. Chain several (`loras` →
  `loras`) to stack multiple LoRAs.
- **Civitai Embedding Selector** — pick textual-inversion embeddings; chain (`embeddings` →
  `embeddings`) and wire into a recipe node's `embeddings` input.
- **Civitai ControlNet** — pick a preprocessor, weight, step range, optional control image; chain
  and wire into a recipe node's `control_nets` input.

`chat messages` and other freeform structures remain JSON text inputs.

## Browse your generations

The **Civitai** sidebar tab (the logo icon in the left rail) lists your Civitai generation history —
every workflow you've run, across all media types and sources (web / API / ComfyUI) — pulled from
the orchestrator and scoped to your account. Filter by media kind, scroll to paginate, and click any
result for a lightbox. Pull a result back into the graph three ways: **add to canvas** (creates the
matching loader node — image→`LoadImage`, video/audio/3D→their loaders — wired to the imported file),
**fill** a selected loader node, or **drag** a thumbnail onto the canvas. If no credentials are
configured, the tab shows a connect panel (OAuth sign-in or paste an API key).

## Development

Nodes are **generated** — never edit `civitai_comfy_nodes/generated/` by hand. To change
node shapes, edit `codegen/overrides.json` (or the codegen itself) and regenerate:

```bash
python -m codegen.generate        # regenerate from spec/v2-consumers.json
pytest tests -q                   # unit tests (no ComfyUI or network needed)
CIVITAI_API_TOKEN=... pytest -m e2e -o addopts="" tests/test_e2e.py   # prod smoke test
```

To pick up orchestration API changes, see `scripts/sync-spec.sh`.
