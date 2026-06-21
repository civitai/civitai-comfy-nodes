# civitai-comfy-nodes

> 🔹 **Now in early preview — installable from the [Comfy Registry](https://registry.comfy.org/nodes/civitai-comfy-nodes).**
> Still under active development: nodes, APIs, and behavior may change without
> notice. Use at your own risk.

ComfyUI custom nodes for the [Civitai Orchestration API](https://developer.civitai.com/orchestration/).
Run Civitai's cloud recipes — image/video/audio generation, upscaling, training, captioning,
moderation — as nodes inside any local ComfyUI graph. No local GPU or model downloads needed;
jobs run on Civitai's fleet and are billed in Buzz.

## Install

### From the Comfy Registry (recommended)

The pack is published to the [Comfy Registry](https://registry.comfy.org/nodes/civitai-comfy-nodes),
so [ComfyUI Manager](https://docs.comfy.org/manager/overview) can install and update it
for you:

- **In ComfyUI Manager:** open **Manager → Custom Nodes Manager**, search for **Civitai Comfy Nodes**
  (publisher `civitai`), and click **Install**, then restart ComfyUI.
- **With [comfy-cli](https://docs.comfy.org/comfy-cli/getting-started):**

  ```bash
  comfy node registry-install civitai-comfy-nodes
  ```

See the [Comfy Registry docs](https://docs.comfy.org/registry/overview) for details.

### From source

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

- **Civitai/Image** — Image Gen (one node per engine: Flux2, OpenAI, Google, Seedream, …), Upscaler, Background Removal
- **Civitai/Video** — Video Gen (one node per engine: Wan, Kling, Vidu, Veo3, LTX, Sora, …), Upscaler, Interpolation, Enhancement
- **Civitai/Audio** — Text To Speech, Transcription, Audio Captioning, ACE Step Audio
- **Civitai/Text** — Chat Completion (plus a simple single-turn wrapper), Prompt Enhancement, Media Captioning
- **Civitai/Analysis** — Media Rating, WD Tagging, XGuard Moderation
- **Civitai/Training** — Training, Image Resource Training
- **Civitai/Misc** — Poly Gen (3D mesh generation)
- **Civitai/Loaders** — Model Selector, LoRA Selector, Embedding Selector, ControlNet (see below)

Every node returns its media outputs as native Comfy types (IMAGE/VIDEO/AUDIO) plus
`workflow_id` and `raw_json` for debugging and cost inspection. Models and LoRAs are
referenced by [AIR URNs](https://developer.civitai.com/guide/air) (e.g.
`urn:air:sdxl:checkpoint:civitai:101055@128078`).

### Models, LoRAs, ControlNets & embeddings

Recipe nodes expose their model references as typed **sockets**, not text widgets — `model`/`vae`
are `CIVITAI_AIR`, `loras` is `CIVITAI_LORAS`, `embeddings` is `CIVITAI_EMBEDDINGS`. You fill them by
wiring a **Civitai/Loaders** selector node; each selector has a **🔍 Browse Civitai** button (a
searchable card grid of generation-capable models via a same-origin proxy to
`civitai.com/api/v1/models`) and shows an on-node preview (thumbnail + name) of its current
resource. Recipe nodes themselves have no Browse button — change a model by wiring a selector.

- **Civitai Model Selector** — pick a model once; it serves two purposes through its two outputs:
  1. **Choose the model a Civitai recipe node runs on.** Wire its `air` output into a recipe node's
     `model` / `vae` socket. The job runs on Civitai's fleet, so nothing is downloaded locally.
  2. **Auto-download a model for a local loader.** Wire its `path` output into any *standard* loader's
     file widget — e.g. drop it in front of a **Load LoRA** (`lora_name`) or **Load Checkpoint**
     (`ckpt_name`) node and it downloads the model into the matching ComfyUI folder for you, no manual
     file management. No need to replace the loader — just feed it.

  The download happens **only when `path` is connected**, so use (1) stays cloud-only and never pulls
  files down.
- **Civitai LoRA Selector** — holds **multiple LoRAs in one node**: each row has an enable toggle,
  the model (pick/replace via Browse Civitai), a strength, and a **keywords** field; **＋ Add LoRA**
  appends another. Wire its `loras` output into a recipe node's `loras` / `additional_networks` input
  (chain another selector via the `loras` input to combine stacks). Wire MODEL + CLIP to also
  download & apply the enabled LoRAs locally. The **keywords** field is auto-filled from the LoRA's
  trained words purely as a reminder — a LoRA applies by its AIR + strength, so the generator
  **ignores** these words; to actually invoke the trained concept, paste them into your recipe node's
  **prompt**.
- **Civitai Embedding Selector** — pick textual-inversion embeddings; chain (`embeddings` →
  `embeddings`) and wire into a recipe node's `embeddings` input. Embeddings **apply automatically**
  (the sdcpp pipeline prepends each one to the positive prompt), so no trigger word is needed. To use
  a **negative** embedding, reference it by its model name in the recipe node's **negative prompt** —
  that places it on the negative side instead. (Only sdcpp ecosystems — sd1/sdxl — expose an
  `embeddings` input.)
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

### CustomComfy offload

The **Run on Civitai** action submits the current graph as a `customComfy` workflow. Local model
widgets are rewritten to AIRs when the model can be resolved by embedded metadata hash or computed
hash, and installed custom node packs are advertised when a versioned nodepack AIR can be inferred.

Use **Civitai/Offload/Civitai Offload Start** and **Civitai Offload End** to delimit a cloud region.
Place `Start` to the left of the nodes to offload, put the normal Comfy workflow between the markers,
include the user output node such as `SaveImage` inside that region, and place `End` to the right.
Nodes after `End` are not included in the submitted customComfy job.

The markers are selection hints for **Run on Civitai**. When the cloud job returns an output asset,
the node pack imports that asset into local Comfy and queues the downstream nodes after `End` as a
local continuation.

## Development

Nodes are **generated** — never edit `civitai_comfy_nodes/generated/` by hand. To change
node shapes, edit `codegen/overrides.json` (or the codegen itself) and regenerate:

```bash
python -m codegen.generate        # regenerate from spec/v2-consumers.json
pytest tests -q                   # unit tests (no ComfyUI or network needed)
CIVITAI_API_TOKEN=... pytest -m e2e -o addopts="" tests/test_e2e.py   # prod smoke test
```

To pick up orchestration API changes, see `scripts/sync-spec.sh`.
