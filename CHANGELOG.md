# Changelog

All notable changes to this pack are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The section matching the `pyproject.toml` version is published to the Comfy Registry
"Updates" tab and to the matching GitHub Release on each version bump — see
[`.github/workflows/publish.yml`](.github/workflows/publish.yml). Add a new `## [x.y.z]`
section at the top before bumping the version.

## [Unreleased]

### Added
- **Run on Civitai**: submit the current graph (or the region between the Offload Start/End
  markers) as a `customComfy` workflow, with the remote run's progress, previews, logs and live
  Buzz cost replayed onto the local canvas and the outputs imported back for the local tail.
- Toolbar **Run on Civitai** button next to ComfyUI's Run button, plus a **Pay with** wallet
  picker (Blue / Green / Yellow Buzz with live balances). The chosen wallet is stored in the pack
  settings and pins offload runs to that wallet.

## [0.5.0] - 2026-08-27

### Added
- **Civitai Link client**: pair the pack with civitai.com from the Civitai sidebar tab, then the
  site's "download via Civitai Link" button downloads models into the matching ComfyUI model
  folder (SHA256-verified, progress reported back, model lists refreshed). The site shows which
  models are already installed; removing one from the site deletes it locally. New dependency
  `python-socketio[client]`; toggle under Settings › Civitai › Civitai Link.

### Changed
- `download_model` streams through a shared core that hashes while writing and supports
  cancellation outside a running prompt.
- Local-model resolution (`model_resolve`, shared hash cache) keeps the canonical whole-file SHA256 from the Civitai by-hash lookup,
  so safetensors resolved via their embedded hash no longer need a full read.

## [0.4.0] - 2026-07-06

### Added
- **Import from Civitai** button in the Model Library sidebar (local ComfyUI): opens the
  Browse Civitai picker pre-filled with your library search; picking a model downloads its
  primary file — plus required CLIP/VAE component files — into the matching model folders
  and refreshes the library. Hidden in hosted comfy-cloud sessions, where models resolve
  via session pins instead.
- **Import from URL** in the Browse Civitai picker: paste a civitai.com model/version page
  URL, a download URL, or an AIR to import models search doesn't surface (e.g. base models
  without a registered ecosystem).
- Gallery: outputs that aren't image/video/audio/3D (extensionless customComfy assets,
  nodepack snapshot layers) now show as plain files with an **Other** filter and an
  Open ↗ action, instead of rendering as broken images.

### Changed
- Gallery infers the media kind from the output filename in the blob id when a customComfy
  output declares no type, so audio/video outputs land under the right filter instead of
  all showing as images.

### Fixed
- Generic "Model"/"Pruned Model" file types no longer force downloads into `checkpoints/` —
  the model's AIR type decides the folder, so e.g. a LoRA whose primary file is typed
  "Model" lands in `loras/`.

## [0.3.1] - 2026-06-29

### Fixed
- Browse Civitai picker now lists all models, not only those flagged for onsite generation.
  The `supportsGeneration` filter was excluding eligible resources.

## [0.3.0] - 2026-06-26

### Added
- **Model Selector** exposes a version's VAE and CLIP files as extra outputs (`vae`, `clip`,
  `clip 2`, `clip 3`); the picker adapts outputs to the selected model. Covers multi-component
  models like Z-Image-Turbo and WAN.
- **Hosted credentials** read per-prompt from `extra_data.civitai` (no cross-user leakage in
  pooled containers, no browser-login fallback).
- New generated nodes: **Qwen Image Bench**, **boogu**/**krea2** image variants, **HappyHorse
  v1.1** video, **AI-Toolkit Anima** training.

### Changed
- Files download into the folder for each file's Civitai type (e.g. a Checkpoint whose primary
  is a Diffusion Model lands in `diffusion_models/`).
- Component outputs block obviously-wrong connections by target name (`vae` won't wire to a
  clip/unet input).

### Fixed
- Model Selector declares the primary under the plain version AIR, so workers reuse a held
  checkpoint instead of re-downloading a file-pinned copy.
- `resources_json` is now optional, fixing "Required input is missing" on local/older graphs.

## [0.2.0] - 2026-06-17

### Added
- **PolyGen** nodes now return rigged and animated model outputs: `rigged_model`,
  `rigged_fbx_model`, `animated_model`, `animated_fbx_model`, and `basic_animations`.
  Note: these outputs are inserted ahead of `workflow_id`/`raw_json`, so existing PolyGen
  graphs may need rewiring.

### Changed
- Registry/Manager listing renamed from "Civitai Orchestration" to **"Civitai Comfy Nodes"**,
  matching the repo and `comfy node install civitai-comfy-nodes`.
- **Media Captioning**: `temperature` and `max_tokens` are now optional inputs
  (defaults 0.5 and 300).
- **AI-Toolkit training**: `epochs` maximum raised from 20 to 200 across all variants.
- **SD1 image generation**: `clip_skip` default changed from -1 to 2 (Create Image,
  Create Variant).

## [0.1.0] - 2026-06-16

### Added
- Initial early-preview release: ~160 generated nodes spanning the Civitai Orchestration
  API (image / video / audio / text / analysis / training / 3D), the **Civitai/Loaders**
  selector nodes (model / LoRA / embedding / ControlNet) with a Browse Civitai picker, the
  Civitai generation-history sidebar, and OAuth + API-key authentication.
