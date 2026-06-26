# Changelog

All notable changes to this pack are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The section matching the `pyproject.toml` version is published to the Comfy Registry
"Updates" tab and to the matching GitHub Release on each version bump — see
[`.github/workflows/publish.yml`](.github/workflows/publish.yml). Add a new `## [x.y.z]`
section at the top before bumping the version.

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
