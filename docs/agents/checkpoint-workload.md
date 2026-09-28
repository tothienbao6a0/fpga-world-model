# Agent work: checkpoint-backed world-model pivot

## What ran

Codex revisited the original pasted context and the planning-focused repo after Bao clarified the target is a serious world-model hardware project. It checked the official JEPA-WMs source and model card, downloaded the 212 MB Push-T checkpoint, pinned and hashed it, loaded its predictor and proprioception weights strictly into the upstream implementation, and ran two- and four-frame CPU inference measurements. `make check`, `make model-check`, and the archived prototype's own `make check` all passed locally.

## What it changed

- `worldmodel/` now defines the pinned model workload, checkpoint verification, executable predictor benchmark, INT8 reference/verification, and tests.
- `rtl/qkv_tile.sv` adds a synthesizable QKV projection tile checked against quantized checkpoint weights.
- `results/` records the measured CPU runs with hashes, shapes, timing method, and input classification.
- `README.md`, `AGENTS.md`, `Makefile`, and the docs index describe the world-model inference project and its gates.
- The previous toy planning code and notes were moved intact to `archive/planning-prototype/`. Git history also preserves their original paths.

## What it was not allowed to touch

Automatic approval review rejected broad deletion of the previous prototype because its value outside planning had not been established. The work was archived instead. No cloud resources were launched and no GPU or FPGA timing was claimed.

## What a reviewer should check by hand

- **This is a real trained predictor, but its input latents are synthetic.** Strict weight loading and the checkpoint SHA were checked, and the forward pass produced action-dependent outputs. No DINO video encoder or task-quality evaluation ran.
- **The CPU timings are predictor-only.** The script synchronizes supported accelerator devices before and after timing, but these committed results use one CPU thread and exclude model loading, visual encoding, host transfer, and decoding.
- **The operation estimate is incomplete by design.** It counts the large matrix and attention products from the verified shapes. It does not predict FPGA runtime, bandwidth, or power.
