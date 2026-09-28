# Agent work: candidate-action reuse

## What ran

The task was to test whether many candidate actions can share work in the released JEPA-WM Push-T predictor. I read the repo workload study and the pinned upstream `VisionTransformerAdaLN` forward path. The code showed that AdaLN injects action conditioning before attention and MLP in every block, narrowing exact cross-candidate caching to the model input. I ran the official checkpoint on seeded synthetic encoded latents for 1, 2, 4, and 8 candidates, and simulated a four-candidate INT8 QKV weight-broadcast tile with a slice of the released weights.

## What it changed

- `worldmodel/candidates.py` and `worldmodel/candidate_bench.py` add serial, official batched, and shared-input candidate execution with output comparison and recorded CPU timings.
- `worldmodel/bench.py` exposes its pinned checkpoint loader to the new benchmark without changing the original forward path.
- `rtl/qkv_candidate_tile.sv` and `worldmodel/verify_candidate_rtl.py` implement and verify weight broadcast to four candidate activation vectors; the latter records interface traffic counts.
- `results/jepa_wm_pusht_candidate_cpu_macos_arm64.json` captures the measured CPU run; the README and workload study interpret it.
- `Makefile` includes the new synthetic and checkpoint-backed checks. No upstream model source, checkpoint, dataset, or cloud resource was changed.

## What it was not allowed to touch

This increment stayed at predictor inference and a QKV tile. It did not claim task quality, a full accelerator, routed timing, or FPGA speedup because there is no real trajectory evaluation or cloud FPGA result yet. The official task dataset remains gated; no access conditions were bypassed.

## What a reviewer should check by hand

- **Exact block-output caching is unavailable for this model.** The pinned upstream forward path feeds action embeddings into AdaLN at every block. This was established from source inspection; the benchmark only confirms output equivalence of three execution paths on synthetic inputs.
- **The 4× number is weight-interface traffic for one tile.** RTL simulation matched the integer oracle for four synthetic candidate vectors and the first 16 rows of the checkpoint QKV matrix. It says nothing about DRAM traffic, DSP feasibility, timing closure, or end-to-end latency.
- **The CPU benefit is small and noisy.** Eight two-frame candidates measured 604.0 ms serial versus 577.7 ms with shared input projection on one Mac CPU thread, five repeats. A GPU run and repeated hardware measurements are needed before selecting an accelerator architecture.

`make check` and `make model-check` passed after the final code changes. A separate four-frame/two-candidate run also passed the numerical comparison.
