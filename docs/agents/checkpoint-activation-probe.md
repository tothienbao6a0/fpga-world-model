# Agent work: checkpoint activation probe

## What ran

I followed the pinned JEPA-WM Push-T predictor into block 0's QKV projection, capturing its input and first 16 output rows while four candidate actions used one seeded synthetic encoded visual context. I compared two symmetric INT8 activation scales: one per token shared across candidates, and one for the full layer. Per-row weight scales and INT32 biases were used in both cases. The RTL simulator checked four sampled tokens in each mode against the integer oracle.

## What it changed

- `worldmodel/quant.py` adds pure quantization arithmetic for candidate activations with either scale choice.
- `worldmodel/activation_probe.py` captures real predictor activations, measures all-token software error, and feeds selected values into the existing candidate RTL verifier.
- Nearby tests cover scale handling, numerical summaries, and tensor shape checks.
- `results/jepa_wm_pusht_qkv_candidate_activations_cpu_macos_arm64.json` records the run; the README and workload study explain its scope.
- `Makefile` runs a checkpoint-backed activation probe in `model-check`. The FPGA tile itself was unchanged in this increment.

## What it was not allowed to touch

The probe did not alter upstream weights or code, use gated Push-T trajectories, or claim world-model prediction quality. It measured QKV error for 16 rows, not full predictor INT8 inference.

## What a reviewer should check by hand

- **The inputs are genuinely action conditioned.** A forward pre-hook on block 0 QKV captured different inputs for different candidate actions; the four sampled tokens differ by 0.72–1.30 at maximum from candidate 0. The visual context remains synthetic.
- **Software and RTL have different coverage.** Software compared 32,768 QKV scalars across every token and four candidates. RTL verified 512 scalars for four selected tokens and two scale modes, all against the integer oracle.
- **Layer scaling trades simplicity for error in this run.** Relative L2 error rose from 1.96% to 2.38% on the measured QKV slice. That is not evidence of a task-success difference or a hardware performance benefit.

`make check` and `make model-check` passed after the final edits.
