# Agent work: initial world-model planning benchmark

## What ran

Codex implemented the repo's first runnable benchmark after reading the pasted project discussion, the shared agent rules, and primary research cited in the [direction review](../research/2026-09-direction-review.md) and [opportunity map](../research/2026-09-opportunity-map.md). The original idea shifted from a claimed novel FPGA rollout engine to a measured question about decision quality under numerical and learned-model error.

The gate `python3 -m unittest discover -s worldbench -p '*_test.py'` passed 17 tests. The CLI was run on the linear environment with 12 episodes and on the nonlinear environment with seeds 7–11, 24 episodes and 16 steps per seed. Results and exact commands are in [experiment 001](../experiments/001-toy-precision.md) and [experiment 002](../experiments/002-model-mismatch.md). The working tree was clean after each of three commits was pushed to `main`.

## What it changed

- `worldbench/dynamics.py` fits a tiny linear latent transition model, supports a nonlinear reference environment, and emulates fractional-bit arithmetic.
- `worldbench/planner.py` enumerates action sequences, scores them, and optionally rechecks a per-action shortlist at full precision.
- `worldbench/experiment.py` runs closed-loop episodes and records local action flips, model-relative regret, final cost, transition count, and Python timing.
- `worldbench/cli.py` makes the experiment reproducible and exports JSON summaries and decision traces.
- The tests beside those modules cover model fitting, action changes under low precision, full refinement, CLI output, and environment selection.

The boundary was software benchmarking and research documentation. No RTL, HLS, board integration, or external model dependency was added.

## What it was not allowed to touch

No hardware performance or energy number was claimed because there is no specified board or synthesis result. No pretrained model was downloaded or modified; the toy system first establishes trace semantics and exposes model-error confounding. No real robot control was attempted.

## What a reviewer should check by hand

- **The precision result is not a speedup.** Python fixed-point emulation takes longer than floating-point arithmetic. The measured times are only for this software implementation; inspect the CLI summary before treating them as hardware estimates.
- **Action agreement is not task success.** In the nonlinear experiment the full-precision model is misspecified; 5-bit planning had lower true final cost in four of five seeds, while refinement restored full-precision behavior. This was observed in the five CLI runs, and the narrow interpretation is recorded in experiment 002.
- **The next model must be chosen before hardware design.** The current two-state linear predictor and exhaustive 3-action search do not establish memory traffic, quantization behavior, or planning quality for TD-MPC2 or JEPA-WMs. Review the opportunity map's proposed decision trace and choose one released checkpoint and target board before drawing an FPGA architecture.
