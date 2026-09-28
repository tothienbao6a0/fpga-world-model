# Project index

- I want to understand the project and the real model workload: [world-model inference study](research/world-model-workload.md).
- I want to reproduce checkpoint-backed measurements: [setup and commands](../README.md#reproduce).
- I want to inspect the first measured runs: [two-frame CPU result](../results/jepa_wm_pusht_cpu_macos_arm64.json) and [four-frame CPU result](../results/jepa_wm_pusht_cpu_4frames_macos_arm64.json).
- I want to inspect the first real-model hardware block: [INT8 QKV tile and its limits](research/world-model-workload.md#first-hardware-block).
- I want to see whether candidate actions share useful work: [candidate-action measurements and broadcast tile](research/world-model-workload.md#candidate-action-reuse).
- I want to see INT8 error on actual predictor activations: [QKV activation probe](research/world-model-workload.md#checkpoint-activation-probe).
- I want to review what changed in the pivot: [agent handoff](agents/checkpoint-workload.md).
- I want to review the candidate-action increment: [candidate reuse handoff](agents/candidate-action-reuse.md).
- I want to review the checkpoint activation probe: [probe handoff](agents/checkpoint-activation-probe.md).
- I want the previous planning prototype: [archived prototype](../archive/planning-prototype/README.md).
