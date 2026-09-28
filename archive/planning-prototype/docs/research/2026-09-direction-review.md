# Direction review: learned world-model planning on FPGA

Research checked 2026-09-27. This is a literature and feasibility review; the repo contains no implementation, measured profile, or hardware result yet. The pasted discussion proposed “Bonsai,” an anytime FPGA planner that allocates learned latent rollouts based on decision margin and uncertainty. The proposal is worth testing, but its prior novelty and performance language outruns the evidence.

The subsequent [opportunity map](2026-09-opportunity-map.md) found more direct overlap: the [COMS 6998 project menu](https://wan-research-group.github.io/coms6998-f26/project.html) explicitly proposes deadline-aware world-model planning, decision-preserving quantization, and planning-aware memory architecture. These are project prompts rather than published results, but they make a broad “Bonsai” pitch even less distinctive. Its research value must come from a specific mechanism and measured result.

## What the literature establishes

| Existing result | What it means for this project |
| --- | --- |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) plans through a learned decoder-free latent model. Its [official planner](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py) samples trajectories, evaluates rewards/transitions and terminal value, selects elites, and iterates. | A credible model and baseline exist. Its continuous-action MPPI-like planner is **not** an action tree, so replacing it with a small discrete action library changes the planning problem and must be compared on return. |
| [FPGA MPPI](https://arxiv.org/abs/2601.17231) reports 3.1–7.5× speedup and 2.5–5.4× less energy than its embedded GPU/CPU baselines for sampling-based control. | “Parallel rollouts on FPGA” is already covered. Those numbers belong to that paper's workloads and cannot be transferred to learned latent dynamics. Its pipelining deliberately avoids control-flow checks; adaptive frontier scheduling creates the opposite hardware pressure. |
| [WMHA](https://arxiv.org/abs/2609.16244) targets static-shape diffusion-transformer denoising with a VLIW sequencer and FP8/BF16 datapath. | It is a different workload. It does not establish the efficacy of a learned-dynamics planner, and Bonsai should not claim to be the first world-model hardware accelerator. |
| [Dream and Search to Control](https://arxiv.org/abs/2010.09832), [ELAST](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html), and [L-MAP](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9a419ce12db8db2e35ee68c1f58c1a36-Abstract-Conference.html) already search learned latent dynamics, including continuous actions, explicit trees, and learned macro-actions respectively. | Prefix sharing, latent trees, and action chunks are prior techniques. The possible contribution is their measured *hardware scheduling trade-off* under a deadline, not the search formulation alone. |
| [Adaptive Horizon MPC](https://arxiv.org/abs/1602.08619) and [Neural Horizon MPC](https://arxiv.org/abs/2408.09781) vary or approximate planning work to save computation. | “Spend less planning compute when the decision is easy” is prior art. A new claim must specify the learned-model workload, the accelerator mechanism, and a measured advantage over adaptive software. |
| [Speculation and correction with TD-MPC2](https://arxiv.org/abs/2512.17250) reuses an action queue and replans after latent mismatch; it reports 25% lower step latency with a 7.1% return reduction on one DMC Humanoid-Walk experiment. | The earlier “speculative imagination” framing is occupied and its quality cost matters. This is adjacent, though it reduces replanning *across control steps* rather than deciding which branches to expand *within a step*. |

**Inference from these sources:** I found no direct demonstration of a deadline-aware FPGA frontier scheduler for *learned latent dynamics* in this review. That is a possible gap, not proof of novelty; a full paper search and comparison to hardware tree-search systems would still be required before a publication claim. [CPU–FPGA MCTS](https://arxiv.org/abs/2208.11208) already accelerates in-tree operations, so even “tree scheduling in SRAM” is not by itself new.

## Corrections to the pasted discussion

- “Approximately 9,200 latent predictions per action” is not a general TD-MPC2 fact. The count depends on samples, horizon, iterations, policy trajectories, and termination. Derive and report the count from the exact configuration and executed code path.
- “Upper value = predicted value + uncertainty” is a **heuristic score**, not a valid branch-and-bound upper bound. Model uncertainty is not automatically an error bound on total return. Calling the method branch-and-bound or claiming safe pruning needs a proved bound under explicit assumptions; otherwise call it adaptive pruning and measure wrong-prune frequency.
- A lower rollout count does not imply lower decision latency. Smaller, irregular batches can reduce accelerator utilization and add compaction, dispatch, or transfer cost. This is the central risk to test.
- A low-precision model can change action ranking even when prediction error looks small. Measure action agreement, return, and missed deadlines, not just latent MSE or operations saved.
- The previous 97–99% return target and 1/5/10 ms examples were illustrative aspirations, not sourced estimates. Set deadlines from a selected task and controller, then report actual distributions.

## Working research question

For a fixed pretrained latent dynamics model and a fixed control deadline, can an FPGA scheduler allocate rollout work adaptively and improve the **return–deadline-miss–energy** trade-off over (1) the original planner, (2) strong adaptive software planning, and (3) a fixed-schedule FPGA implementation?

The narrow potential systems contribution is a measured implementation of compacting surviving latent states, reusing shared prefixes where candidate sequences truly share them, and scheduling remaining model calls before a deadline. Uncertainty can guide scheduling, but it needs calibration on held-out trajectories and shift tests. Avoid a GPU verifier, robot, new world-model architecture, and formal safety claim in the first version.

## Experiments and decision gates

1. **Reproduce a baseline.** Select one official [TD-MPC2 checkpoint/task](https://github.com/nicklashansen/tdmpc2), pin code and model versions, and reproduce closed-loop return over multiple seeds. Record planner configuration and a per-decision trace: model calls, shapes, time in encoding/dynamics/reward/value/sampling/selection, and p50/p95/p99 wall time. Use actual hardware and synchronization when timing accelerators. A second task with different action dimension or dynamics is needed before generalizing.
2. **Check that adaptive planning helps in software.** Compare the official planner, reduced fixed budgets, a deadline-aware adaptive scheduler, and (only if tree structure is justified) a fixed-depth tree. Match total model, candidate-generation method, task, seeds, deadline, and tuning budget. Plot return against actual decision latency and number of missed deadlines. Include a no-planning policy baseline to show the planner earns its cost.
3. **Check trustworthiness.** On held-out and shifted conditions, measure whether low-cost scores rank actions correctly, wrong-prune rate relative to a high-budget reference, value error by depth, and calibration/coverage of any uncertainty estimate. If the first action cannot be distinguished reliably, pruning savings may be a false economy.
4. **Build a hardware cost model before RTL.** Choose a real FPGA board/toolchain and model dimensions. Estimate weights, live frontier state, BRAM/DSP use, compute throughput, off-chip bandwidth, host link latency, and quantization loss. A CPU+GPU whole-system timing comparison is mandatory; kernel-only cycles are insufficient. If the model or frontier cannot stay suitably on chip, revisit the architecture.
5. **Prototype only the demonstrated bottleneck.** If software profiling shows regular dynamics inference dominates, implement a fixed-schedule quantized rollout baseline first. If scheduling/compaction dominates and adaptive planning improves the Pareto frontier, add the adaptive frontier unit. Report synthesis/timing closure and measured board latency/power if a board is available; label simulation and estimates clearly when it is not.

**Proceed to FPGA** if adaptive software improves the return/deadline trade-off, the model remains accurate enough at the relevant horizon and precision, and a board-level cost model leaves meaningful headroom after I/O. **Pivot** if a batched GPU/CPU planner is already under deadline or adaptive pruning loses its savings to irregularity. **Stop this framing** if model error or macro-action discretization destroys control quality. These are decision rules, not thresholds claimed by a paper.

## Questions still needing local facts

- Which FPGA board, host connection, power measurement method, and synthesis tools are actually available?
- What is the HPML course deadline and expected artifact: working board, RTL simulation, or systems study?
- Which control task and target control rate matter to Bao and Wesley? Is a physical crawler a later demo or a required deliverable?
- Does the selected checkpoint license and environment setup allow reproducible use in this repo?

## Primary sources

1. Hansen et al., [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828); [official code and checkpoints](https://github.com/nicklashansen/tdmpc2).
2. Desai et al., [Real-Time, Energy-Efficient, Sampling-Based Optimal Control via FPGA Acceleration](https://arxiv.org/abs/2601.17231).
3. Chaurasia, [The World Model Hardware Accelerator](https://arxiv.org/abs/2609.16244).
4. Koul et al., [Dream and Search to Control](https://arxiv.org/abs/2010.09832); Gieselmann and Pokorny, [Latent Planning via Expansive Tree Search](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html).
5. [L-MAP: Scalable Decision-Making in Stochastic Environments through Learned Temporal Abstraction](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9a419ce12db8db2e35ee68c1f58c1a36-Abstract-Conference.html).
6. Krener, [Adaptive Horizon MPC](https://arxiv.org/abs/1602.08619); Alsmeier et al., [Neural Horizon MPC](https://arxiv.org/abs/2408.09781).
7. Lin et al., [Accelerating Multi-modal LLM Gaming Performance via Input Prediction and Mishit Correction](https://arxiv.org/abs/2512.17250).
8. Meng et al., [Accelerating Monte-Carlo Tree Search on CPU–FPGA Heterogeneous Platform](https://arxiv.org/abs/2208.11208).
