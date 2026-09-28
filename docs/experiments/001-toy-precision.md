# Experiment 001: coarse planning and selective refinement

Run on 2026-09-27 with seed 7 using the selective-refinement implementation:

```sh
python3 -m worldbench --episodes 12 --steps 12 --horizon 4 --bits 3 5 8 --refine-top-k 2
```

The model was fitted to 256 deterministic transitions of a two-state damped point mass. Every planner sees the same 12 initial conditions, but its later states can diverge as it chooses different actions. The full-precision planner at each *visited* state is the local reference. Lower final cost is better. A 4-step search evaluates 81 action sequences, or 324 model transitions, per decision. Refinement repeats two sequences for each of the three first actions at full precision, adding 24 transitions.

| Planner | Local first-action flips | Mean final cost | Mean model transitions / decision |
| --- | ---: | ---: | ---: |
| Full precision | 0% | 0.1934 | 324 |
| 3 fractional bits | 27.8% | 0.4704 | 324 |
| 3 bits + refine 2/action | 29.2% | 0.4398 | 348 |
| 5 fractional bits | 4.2% | 0.1883 | 324 |
| 5 bits + refine 2/action | 0% | 0.1934 | 348 |
| 8 fractional bits | 0% | 0.1934 | 324 |

**Observation:** refining a tiny shortlist is not automatically better. At 3 bits, coarse rankings can exclude the best full-precision sequence before refinement; local first-action flips rose. At 5 bits, the shortlist was adequate for this seed and restored the full-precision action at every visited state. The slightly lower final cost of plain 5-bit planning is an outcome of these particular trajectories, not evidence it is a better controller.

**Limits:** this is one seed and a toy model whose dynamics class matches the simulator exactly. “Local first-action flip” compares each planner to full precision at its own visited state; it is not a paired trajectory action-agreement metric. The model-relative regret diagnostic is not regret against the environment. Python timing is dominated by emulation and cannot predict FPGA speed or energy. No claim about general world models follows from this run.

**Next discriminating test:** use a released nonlinear visual or continuous-control model, hold candidate banks fixed, and measure whether coarse scores retain the true best first action's strongest plans. Compare top-k shortlists to margin-triggered full recomputation under a matched compute budget. Do not build a selective-precision datapath until that signal is reliable.
