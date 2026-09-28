# Experiment 003: candidate breadth before cloud FPGA

## Question and setup

Does increasing the streamed candidate bank from nine to all 81 length-four action sequences improve the toy controller enough to justify nine times as many action cycles? The nine-plan bank explores the first two actions and holds the last two at zero. The 81-plan bank explores all four positions. Both fixed-point and floating-point planners see the same bank within each run.

Reproduce with `python3 -m fpga.sweep`. Defaults: seeds 7–11, 24 episodes per seed, 16 decisions per episode, 256 model-fitting transitions per seed, and both linear and nonlinear toy environments. Each row covers 1,920 decisions. The fixed-point figures use the bit-exact software contract; one 81-plan stream was also checked against the RTL in `make check`.

| Environment | Plans | Ideal cycles/decision | Fixed-vs-float action flips | Fixed mean final cost | Float mean final cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 9 | 37 | 89 / 1,920 | 0.1593 | 0.1613 |
| Linear | 81 | 325 | 21 / 1,920 | 0.2196 | 0.2112 |
| Nonlinear | 9 | 37 | 48 / 1,920 | 0.1490 | 0.1467 |
| Nonlinear | 81 | 325 | 28 / 1,920 | 0.2007 | 0.2070 |

## Interpretation

Broader search made the fixed-point choice closer to the float planner's choice in both environments. It did **not** improve closed-loop task cost in this small study. A plausible explanation is that the receding-horizon objective and imperfect learned dynamics can favor short-term action sequences that are worse after replanning; the experiment does not isolate that mechanism. The nine-plan bank acts as an action constraint, so this is not a pure arithmetic comparison.

The 37 and 325 figures are ideal stream schedules: one start cycle plus four cycles per candidate, with no stalls. They are not measured device latency. Before using AWS F2, we need a target clock, host-transfer measurement, a meaningful world-model checkpoint, and a CPU/GPU baseline. The result argues for keeping candidate breadth configurable rather than assuming exhaustive search is worth its cost.
