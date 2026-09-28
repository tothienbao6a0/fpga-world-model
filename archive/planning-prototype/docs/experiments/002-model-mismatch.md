# Experiment 002: precision when the world model is wrong

Run on 2026-09-27 with the nonlinear environment. It adds velocity-dependent drag to the simulator, but the learned model remains linear. For each seed 7–11:

```sh
python3 -m worldbench --environment nonlinear --episodes 24 --steps 16 --horizon 4 --bits 3 5 --refine-top-k 2 --seed 7
```

Change `--seed` for the other four runs. Each seed has 24 episodes and 384 decisions per mode. Held-out one-step mean absolute prediction error was 0.0107–0.0121 state units, versus approximately zero for the linear environment. This error is measured on held-out uniformly sampled states, not the closed-loop state distribution.

| Seed | Full-precision final cost | 5-bit final cost | 5-bit + refine final cost | 3-bit + refine local action-flip rate |
| --- | ---: | ---: | ---: | ---: |
| 7 | 0.2178 | 0.1460 | 0.2178 | 23.4% |
| 8 | 0.1245 | 0.1663 | 0.1245 | 22.1% |
| 9 | 0.1488 | 0.1340 | 0.1488 | 27.9% |
| 10 | 0.2249 | 0.0974 | 0.2249 | 18.5% |
| 11 | 0.3191 | 0.1475 | 0.3191 | 23.2% |

Lower final cost is better. The 5-bit planner finished with lower true task cost in four of five seeds, although it differed from the full-precision model's preferred first action on 1.6–4.7% of its visited states. Refining two sequences per first action recovered the full-precision trajectory in all five seeds, which sometimes **raised** true task cost. The 3-bit refinement shortlist frequently failed to recover the full-precision first action despite adding 24 high-precision transitions to 324 coarse transitions per decision.

This does **not** show that quantization improves control. The full-precision planner uses a misspecified model and one planning objective; coarse arithmetic can incidentally offset its error in this small task. The robust conclusion is narrower: local agreement with the full-precision planner and actual closed-loop outcome can disagree. A hardware proposal that reports only action agreement or latent error can therefore miss the relevant failure. The next benchmark should use a released nonlinear world model, multiple tasks, and stronger planners before choosing a precision policy.
