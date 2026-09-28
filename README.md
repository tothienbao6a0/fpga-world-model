# FPGA world-model planning research

This repository is investigating whether an FPGA can improve **decision latency and energy** for planning with a small learned latent dynamics model. The current working idea is a deadline-aware scheduler that evaluates candidate action futures, spends more work on close decisions, and returns an action before a fixed control deadline.

This is a research hypothesis, not a demonstrated speedup or a settled novelty claim. The first runnable baseline is now in `worldbench/`. It learns a two-state linear dynamics model from simulator transitions, searches candidate action sequences, and repeats the task with full-precision and fixed-point arithmetic. It records action flips, model-relative score regret, final task cost, and Python planning time.

```sh
python3 -m unittest discover -s worldbench -p '*_test.py'
python3 -m worldbench --episodes 12 --steps 12 --horizon 4 --bits 3 5 8 --trace traces/first-run.jsonl
```

The summary prints as JSON; `--trace` writes one JSON line per control decision. This is a **toy workload**: its dynamics are exactly representable by the learned model, candidate search is exhaustive, and Python fixed-point emulation is slower than floating point. The timing numbers are only a software baseline, not FPGA performance or energy evidence. The next build is a more realistic checkpoint-backed decision trace and a hardware cost model.

Start with [the research index](docs/README.md), [the September 2026 evidence review](docs/research/2026-09-direction-review.md), and [the broader opportunity map](docs/research/2026-09-opportunity-map.md).
