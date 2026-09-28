"""Train once, run paired closed-loop episodes, and emit inspectable JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from worldbench.dynamics import fit_model, training_samples
from worldbench.experiment import initial_conditions, run_episode, summarize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--trace", type=Path, help="write per-decision JSONL here")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.steps < 1 or args.horizon < 1:
        parser.error("--episodes, --steps, and --horizon must be positive")
    if any(bit < 0 for bit in args.bits):
        parser.error("--bits values must be nonnegative")
    try:
        model = fit_model(training_samples(args.train_samples, args.seed))
        conditions = initial_conditions(args.episodes, args.seed + 1)
    except ValueError as error:
        parser.error(str(error))
    all_traces = []
    summaries = {}
    for bits in (None, *dict.fromkeys(args.bits)):
        traces, costs = [], []
        for episode, (state, target) in enumerate(conditions):
            episode_traces, cost = run_episode(model, state, target, episode, args.steps, args.horizon, bits)
            traces.extend(episode_traces)
            costs.append(cost)
        summaries["full" if bits is None else str(bits)] = summarize(traces, costs)
        all_traces.extend(traces)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        with args.trace.open("w", encoding="utf-8") as output:
            for trace in all_traces:
                output.write(json.dumps(asdict(trace), allow_nan=False) + "\n")
    print(json.dumps({"model": asdict(model), "settings": vars(args) | {"trace": str(args.trace) if args.trace else None}, "results": summaries}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
