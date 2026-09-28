"""Compare sparse and exhaustive hardware contracts over multiple seeds."""

from __future__ import annotations

import argparse
import json

from fpga.bench import run_benchmark


def run_sweep(episodes: int, steps: int, first_seed: int, seeds: int) -> list[dict]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    summaries = []
    for nonlinear in (False, True):
        for exhaustive in (False, True):
            runs = [run_benchmark(episodes, steps, seed, nonlinear, exhaustive)
                    for seed in range(first_seed, first_seed + seeds)]
            decisions = sum(run["decisions"] for run in runs)
            summaries.append({
                "environment": runs[0]["environment"],
                "candidate_bank": runs[0]["candidate_bank"],
                "candidate_count": runs[0]["candidate_count"],
                "ideal_stream_cycles_per_decision": runs[0]["ideal_stream_cycles_per_decision"],
                "seeds": seeds,
                "decisions": decisions,
                "local_action_flip_rate": sum(run["local_action_flip_rate"] * run["decisions"] for run in runs) / decisions,
                "hardware_contract_mean_final_cost": sum(run["hardware_contract_mean_final_cost"] for run in runs) / seeds,
                "float_mean_final_cost": sum(run["float_mean_final_cost"] for run in runs) / seeds,
            })
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--first-seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        summary = run_sweep(args.episodes, args.steps, args.first_seed, args.seeds)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
