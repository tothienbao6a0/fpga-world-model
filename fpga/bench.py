"""Run the hardware arithmetic contract inside a closed-loop control task."""

from __future__ import annotations

import argparse
import json
from itertools import product

from fpga.reference import evaluate, model_weights, to_q
from worldbench.dynamics import fit_model, nonlinear_step, step, training_samples
from worldbench.experiment import initial_conditions
from worldbench.planner import choose_action


def candidate_bank() -> tuple[tuple[int, ...], ...]:
    """Nine length-four plans; all three first actions are represented."""
    return tuple((first, second, 0, 0) for first, second in product((-1, 0, 1), repeat=2))


def run_benchmark(episodes: int, steps: int, seed: int, nonlinear: bool) -> dict:
    if episodes < 1 or steps < 1:
        raise ValueError("episodes and steps must be positive")
    dynamics = nonlinear_step if nonlinear else step
    model = fit_model(training_samples(256, seed, dynamics))
    position_weights, velocity_weights = model_weights(model)
    candidates = candidate_bank()
    flips = 0
    hardware_final_costs = []
    float_final_costs = []
    for initial_state, target in initial_conditions(episodes, seed + 1):
        hardware_state = float_state = initial_state
        for _ in range(steps):
            hardware_decision = evaluate(
                to_q(hardware_state[0]), to_q(hardware_state[1]), to_q(target),
                position_weights, velocity_weights, candidates,
            )
            local_reference = choose_action(model, hardware_state, target, candidates)
            flips += hardware_decision.action != local_reference.action
            hardware_state = dynamics(hardware_state, hardware_decision.action)
            float_decision = choose_action(model, float_state, target, candidates)
            float_state = dynamics(float_state, float_decision.action)
        hardware_final_costs.append((hardware_state[0] - target) ** 2 + 0.15 * hardware_state[1] ** 2)
        float_final_costs.append((float_state[0] - target) ** 2 + 0.15 * float_state[1] ** 2)
    return {
        "episodes": episodes,
        "decisions": episodes * steps,
        "environment": "nonlinear" if nonlinear else "linear",
        "candidate_count": len(candidates),
        "horizon": len(candidates[0]),
        "ideal_stream_cycles_per_decision": 1 + len(candidates) * len(candidates[0]),
        "local_action_flip_rate": flips / (episodes * steps),
        "hardware_contract_mean_final_cost": sum(hardware_final_costs) / episodes,
        "float_mean_final_cost": sum(float_final_costs) / episodes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--environment", choices=("linear", "nonlinear"), default="linear")
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(args.episodes, args.steps, args.seed, args.environment == "nonlinear")
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
