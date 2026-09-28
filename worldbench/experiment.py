"""Closed-loop experiment arithmetic, separate from command-line I/O."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from time import perf_counter_ns

from worldbench.dynamics import LinearWorldModel, State, step
from worldbench.planner import candidate_sequences, choose_action


@dataclass(frozen=True)
class Trace:
    episode: int
    time_step: int
    bits: int | None
    position: float
    velocity: float
    target: float
    action: int
    reference_action: int
    action_flip: bool
    reference_score_of_action: float
    reference_best_score: float
    score_regret: float
    predicted_margin: float | None
    planning_ns: int
    transitions: int


def initial_conditions(episodes: int, seed: int) -> tuple[tuple[State, float], ...]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = Random(seed)
    return tuple(((rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 0.5)), rng.choice((-1.25, 1.25))) for _ in range(episodes))


def run_episode(
    model: LinearWorldModel,
    initial_state: State,
    target: float,
    episode: int,
    steps: int,
    horizon: int,
    bits: int | None,
) -> tuple[list[Trace], float]:
    if steps < 1:
        raise ValueError("steps must be positive")
    candidates = candidate_sequences(horizon)
    state = initial_state
    traces = []
    for time_step in range(steps):
        reference = choose_action(model, state, target, candidates)
        start = perf_counter_ns()
        decision = choose_action(model, state, target, candidates, bits)
        planning_ns = perf_counter_ns() - start
        # Compare first actions, since the controller replans next step. This is
        # model-relative diagnostic regret, not true environment regret.
        reference_score_of_action = reference.score_for_action(decision.action)
        traces.append(
            Trace(
                episode, time_step, bits, state[0], state[1], target,
                decision.action, reference.action, decision.action != reference.action,
                reference_score_of_action, reference.score,
                max(0.0, reference.score - reference_score_of_action),
                decision.margin, planning_ns, decision.transitions,
            )
        )
        state = step(state, decision.action)
    final_cost = (state[0] - target) ** 2 + 0.15 * state[1] ** 2
    return traces, final_cost


def percentile(values: list[int], percent: float) -> int:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * percent / 100))]


def summarize(traces: list[Trace], final_costs: list[float]) -> dict:
    if not traces or not final_costs:
        raise ValueError("nonempty traces and final_costs required")
    times = [trace.planning_ns for trace in traces]
    return {
        "decisions": len(traces),
        "episodes": len(final_costs),
        "action_flip_rate": sum(trace.action_flip for trace in traces) / len(traces),
        "mean_model_relative_regret": sum(trace.score_regret for trace in traces) / len(traces),
        "mean_final_cost": sum(final_costs) / len(final_costs),
        "p50_planning_ns": percentile(times, 50),
        "p95_planning_ns": percentile(times, 95),
    }
