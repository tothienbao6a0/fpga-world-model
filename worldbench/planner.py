"""Pure candidate generation and rollout scoring; no file or clock access."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from worldbench.dynamics import Action, LinearWorldModel, State


@dataclass(frozen=True)
class Decision:
    action: Action
    sequence: tuple[Action, ...]
    score: float
    margin: float | None
    transitions: int
    first_action_scores: tuple[tuple[Action, float], ...]

    def score_for_action(self, action: Action) -> float:
        for candidate_action, score in self.first_action_scores:
            if candidate_action == action:
                return score
        raise ValueError(f"action {action} is not in the candidate bank")


def candidate_sequences(horizon: int) -> tuple[tuple[Action, ...], ...]:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return tuple(product((-1, 0, 1), repeat=horizon))


def score_sequence(
    model: LinearWorldModel,
    state: State,
    target: float,
    sequence: tuple[Action, ...],
    bits: int | None,
) -> float:
    current = state
    cost = 0.0
    for action in sequence:
        current = model.predict(current, action, bits)
        cost += 0.04 * action * action
    cost += (current[0] - target) ** 2 + 0.15 * current[1] ** 2
    return -cost


def choose_action(
    model: LinearWorldModel,
    state: State,
    target: float,
    candidates: tuple[tuple[Action, ...], ...],
    bits: int | None = None,
) -> Decision:
    if not candidates or any(not sequence for sequence in candidates):
        raise ValueError("candidates must contain nonempty sequences")
    scored = sorted(
        ((score_sequence(model, state, target, sequence, bits), sequence) for sequence in candidates),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, best_sequence = scored[0]
    by_action = {}
    for score, sequence in scored:
        by_action.setdefault(sequence[0], score)
    next_score = max((score for action, score in by_action.items() if action != best_sequence[0]), default=None)
    margin = best_score - next_score if next_score is not None else None
    return Decision(best_sequence[0], best_sequence, best_score, margin, sum(map(len, candidates)), tuple(sorted(by_action.items())))
