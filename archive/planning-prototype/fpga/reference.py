"""Bit-exact Q8.8 oracle for the streaming RTL engine."""

from __future__ import annotations

from dataclasses import dataclass

from worldbench.dynamics import LinearWorldModel

SCALE = 256
MIN_Q = -32768
MAX_Q = 32767


def to_q(value: float) -> int:
    scaled = round(value * SCALE)
    if not MIN_Q <= scaled <= MAX_Q:
        raise ValueError("value is outside signed Q8.8 range")
    return scaled


def pack_weights(weights: tuple[int, int, int, int]) -> int:
    return sum((weight & 0xFFFF) << (16 * index) for index, weight in enumerate(weights))


def model_weights(model: LinearWorldModel) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    return tuple(to_q(value) for value in model.position_weights), tuple(to_q(value) for value in model.velocity_weights)


def dot4(weights: tuple[int, int, int, int], position: int, velocity: int, action: int) -> int:
    total = weights[0] * position + weights[1] * velocity + weights[2] * (action * SCALE) + weights[3] * SCALE
    return max(MIN_Q, min(MAX_Q, total >> 8))


@dataclass(frozen=True)
class Result:
    action: int
    cost: int


def evaluate(
    position: int,
    velocity: int,
    target: int,
    position_weights: tuple[int, int, int, int],
    velocity_weights: tuple[int, int, int, int],
    candidates: tuple[tuple[int, ...], ...],
) -> Result:
    if not candidates or len({len(sequence) for sequence in candidates}) != 1:
        raise ValueError("candidates must have equal, nonzero length")
    if not candidates[0] or any(action not in (-1, 0, 1) for sequence in candidates for action in sequence):
        raise ValueError("actions must be -1, 0, or 1")
    best = Result(0, (1 << 64) - 1)
    for sequence in candidates:
        p, v = position, velocity
        penalty = 0
        for action in sequence:
            p, v = dot4(position_weights, p, v, action), dot4(velocity_weights, p, v, action)
            penalty += 2621 if action else 0
        cost = (p - target) ** 2 + ((v * v * 9830) >> 16) + penalty
        if cost < best.cost:
            best = Result(sequence[0], cost)
    return best
