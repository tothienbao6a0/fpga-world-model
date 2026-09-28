"""Pure signed INT8 quantization and INT32 QKV tile reference."""

from __future__ import annotations


def symmetric_int8(values: tuple[float, ...]) -> tuple[tuple[int, ...], float]:
    if not values:
        raise ValueError("at least one value is required")
    peak = max(abs(value) for value in values)
    scale = peak / 127 if peak else 1.0
    return tuple(max(-127, min(127, round(value / scale))) for value in values), scale


def quantized_tile(
    activations: tuple[int, ...],
    weights: tuple[tuple[int, ...], ...],
    biases: tuple[int, ...],
) -> tuple[int, ...]:
    if not activations or len(weights) != len(biases) or not weights:
        raise ValueError("tile dimensions must be nonempty and aligned")
    if any(len(row) != len(activations) for row in weights):
        raise ValueError("each weight row must match activation length")
    if any(not -127 <= value <= 127 for value in activations):
        raise ValueError("activations must be signed symmetric INT8")
    if any(not -127 <= value <= 127 for row in weights for value in row):
        raise ValueError("weights must be signed symmetric INT8")
    return tuple(bias + sum(a * w for a, w in zip(activations, row))
                 for row, bias in zip(weights, biases))
