"""Pure signed INT8 quantization and INT32 QKV tile reference."""

from __future__ import annotations


def symmetric_int8(values: tuple[float, ...]) -> tuple[tuple[int, ...], float]:
    if not values:
        raise ValueError("at least one value is required")
    peak = max(abs(value) for value in values)
    scale = peak / 127 if peak else 1.0
    return tuple(max(-127, min(127, round(value / scale))) for value in values), scale


def quantize_candidate_tile(
    activations: tuple[tuple[float, ...], ...],
    weights: tuple[tuple[float, ...], ...],
    biases: tuple[float, ...],
    activation_scale: float | None = None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...], float, tuple[float, ...]]:
    """Quantize one weight-broadcast tile with a shared candidate activation scale."""
    if not activations or not activations[0] or not weights or len(weights) != len(biases):
        raise ValueError("tile dimensions must be nonempty and aligned")
    width = len(activations[0])
    if any(len(row) != width for row in activations + weights):
        raise ValueError("all activation and weight rows must have the same width")
    flattened = tuple(value for row in activations for value in row)
    if activation_scale is None:
        packed, activation_scale = symmetric_int8(flattened)
    else:
        if activation_scale <= 0:
            raise ValueError("activation scale must be positive")
        packed = tuple(max(-127, min(127, round(value / activation_scale))) for value in flattened)
    quantized_activations = tuple(packed[index:index + width] for index in range(0, len(packed), width))
    quantized_weights_and_scales = tuple(symmetric_int8(row) for row in weights)
    quantized_weights = tuple(row for row, _ in quantized_weights_and_scales)
    weight_scales = tuple(scale for _, scale in quantized_weights_and_scales)
    quantized_biases = tuple(round(bias / (activation_scale * scale))
                             for bias, scale in zip(biases, weight_scales))
    return quantized_activations, quantized_weights, quantized_biases, activation_scale, weight_scales


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
