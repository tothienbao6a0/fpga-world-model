"""Pure dynamics, quantization, and a tiny learned linear world model."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

State = tuple[float, float]
Action = int
FEATURES = 4


def step(state: State, action: Action) -> State:
    """Reference environment: a damped point mass with bounded acceleration."""
    position, velocity = state
    next_velocity = 0.82 * velocity + 0.32 * action
    return position + next_velocity, next_velocity


def quantize(value: float, fractional_bits: int | None) -> float:
    if fractional_bits is None:
        return value
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be nonnegative")
    scale = 1 << fractional_bits
    return round(value * scale) / scale


@dataclass(frozen=True)
class LinearWorldModel:
    """Learned mapping [position, velocity, action, 1] -> next state."""

    position_weights: tuple[float, float, float, float]
    velocity_weights: tuple[float, float, float, float]

    def predict(self, state: State, action: Action, bits: int | None = None) -> State:
        features = (*state, float(action), 1.0)
        outputs = []
        for weights in (self.position_weights, self.velocity_weights):
            total = 0.0
            for weight, feature in zip(weights, features):
                total = quantize(total + quantize(weight, bits) * quantize(feature, bits), bits)
            outputs.append(total)
        return outputs[0], outputs[1]


def _solve(matrix: list[list[float]], rhs: list[float]) -> tuple[float, ...]:
    """Solve a small positive-definite ridge system with pivoted elimination."""
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for col in range(FEATURES):
        pivot = max(range(col, FEATURES), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("training data is rank deficient")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(FEATURES):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return tuple(row[-1] for row in augmented)


def fit_model(samples: list[tuple[State, Action, State]], ridge: float = 1e-8) -> LinearWorldModel:
    if not samples:
        raise ValueError("at least one transition is required")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    gram = [[0.0] * FEATURES for _ in range(FEATURES)]
    targets = [[0.0] * FEATURES for _ in range(2)]
    for state, action, next_state in samples:
        features = (*state, float(action), 1.0)
        for i in range(FEATURES):
            for j in range(FEATURES):
                gram[i][j] += features[i] * features[j]
            for output in range(2):
                targets[output][i] += features[i] * next_state[output]
    for i in range(FEATURES):
        gram[i][i] += ridge
    return LinearWorldModel(_solve(gram, targets[0]), _solve(gram, targets[1]))


def training_samples(count: int, seed: int) -> list[tuple[State, Action, State]]:
    if count < FEATURES:
        raise ValueError(f"count must be at least {FEATURES}")
    rng = Random(seed)
    samples = []
    for _ in range(count):
        state = (rng.uniform(-2, 2), rng.uniform(-1, 1))
        action = rng.choice((-1, 0, 1))
        samples.append((state, action, step(state, action)))
    return samples
