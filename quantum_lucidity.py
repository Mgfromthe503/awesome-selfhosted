"""Utilities for the toy "Quantum Lucidity" prediction workflow.

This module provides a dependency-light version of the user supplied script,
including definitions for all referenced functions and deterministic behavior
that is easy to test.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


def unsolved_equations_solver(
    equation: Callable[[float], float],
    initial_guess: float,
    max_iter: int = 100,
    tolerance: float = 1e-10,
) -> float:
    """Solve a single-variable equation f(x)=0 via Newton's method.

    This replaces SciPy's ``fsolve`` so the script works in minimal
    environments without external dependencies.
    """

    x = float(initial_guess)
    h = 1e-6

    for _ in range(max_iter):
        fx = equation(x)
        if abs(fx) < tolerance:
            return x

        # Numerical derivative.
        dfx = (equation(x + h) - equation(x - h)) / (2 * h)
        if abs(dfx) < 1e-14:
            raise ValueError("Derivative too small; Newton method stalled")

        x -= fx / dfx

    raise ValueError("Solver did not converge within max_iter")


def pi_approximation(iterations: int) -> float:
    """Approximate pi using the Leibniz series."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    return 4 * sum(((-1) ** k) / (2 * k + 1) for k in range(iterations))


def feedback_adjustment(
    observed_outcome: float, predicted_outcome: float, coefficients: Iterable[float]
) -> List[float]:
    """Shift all coefficients by prediction error."""

    adjustment_factor = observed_outcome - predicted_outcome
    return [coef + adjustment_factor for coef in coefficients]


def elemental_factors(elements: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Extract Earth/Water/Air/Fire coefficients with zero defaults."""

    return (
        float(elements.get("Earth", 0.0)),
        float(elements.get("Water", 0.0)),
        float(elements.get("Air", 0.0)),
        float(elements.get("Fire", 0.0)),
    )


def numerology_angel_factors(B: int, S: Dict[str, float]) -> Tuple[float, float]:
    """Generate normalized numerology features from integer and metrics mapping."""

    digits = [int(d) for d in str(abs(int(B)))]
    digit_sum = sum(digits)
    nb = (digit_sum % 100) / 100.0

    if S:
        avg_metric = sum(float(v) for v in S.values()) / len(S)
    else:
        avg_metric = 0.0
    as_factor = 1 / (1 + math.exp(-avg_metric / 10))

    return nb, as_factor


def machine_learning_based_qlp(features: Sequence[float]) -> float:
    """Deterministic stand-in for model inference.

    Uses fixed weights and a sigmoid transform to produce a score in [0, 1].
    """

    weights = [0.22, 0.18, 0.15, 0.13, 0.12, 0.1, 0.1]
    if len(features) != len(weights):
        raise ValueError("features must contain exactly 7 values")

    score = sum(w * float(f) for w, f in zip(weights, features))
    return 1 / (1 + math.exp(-score))


def quantum_lucidity(Ce: float, Cw: float, Ca: float, Cf: float, NB: float, AS: float, T_QL: float) -> float:
    """Compute quantum lucidity prediction score from 7 factors."""

    return machine_learning_based_qlp([Ce, Cw, Ca, Cf, NB, AS, T_QL])


def main() -> None:
    M = {"Earth": 0.4, "Water": 0.5, "Air": 0.6, "Fire": 0.7}
    B = 112288
    S = {"SomeMetric": 42}

    Ce, Cw, Ca, Cf = elemental_factors(M)
    NB, AS = numerology_angel_factors(B, S)
    T_QL = 0.5
    qlp_result = quantum_lucidity(Ce, Cw, Ca, Cf, NB, AS, T_QL)
    print("Quantum Lucidity Prediction:", qlp_result)

    observed_outcome = 0.75
    adjusted_coefficients = feedback_adjustment(observed_outcome, qlp_result, [Ce, Cw, Ca, Cf])
    print("Adjusted Coefficients:", adjusted_coefficients)

    solution = unsolved_equations_solver(lambda x: x**2 - 2, 1)
    print("Solution to x^2 - 2 = 0:", solution)

    print("Approximation of Pi:", pi_approximation(10000))


if __name__ == "__main__":
    main()
