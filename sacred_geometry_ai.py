"""Sacred geometry utilities for 11-dimensional vectors.

This module provides:
- ``Vector11D`` for common vector math in 11 dimensions.
- Fractal point generation helpers inspired by classical sacred-geometry shapes.
- ``SacredGeometryAI``: a lightweight prototype-based classifier/scorer that
  demonstrates how geometric features can be used in simple AI workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class Vector11D:
    """An immutable 11-dimensional vector with sacred-geometry helpers."""

    coords: Tuple[float, ...]

    def __init__(self, *coords: float) -> None:
        if len(coords) != 11:
            raise ValueError("Vector11D requires exactly 11 coordinates")
        object.__setattr__(self, "coords", tuple(float(c) for c in coords))

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "Vector11D":
        return cls(*values)

    def to_list(self) -> List[float]:
        return list(self.coords)

    def __getitem__(self, index: int) -> float:
        return self.coords[index]

    def __add__(self, other: "Vector11D") -> "Vector11D":
        return Vector11D(*(a + b for a, b in zip(self.coords, other.coords)))

    def __sub__(self, other: "Vector11D") -> "Vector11D":
        return Vector11D(*(a - b for a, b in zip(self.coords, other.coords)))

    def __mul__(self, scalar: float) -> "Vector11D":
        return Vector11D(*(a * scalar for a in self.coords))

    def dot_product(self, other: "Vector11D") -> float:
        return sum(a * b for a, b in zip(self.coords, other.coords))

    def norm(self) -> float:
        return math.sqrt(sum(c * c for c in self.coords))

    def distance(self, other: "Vector11D") -> float:
        return (self - other).norm()

    def cosine_similarity(self, other: "Vector11D") -> float:
        denom = self.norm() * other.norm()
        if denom == 0:
            return 0.0
        return self.dot_product(other) / denom

    def project_onto_sphere(self, radius: float) -> "Vector11D":
        current_norm = self.norm()
        if current_norm == 0:
            return Vector11D(*([0.0] * 11))
        return self * (radius / current_norm)

    def sacred_signature(self) -> Dict[str, float]:
        """Return simple geometric features useful for lightweight AI scoring."""
        abs_sum = sum(abs(c) for c in self.coords)
        return {
            "norm": self.norm(),
            "energy": sum(c * c for c in self.coords),
            "balance": 0.0 if abs_sum == 0 else sum(self.coords) / abs_sum,
        }


def _regular_polygon_vertices(sides: int, radius: float = 1.0) -> List[Point2D]:
    if sides < 3:
        raise ValueError("A shape must have at least 3 sides")

    step = (2 * math.pi) / sides
    return [
        (radius * math.cos(i * step), radius * math.sin(i * step))
        for i in range(sides)
    ]


def generate_fractal_points(
    shape: str,
    iterations: int,
    *,
    seed: int = 7,
    contraction: float = 0.5,
) -> List[Point2D]:
    """Generate 2D chaos-game points from sacred-geometry seed shapes.

    Supported shape names: ``triangle``, ``square``, ``pentagon``, ``hexagon``.
    """

    shape_to_sides = {
        "triangle": 3,
        "square": 4,
        "pentagon": 5,
        "hexagon": 6,
    }
    if shape not in shape_to_sides:
        raise ValueError(f"Unsupported shape: {shape}")
    if iterations <= 0:
        return []
    if not 0 < contraction < 1:
        raise ValueError("contraction must be between 0 and 1")

    vertices = _regular_polygon_vertices(shape_to_sides[shape])
    rng = random.Random(seed)
    point = (0.0, 0.0)
    results: List[Point2D] = []

    for _ in range(iterations):
        vx, vy = rng.choice(vertices)
        px, py = point
        point = (
            px + contraction * (vx - px),
            py + contraction * (vy - py),
        )
        results.append(point)

    return results


class SacredGeometryAI:
    """A tiny prototype-based model for sacred-geometry labeling and scoring."""

    def __init__(self, prototypes: Dict[str, Vector11D] | None = None) -> None:
        self.prototypes = prototypes or {
            "tetrahedron": Vector11D(1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0),
            "cube": Vector11D(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0),
            "sphere": Vector11D(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        }

    def predict_shape(self, sample: Vector11D) -> Tuple[str, float]:
        best_name = "unknown"
        best_score = -1.0
        for name, prototype in self.prototypes.items():
            score = sample.cosine_similarity(prototype)
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, best_score

    def harmony_score(self, sample: Vector11D) -> float:
        """Return a bounded score in [0, 1] from vector balance + norm agreement."""
        shape, similarity = self.predict_shape(sample)
        _ = shape
        signature = sample.sacred_signature()

        balance_component = 1.0 - min(1.0, abs(signature["balance"]))
        norm_component = 1.0 / (1.0 + abs(signature["norm"] - 1.0))
        score = 0.6 * max(0.0, similarity) + 0.2 * balance_component + 0.2 * norm_component
        return max(0.0, min(1.0, score))


def build_feature_vector(values: Iterable[float]) -> Vector11D:
    """Utility to coerce input values into an 11D vector for model workflows."""
    values = list(values)
    if len(values) != 11:
        raise ValueError("Feature vectors must contain exactly 11 values")
    return Vector11D(*values)
