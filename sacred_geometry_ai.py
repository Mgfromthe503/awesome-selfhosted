"""Utilities for experimenting with sacred-geometry-inspired vector operations.

This module intentionally keeps dependencies minimal while providing:
- A fixed-size 11D vector representation.
- Projection of a vector onto a sphere.
- Fractal point generation for common theoretical sacred-geometry seed shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, List, Sequence, Tuple

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class Vector11D:
    """Simple immutable 11-dimensional vector."""

    coords: Tuple[float, ...]

    def __init__(self, *coords: float):
        if len(coords) != 11:
            raise ValueError("Vector11D requires exactly 11 coordinates.")
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
        return Vector11D(*(c * scalar for c in self.coords))

    def dot_product(self, other: "Vector11D") -> float:
        return sum(a * b for a, b in zip(self.coords, other.coords))

    def norm(self) -> float:
        return math.sqrt(sum(c * c for c in self.coords))

    def distance(self, other: "Vector11D") -> float:
        return (self - other).norm()

    def cosine_similarity(self, other: "Vector11D") -> float:
        denominator = self.norm() * other.norm()
        if denominator == 0:
            return 0.0
        return self.dot_product(other) / denominator

    def project_onto_sphere(self, radius: float) -> "Vector11D":
        if radius < 0:
            raise ValueError("radius must be non-negative")
        magnitude = self.norm()
        if magnitude == 0:
            return Vector11D(*([0.0] * 11))
        scale = radius / magnitude
        return self * scale

    def fractal_nature_points(
        self,
        shape: str,
        iterations: int,
        *,
        seed_point: Point2D = (0.0, 0.0),
    ) -> List[Point2D]:
        """Generate fractal points from a sacred-geometry seed shape.

        Supported shapes:
        - ``triangle`` (Sierpinski triangle anchors)
        - ``square`` (carpet-like anchors)
        - ``pentagon`` (golden-ratio inspired anchors)
        - ``hexagon`` (flower-of-life style anchors)
        """
        if iterations < 0:
            raise ValueError("iterations must be >= 0")

        anchors = _shape_anchors(shape)
        if not anchors:
            raise ValueError(f"Unsupported shape: {shape}")

        point = seed_point
        points: List[Point2D] = [point]

        # Deterministic sequence derived from vector coordinates.
        for i in range(max(1, iterations * len(anchors))):
            anchor_index = int(abs(self.coords[i % 11]) + i) % len(anchors)
            anchor = anchors[anchor_index]
            point = ((point[0] + anchor[0]) / 2.0, (point[1] + anchor[1]) / 2.0)
            points.append(point)

        return points


def _regular_polygon_vertices(sides: int, radius: float = 1.0) -> List[Point2D]:
    return [
        (
            radius * math.cos((2 * math.pi * i) / sides),
            radius * math.sin((2 * math.pi * i) / sides),
        )
        for i in range(sides)
    ]


def _shape_anchors(shape: str) -> List[Point2D]:
    normalized = shape.strip().lower()
    if normalized == "triangle":
        return [(0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2)]
    if normalized == "square":
        return [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    if normalized == "pentagon":
        return _regular_polygon_vertices(5)
    if normalized == "hexagon":
        return _regular_polygon_vertices(6)
    return []
