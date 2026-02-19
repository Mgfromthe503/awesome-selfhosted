"""Sacred geometry coding helpers and a production-friendly 7D vector model.

This module turns the user-provided sacred geometry mappings and 7D math notes
into reusable Python code suitable for deployment-oriented software.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping

# Canonical sacred-geometry mappings from the provided specification.
GEOMETRY_DICT: Mapping[str, str] = {
    "Seed of Life": "SOL",
    "Flower of Life": "FOL",
    "Metatron's Cube": "MC",
    "Sri Yantra": "SY",
    "Vesica Pisces": "VP",
    "Platonic Solids": "PS",
    "Golden Ratio": "GR",
    "Torus": "T",
    "Merkaba": "M",
    "Tree of Life": "TOL",
}

SYMBOL_CODES: Mapping[str, str] = {
    "circle": "C1",
    "triangle": "T1",
    "square": "S1",
    "pentagon": "P1",
    "hexagon": "H1",
    "seed_of_life": "SOL1",
    "flower_of_life": "FOL1",
    "metatrons_cube": "MC1",
}

CODING_SYSTEM: Mapping[str, str] = {
    "Seed of Life": "01",
    "Flower of Life": "02",
    "Metatron's Cube": "03",
    "Sri Yantra": "04",
    "Tree of Life": "05",
}


def extract_keywords(input_text: str) -> List[str]:
    """Extract supported sacred-geometry terms from free text.

    Returns canonical keys used by ``CODING_SYSTEM``.
    """

    lowered = input_text.lower()
    aliases = {
        "seed of life": "Seed of Life",
        "flower of life": "Flower of Life",
        "metatron's cube": "Metatron's Cube",
        "metatrons cube": "Metatron's Cube",
        "sri yantra": "Sri Yantra",
        "tree of life": "Tree of Life",
    }
    return [canonical for alias, canonical in aliases.items() if alias in lowered]


def map_to_symbol(input_text: str) -> List[str]:
    """Map natural-language text to sacred-geometry numeric codes."""

    keywords = extract_keywords(input_text)
    return [CODING_SYSTEM[key] for key in keywords if key in CODING_SYSTEM]


@dataclass(frozen=True)
class Vector7D:
    """A strict 7-dimensional real vector with common ML-friendly operations."""

    coords: tuple[float, float, float, float, float, float, float]

    def __init__(self, *coords: float):
        if len(coords) != 7:
            raise ValueError("Vector7D requires exactly 7 coordinates")
        object.__setattr__(self, "coords", tuple(float(value) for value in coords))

    @classmethod
    def from_list(cls, values: Iterable[float]) -> "Vector7D":
        values_tuple = tuple(values)
        return cls(*values_tuple)

    def to_list(self) -> List[float]:
        return list(self.coords)

    def __str__(self) -> str:
        return "(" + ", ".join(f"{coord:g}" for coord in self.coords) + ")"

    def __getitem__(self, index: int) -> float:
        return self.coords[index]

    def __add__(self, other: "Vector7D") -> "Vector7D":
        return Vector7D(*(a + b for a, b in zip(self.coords, other.coords)))

    def __sub__(self, other: "Vector7D") -> "Vector7D":
        return Vector7D(*(a - b for a, b in zip(self.coords, other.coords)))

    def __mul__(self, scalar: float) -> "Vector7D":
        return Vector7D(*(a * scalar for a in self.coords))

    def __rmul__(self, scalar: float) -> "Vector7D":
        return self.__mul__(scalar)

    def dot_product(self, other: "Vector7D") -> float:
        return sum(a * b for a, b in zip(self.coords, other.coords))

    def norm(self) -> float:
        return math.sqrt(sum(coord**2 for coord in self.coords))

    def distance(self, other: "Vector7D") -> float:
        return (self - other).norm()

    def cosine_similarity(self, other: "Vector7D") -> float:
        left_norm = self.norm()
        right_norm = other.norm()
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return self.dot_product(other) / (left_norm * right_norm)

    def normalize(self) -> "Vector7D":
        magnitude = self.norm()
        if magnitude == 0:
            return Vector7D(*(0.0 for _ in range(7)))
        return Vector7D(*(coord / magnitude for coord in self.coords))

    def project_onto(self, basis: Iterable["Vector7D"]) -> "Vector7D":
        """Project this vector onto the span of the provided basis vectors.

        The basis is assumed to be orthonormal. Non-orthonormal support could be
        added later through Gram-Schmidt or least-squares projection.
        """

        total = Vector7D(*(0.0 for _ in range(7)))
        for vector in basis:
            total = total + (self.dot_product(vector) * vector)
        return total
