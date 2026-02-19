"""Utilities for encoding sacred geometry terms from free-form text."""

from __future__ import annotations

import re
from typing import Mapping

GEOMETRY_CODES: Mapping[str, str] = {
    "circle": "001",
    "triangle": "002",
    "square": "003",
    "pentagon": "004",
    "hexagon": "005",
}

SACRED_GEOMETRY_VALUES: Mapping[str, int] = {
    "flower of life": 1,
    "metatron's cube": 2,
    "sri yantra": 3,
    "seed of life": 4,
    "fruit of life": 5,
}


def sacred_geometry_code(input_text: str, *, geometry_dict: Mapping[str, str] | None = None) -> str:
    """Return concatenated 3-digit codes for any known shapes in ``input_text``.

    The parser is case-insensitive and ignores punctuation.
    """

    shape_codes = geometry_dict or GEOMETRY_CODES
    normalized = re.sub(r"[^a-zA-Z\s]", " ", input_text).lower()
    return "".join(shape_codes[word] for word in normalized.split() if word in shape_codes)


def sacred_symbol_value(symbol: str, *, symbol_dict: Mapping[str, int] | None = None) -> int:
    """Lookup a sacred geometry symbol value in a case-insensitive way."""

    values = symbol_dict or SACRED_GEOMETRY_VALUES
    key = symbol.strip().lower()
    if key not in values:
        raise KeyError(f"Unknown sacred geometry symbol: {symbol}")
    return values[key]


if __name__ == "__main__":
    input_text = "Draw a circle around the square and connect it to the triangle"
    print(sacred_geometry_code(input_text))
