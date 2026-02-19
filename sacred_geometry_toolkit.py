"""Sacred geometry toolkit.

This module provides educational geometry and data-visualization utilities inspired by
sacred-geometry motifs. It is not a medical device and does not provide clinical advice.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

Vector3 = Tuple[float, float, float]


SHAPE_KNOWLEDGE_BASE: Dict[str, Dict[str, object]] = {
    "point": {"family": "primitive", "dimension": 0, "sides": 0, "meaning": "origin and unity"},
    "line": {"family": "primitive", "dimension": 1, "sides": 1, "meaning": "direction and connection"},
    "circle": {"family": "planar", "dimension": 2, "sides": 0, "meaning": "wholeness and cycles"},
    "ellipse": {"family": "planar", "dimension": 2, "sides": 0, "meaning": "orbital balance"},
    "vesica piscis": {"family": "sacred", "dimension": 2, "sides": 0, "meaning": "duality and emergence"},
    "triangle": {"family": "polygon", "dimension": 2, "sides": 3, "meaning": "triadic stability"},
    "square": {"family": "polygon", "dimension": 2, "sides": 4, "meaning": "structure and order"},
    "rectangle": {"family": "polygon", "dimension": 2, "sides": 4, "meaning": "grounded extension"},
    "pentagon": {"family": "polygon", "dimension": 2, "sides": 5, "meaning": "organic growth"},
    "hexagon": {"family": "polygon", "dimension": 2, "sides": 6, "meaning": "harmony and efficiency"},
    "heptagon": {"family": "polygon", "dimension": 2, "sides": 7, "meaning": "mystic alignment"},
    "octagon": {"family": "polygon", "dimension": 2, "sides": 8, "meaning": "transition and balance"},
    "enneagon": {"family": "polygon", "dimension": 2, "sides": 9, "meaning": "completion cycle"},
    "decagon": {"family": "polygon", "dimension": 2, "sides": 10, "meaning": "structured expansion"},
    "hendecagon": {"family": "polygon", "dimension": 2, "sides": 11, "meaning": "threshold patterning"},
    "dodecagon": {"family": "polygon", "dimension": 2, "sides": 12, "meaning": "cosmic order"},
    "hexagram": {"family": "star", "dimension": 2, "sides": 6, "meaning": "union of opposites"},
    "pentagram": {"family": "star", "dimension": 2, "sides": 5, "meaning": "proportional life-force"},
    "star tetrahedron": {"family": "star", "dimension": 3, "sides": 8, "meaning": "dynamic polarity"},
    "seed of life": {"family": "sacred", "dimension": 2, "sides": 0, "meaning": "creation template"},
    "egg of life": {"family": "sacred", "dimension": 3, "sides": 0, "meaning": "developmental matrix"},
    "flower of life": {"family": "sacred", "dimension": 2, "sides": 0, "meaning": "unified field pattern"},
    "fruit of life": {"family": "sacred", "dimension": 3, "sides": 0, "meaning": "informational lattice"},
    "metatron's cube": {"family": "sacred", "dimension": 3, "sides": 0, "meaning": "platonic connectivity"},
    "sri yantra": {"family": "sacred", "dimension": 2, "sides": 0, "meaning": "concentric consciousness"},
    "torus": {"family": "surface", "dimension": 3, "sides": 0, "meaning": "self-renewing flow"},
    "sphere": {"family": "surface", "dimension": 3, "sides": 0, "meaning": "perfect symmetry"},
    "tetrahedron": {"family": "platonic", "dimension": 3, "sides": 4, "meaning": "fire and initiation"},
    "cube": {"family": "platonic", "dimension": 3, "sides": 6, "meaning": "earth and stability"},
    "octahedron": {"family": "platonic", "dimension": 3, "sides": 8, "meaning": "air and balance"},
    "dodecahedron": {"family": "platonic", "dimension": 3, "sides": 12, "meaning": "ether and synthesis"},
    "icosahedron": {"family": "platonic", "dimension": 3, "sides": 20, "meaning": "water and flow"},
    "merkaba": {"family": "sacred", "dimension": 3, "sides": 8, "meaning": "field propulsion metaphor"},
    "fibonacci spiral": {"family": "growth", "dimension": 2, "sides": 0, "meaning": "iterative scaling"},
    "golden spiral": {"family": "growth", "dimension": 2, "sides": 0, "meaning": "phi resonance"},
    "golden rectangle": {"family": "growth", "dimension": 2, "sides": 4, "meaning": "harmonic proportion"},
    "phi grid": {"family": "growth", "dimension": 2, "sides": 0, "meaning": "ratio scaffolding"},
    "mandala": {"family": "sacred", "dimension": 2, "sides": 0, "meaning": "centered complexity"},
    "labyrinth": {"family": "path", "dimension": 2, "sides": 0, "meaning": "guided traversal"},
    "fractal tree": {"family": "fractal", "dimension": 2, "sides": 0, "meaning": "recursive branching"},
    "koch snowflake": {"family": "fractal", "dimension": 2, "sides": 0, "meaning": "infinite boundary"},
    "sierpinski triangle": {"family": "fractal", "dimension": 2, "sides": 3, "meaning": "nested self-similarity"},
}


SYMBOLIC_KNOWLEDGE_BASE: Dict[str, Dict[str, object]] = {
    "flower of life": {"domain": "cosmology", "value": 1, "keywords": ["creation", "unity", "interconnection"]},
    "metatron's cube": {"domain": "geometry", "value": 2, "keywords": ["platonic", "network", "symmetry"]},
    "sri yantra": {"domain": "meditation", "value": 3, "keywords": ["focus", "balance", "convergence"]},
    "seed of life": {"domain": "origins", "value": 4, "keywords": ["seed", "birth", "potential"]},
    "fruit of life": {"domain": "structure", "value": 5, "keywords": ["field", "nodes", "connections"]},
    "merkaba": {"domain": "field-dynamics", "value": 6, "keywords": ["motion", "spin", "duality"]},
    "vesica piscis": {"domain": "duality", "value": 7, "keywords": ["intersection", "bridge", "emergence"]},
    "golden ratio": {"domain": "proportion", "value": 8, "keywords": ["phi", "aesthetics", "scaling"]},
    "fibonacci spiral": {"domain": "growth", "value": 9, "keywords": ["sequence", "expansion", "nature"]},
    "platonic solids": {"domain": "3d-geometry", "value": 10, "keywords": ["tetra", "cube", "dodecahedron"]},
}


ASTRO_SHAPE_MAPPING: Dict[int, str] = {
    1: "circle",
    2: "vesica piscis",
    3: "triangle",
    4: "square",
    5: "pentagon",
    6: "hexagon",
    7: "heptagon",
    8: "octagon",
    9: "enneagon",
    10: "decagon",
    11: "hendecagon",
    12: "dodecagon",
    13: "flower of life",
    14: "metatron's cube",
    15: "sri yantra",
    16: "seed of life",
    17: "fruit of life",
    18: "hexagram",
    19: "pentagram",
    20: "torus",
    21: "tetrahedron",
    22: "cube",
    23: "octahedron",
    24: "dodecahedron",
    25: "icosahedron",
    26: "merkaba",
    27: "fibonacci spiral",
    28: "golden spiral",
    29: "golden rectangle",
    30: "phi grid",
    31: "mandala",
    32: "labyrinth",
    33: "fractal tree",
    34: "koch snowflake",
    35: "sierpinski triangle",
    36: "star tetrahedron",
}


def text_to_sacred_geometry_codes(text: str) -> List[Dict[str, float | str]]:
    """Convert text to symbolic geometry codes.

    Each character is mapped into a tuple of:
    - reduced ordinal value (digital root style, 1-9)
    - angle in degrees for radial placement
    - harmonic ratio tied to the golden ratio
    """
    phi = (1 + math.sqrt(5)) / 2
    codes: List[Dict[str, float | str]] = []
    for ch in text:
        if ch.isspace():
            continue
        ordinal = ord(ch)
        digital_root = 1 + ((ordinal - 1) % 9)
        angle = (ordinal * 137.50776405) % 360
        harmonic = round((digital_root / 9.0) * phi, 6)
        codes.append(
            {
                "char": ch,
                "code": f"SG-{digital_root}-{int(angle):03d}",
                "angle_deg": round(angle, 3),
                "harmonic": harmonic,
            }
        )
    return codes


def _normalize(vertices: Iterable[Vector3], scale: float) -> List[Vector3]:
    return [(x * scale, y * scale, z * scale) for x, y, z in vertices]


def generate_platonic_solid(solid: str, scale: float = 1.0) -> Dict[str, List]:
    """Generate vertices and faces for Platonic solids.

    Supported solids: tetrahedron, cube, octahedron, dodecahedron, icosahedron.
    """
    solid = solid.lower().strip()
    phi = (1 + math.sqrt(5)) / 2

    presets: Dict[str, Dict[str, List]] = {
        "tetrahedron": {
            "vertices": [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)],
            "faces": [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
        },
        "cube": {
            "vertices": [
                (-1, -1, -1),
                (1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, 1, 1),
            ],
            "faces": [
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            ],
        },
        "octahedron": {
            "vertices": [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
            "faces": [
                (0, 2, 4),
                (2, 1, 4),
                (1, 3, 4),
                (3, 0, 4),
                (0, 2, 5),
                (2, 1, 5),
                (1, 3, 5),
                (3, 0, 5),
            ],
        },
        "icosahedron": {
            "vertices": [
                (-1, phi, 0),
                (1, phi, 0),
                (-1, -phi, 0),
                (1, -phi, 0),
                (0, -1, phi),
                (0, 1, phi),
                (0, -1, -phi),
                (0, 1, -phi),
                (phi, 0, -1),
                (phi, 0, 1),
                (-phi, 0, -1),
                (-phi, 0, 1),
            ],
            "faces": [
                (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
            ],
        },
    }

    dodeca_vertices = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                dodeca_vertices.append((x, y, z))
    for a in (-1 / phi, 1 / phi):
        for b in (-phi, phi):
            dodeca_vertices.extend([(0, a, b), (a, b, 0), (b, 0, a)])
    presets["dodecahedron"] = {"vertices": dodeca_vertices, "faces": []}

    if solid not in presets:
        raise ValueError(f"Unsupported solid '{solid}'.")

    data = presets[solid]
    return {
        "solid": solid,
        "vertices": _normalize(data["vertices"], scale),
        "faces": data["faces"],
    }


def translate_astrological_numbers(numbers: Sequence[int]) -> List[Dict[str, str | int]]:
    """Translate chart numbers into symbolic geometry shapes."""
    if not numbers:
        return []
    cycle = max(ASTRO_SHAPE_MAPPING.keys())
    return [{"number": n, "shape": ASTRO_SHAPE_MAPPING[((n - 1) % cycle) + 1]} for n in numbers]


def list_supported_shapes() -> List[str]:
    """Return all known geometry shape tokens sorted alphabetically."""
    return sorted(SHAPE_KNOWLEDGE_BASE.keys())


def get_symbolic_profile(symbol: str) -> Dict[str, object]:
    """Return a normalized symbolic record for sacred terms and shapes."""
    key = symbol.strip().lower()
    if key in SYMBOLIC_KNOWLEDGE_BASE:
        profile = dict(SYMBOLIC_KNOWLEDGE_BASE[key])
        profile["symbol"] = key
        profile["type"] = "symbol"
        return profile
    if key in SHAPE_KNOWLEDGE_BASE:
        profile = dict(SHAPE_KNOWLEDGE_BASE[key])
        profile["symbol"] = key
        profile["type"] = "shape"
        return profile
    raise KeyError(f"Unknown sacred symbol: {symbol}")


def generate_symbolic_training_records() -> List[Dict[str, object]]:
    """Build deterministic records for model fine-tuning and eval scaffolding."""
    records: List[Dict[str, object]] = []
    for name in list_supported_shapes():
        shape = SHAPE_KNOWLEDGE_BASE[name]
        prompt = f"Classify sacred geometry concept '{name}'"
        completion = (
            f"family={shape['family']};dimension={shape['dimension']};"
            f"sides={shape['sides']};meaning={shape['meaning']}"
        )
        records.append({"prompt": prompt, "completion": completion, "source": "shape_knowledge"})
    for name, payload in sorted(SYMBOLIC_KNOWLEDGE_BASE.items()):
        prompt = f"Explain sacred symbol '{name}'"
        completion = (
            f"domain={payload['domain']};value={payload['value']};"
            f"keywords={','.join(payload['keywords'])}"
        )
        records.append({"prompt": prompt, "completion": completion, "source": "symbol_knowledge"})
    return records


def seed_of_life_visualization(elements: Dict[str, float]) -> Dict[str, List[Dict[str, float | str]]]:
    """Build Seed-of-Life radial coordinates for weighted elements."""
    labels = list(elements.keys())
    if not labels:
        return {"nodes": [], "circles": []}

    circles = [{"x": 0.0, "y": 0.0, "r": 1.0, "label": "center"}]
    step = (2 * math.pi) / 6
    for i in range(6):
        circles.append({"x": math.cos(i * step), "y": math.sin(i * step), "r": 1.0, "label": f"ring_{i+1}"})

    nodes = []
    for idx, label in enumerate(labels):
        angle = (2 * math.pi * idx) / len(labels)
        radius = 1.6 + min(max(elements[label], 0.0), 1.0)
        nodes.append({"label": label, "x": radius * math.cos(angle), "y": radius * math.sin(angle), "weight": elements[label]})

    return {"nodes": nodes, "circles": circles}


def optimize_sacred_layout(block_ids: Sequence[str], width: float, height: float) -> List[Dict[str, float | str]]:
    """Place blocks using a golden-angle spiral for balanced web layout prototyping."""
    cx, cy = width / 2.0, height / 2.0
    golden_angle = math.radians(137.50776405)
    placements = []
    for i, block in enumerate(block_ids):
        radius = math.sqrt(i + 1) * min(width, height) * 0.08
        angle = i * golden_angle
        x = min(max(cx + radius * math.cos(angle), 0), width)
        y = min(max(cy + radius * math.sin(angle), 0), height)
        placements.append({"id": block, "x": round(x, 2), "y": round(y, 2)})
    return placements


def calculate_resonance_frequency(shape: str, dimension: float, wave_speed: float = 343.0) -> float:
    """Estimate a resonance frequency from characteristic path length.

    f = wave_speed / wavelength, where wavelength is approximated as perimeter length.
    """
    shape = shape.lower()
    if dimension <= 0:
        raise ValueError("dimension must be positive")

    perimeter_factors = {
        "circle": 2 * math.pi,
        "triangle": 3,
        "square": 4,
        "pentagon": 5,
        "hexagon": 6,
        "heptagon": 7,
        "octagon": 8,
        "enneagon": 9,
        "decagon": 10,
        "dodecagon": 12,
    }
    factor = perimeter_factors.get(shape)
    if factor is None:
        raise ValueError(f"Unsupported shape '{shape}'")

    wavelength = factor * dimension
    return round(wave_speed / wavelength, 6)


@dataclass
class SacredGeometryModel:
    """Simple mutable 3D model for real-time style manipulations."""

    vertices: List[Vector3]

    @classmethod
    def from_platonic(cls, solid: str, scale: float = 1.0) -> "SacredGeometryModel":
        data = generate_platonic_solid(solid, scale)
        return cls(vertices=list(data["vertices"]))

    def scale(self, factor: float) -> None:
        self.vertices = [(x * factor, y * factor, z * factor) for x, y, z in self.vertices]

    def translate(self, dx: float, dy: float, dz: float) -> None:
        self.vertices = [(x + dx, y + dy, z + dz) for x, y, z in self.vertices]

    def rotate_z(self, degrees: float) -> None:
        rad = math.radians(degrees)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        self.vertices = [
            (x * cos_a - y * sin_a, x * sin_a + y * cos_a, z) for x, y, z in self.vertices
        ]


def analyze_flower_of_life_brainwave(brainwaves: Dict[str, float]) -> Dict[str, float | str]:
    """Compute non-diagnostic symmetry metrics from normalized brainwave bands."""
    order = ["delta", "theta", "alpha", "beta", "gamma"]
    vals = [max(brainwaves.get(b, 0.0), 0.0) for b in order]
    total = sum(vals) or 1.0
    norm = [v / total for v in vals]
    symmetry = 1.0 - abs(norm[0] - norm[-1]) - abs(norm[1] - norm[-2])
    coherence = 1.0 - sum(abs(norm[i] - norm[i - 1]) for i in range(1, len(norm))) / 4
    return {
        "symmetry_score": round(max(0.0, symmetry), 6),
        "coherence_score": round(max(0.0, coherence), 6),
        "note": "Research-only exploratory output; not a diagnosis or treatment recommendation.",
    }


def metatrons_cube_dna_analysis(sequence: str) -> Dict[str, object]:
    """Map DNA nucleotides to Metatron-style node connectivity."""
    seq = ''.join(base for base in sequence.upper() if base in "ATCG")
    mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
    nodes = [{"index": i, "base": b, "node": mapping[b]} for i, b in enumerate(seq)]
    edges = [{"from": i, "to": i + 1, "type": "backbone"} for i in range(max(0, len(nodes) - 1))]
    for i in range(len(nodes) - 3):
        if nodes[i]["base"] == nodes[i + 3]["base"]:
            edges.append({"from": i, "to": i + 3, "type": "resonance"})
    return {"clean_sequence": seq, "nodes": nodes, "edges": edges}


def design_ai_energy_healing_device(requirements: Dict[str, str]) -> Dict[str, object]:
    """Create a conceptual system blueprint.

    This returns a software architecture concept only and must not be treated as a
    validated medical device design.
    """
    return {
        "intent": requirements.get("intent", "wellness exploration"),
        "modules": [
            "sensor_ingestion",
            "signal_preprocessing",
            "geometry_feature_encoder",
            "personalization_model",
            "safety_constraints",
            "visual_feedback_ui",
        ],
        "safety": [
            "non-clinical-use-only",
            "human-oversight-required",
            "regulatory-review-required-before-deployment",
        ],
    }


if __name__ == "__main__":
    sample = "Harmony"
    print("Text codes:", text_to_sacred_geometry_codes(sample))
    print("Cube vertex count:", len(generate_platonic_solid("cube")["vertices"]))
    print("Supported shapes:", len(list_supported_shapes()))
    print("Training records:", len(generate_symbolic_training_records()))
