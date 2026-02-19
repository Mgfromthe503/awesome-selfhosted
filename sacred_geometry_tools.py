"""Sacred geometry visualization helpers.

This module provides educational geometry generators and plotting utilities.
The concepts are offered for creative/reflective use and should not be treated
as medical diagnosis or treatment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass
class Solid:
    name: str
    vertices: np.ndarray
    faces: List[List[int]]


def fibonacci_sequence(n: int) -> List[int]:
    """Generate the first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq


def golden_ratio_spiral(turns: float = 5.0, points: int = 1_000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate x, y coordinates for a logarithmic spiral using phi growth."""
    phi = (1 + math.sqrt(5)) / 2
    theta = np.linspace(0, 2 * np.pi * turns, points)
    b = math.log(phi) / (np.pi / 2)
    r = np.exp(b * theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _circle_points(radius: float, center: Point2D, samples: int = 180) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 2 * np.pi, samples)
    return center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)


def seed_of_life_centers(radius: float = 1.0) -> List[Point2D]:
    """Return centers for the seven circles in the Seed of Life."""
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * (np.pi / 3)
        centers.append((radius * np.cos(angle), radius * np.sin(angle)))
    return centers


def flower_of_life_centers(radius: float = 1.0, rings: int = 2) -> List[Point2D]:
    """Hexagonal lattice centers for a Flower of Life pattern."""
    centers = []
    for q in range(-rings, rings + 1):
        r1 = max(-rings, -q - rings)
        r2 = min(rings, -q + rings)
        for r in range(r1, r2 + 1):
            x = radius * (np.sqrt(3) * (q + r / 2))
            y = radius * (1.5 * r)
            centers.append((x, y))
    return centers


def vesica_piscis(radius: float = 1.0, samples: int = 400) -> Dict[str, np.ndarray]:
    """Create two equal circles with centers one radius apart."""
    c1 = (-radius / 2, 0.0)
    c2 = (radius / 2, 0.0)
    x1, y1 = _circle_points(radius, c1, samples)
    x2, y2 = _circle_points(radius, c2, samples)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def tree_of_life_points(scale: float = 1.0) -> Dict[str, Point2D]:
    """Simple 2D layout of the ten Sephirot."""
    base = {
        "Keter": (0, 4),
        "Chokhmah": (1.5, 3),
        "Binah": (-1.5, 3),
        "Chesed": (1.8, 2),
        "Gevurah": (-1.8, 2),
        "Tiferet": (0, 1.6),
        "Netzach": (1.2, 0.6),
        "Hod": (-1.2, 0.6),
        "Yesod": (0, -0.2),
        "Malkuth": (0, -1.4),
    }
    return {k: (v[0] * scale, v[1] * scale) for k, v in base.items()}


def metatrons_cube_points(radius: float = 1.0) -> List[Point2D]:
    """Use the 13-circle layout often used to construct Metatron's Cube."""
    points = flower_of_life_centers(radius=radius, rings=1)
    outer = []
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        outer.append((2 * radius * np.cos(angle), 2 * radius * np.sin(angle)))
    return points + outer


def sri_yantra_triangles(scale: float = 1.0) -> List[np.ndarray]:
    """Approximate interlocking triangles for Sri Yantra visualization."""
    triangles = []
    radii = np.linspace(1.0, 0.3, 9)
    for i, r in enumerate(radii):
        up = i % 2 == 0
        start = np.pi / 2 if up else -np.pi / 2
        verts = []
        for k in range(3):
            a = start + k * (2 * np.pi / 3)
            verts.append((scale * r * np.cos(a), scale * r * np.sin(a)))
        triangles.append(np.array(verts))
    return triangles


def infinity_symbol(a: float = 1.0, samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Lemniscate of Bernoulli parametric form."""
    t = np.linspace(0, 2 * np.pi, samples)
    x = a * np.cos(t) / (1 + np.sin(t) ** 2)
    y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
    return x, y


def labyrinth_path(turns: int = 7, samples: int = 2_000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single-path circular labyrinth-like spiral."""
    theta = np.linspace(0, 2 * np.pi * turns, samples)
    r = np.linspace(0.1, 1.0, samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def hexagram_points(radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    angles_up = np.array([np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3])
    angles_down = angles_up + np.pi
    up = np.column_stack((radius * np.cos(angles_up), radius * np.sin(angles_up)))
    down = np.column_stack((radius * np.cos(angles_down), radius * np.sin(angles_down)))
    return up, down


def cube() -> Solid:
    v = np.array(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ],
        dtype=float,
    )
    f = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
    return Solid("Cube", v, f)


def octahedron() -> Solid:
    v = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=float)
    f = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [0, 2, 5], [2, 1, 5], [1, 3, 5], [3, 0, 5]]
    return Solid("Octahedron", v, f)


def icosahedron() -> Solid:
    phi = (1 + np.sqrt(5)) / 2
    v = np.array(
        [
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
        dtype=float,
    )
    v /= np.linalg.norm(v, axis=1).max()
    f = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    return Solid("Icosahedron", v, f)


def torus(R: float = 2.0, r: float = 0.7, nu: int = 60, nv: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, 2 * np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    x = (R + r * np.cos(vv)) * np.cos(uu)
    y = (R + r * np.cos(vv)) * np.sin(uu)
    z = r * np.sin(vv)
    return x, y, z


def visualize_seed_of_life(ax: plt.Axes, radius: float = 1.0) -> None:
    for c in seed_of_life_centers(radius):
        x, y = _circle_points(radius, c)
        ax.plot(x, y, "b", lw=1)
    ax.set_aspect("equal")


def visualize_flower_of_life(ax: plt.Axes, radius: float = 1.0, rings: int = 2) -> None:
    for c in flower_of_life_centers(radius, rings):
        x, y = _circle_points(radius, c)
        ax.plot(x, y, "purple", lw=0.9)
    ax.set_aspect("equal")


def visualize_vesica_piscis(ax: plt.Axes, radius: float = 1.0) -> None:
    circles = vesica_piscis(radius)
    ax.plot(circles["x1"], circles["y1"], "g")
    ax.plot(circles["x2"], circles["y2"], "g")
    ax.set_aspect("equal")


def visualize_tree_of_life(ax: plt.Axes) -> None:
    points = tree_of_life_points()
    lines = [
        ("Keter", "Chokhmah"), ("Keter", "Binah"), ("Chokhmah", "Binah"),
        ("Chokhmah", "Chesed"), ("Binah", "Gevurah"), ("Chesed", "Tiferet"),
        ("Gevurah", "Tiferet"), ("Tiferet", "Netzach"), ("Tiferet", "Hod"),
        ("Netzach", "Yesod"), ("Hod", "Yesod"), ("Yesod", "Malkuth"),
    ]
    for a, b in lines:
        ax.plot([points[a][0], points[b][0]], [points[a][1], points[b][1]], "k-", lw=1)
    for name, (x, y) in points.items():
        ax.scatter([x], [y], s=35, c="gold")
        ax.text(x + 0.05, y + 0.05, name, fontsize=8)
    ax.set_aspect("equal")


def visualize_metatrons_cube(ax: plt.Axes, radius: float = 1.0) -> None:
    pts = metatrons_cube_points(radius)
    for c in pts:
        x, y = _circle_points(radius, c)
        ax.plot(x, y, color="steelblue", lw=0.7)
    for i, p1 in enumerate(pts):
        for p2 in pts[i + 1 :]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", lw=0.3, alpha=0.5)
    ax.set_aspect("equal")


def visualize_sri_yantra(ax: plt.Axes) -> None:
    for tri in sri_yantra_triangles():
        closed = np.vstack((tri, tri[0]))
        ax.plot(closed[:, 0], closed[:, 1], "maroon", lw=1)
    x, y = _circle_points(1.1, (0, 0))
    ax.plot(x, y, "maroon", lw=1)
    ax.set_aspect("equal")


def visualize_infinity(ax: plt.Axes) -> None:
    x, y = infinity_symbol()
    ax.plot(x, y, "darkblue")
    ax.set_aspect("equal")


def visualize_labyrinth(ax: plt.Axes) -> None:
    x, y = labyrinth_path()
    ax.plot(x, y, "darkgreen")
    ax.set_aspect("equal")


def visualize_hexagram(ax: plt.Axes) -> None:
    up, down = hexagram_points()
    up_c = np.vstack((up, up[0]))
    down_c = np.vstack((down, down[0]))
    ax.plot(up_c[:, 0], up_c[:, 1], "indigo")
    ax.plot(down_c[:, 0], down_c[:, 1], "indigo")
    ax.set_aspect("equal")


def _plot_solid(ax: plt.Axes, solid: Solid) -> None:
    polys = [[solid.vertices[idx] for idx in face] for face in solid.faces]
    collection = Poly3DCollection(polys, alpha=0.35, edgecolor="k")
    ax.add_collection3d(collection)
    ax.scatter(solid.vertices[:, 0], solid.vertices[:, 1], solid.vertices[:, 2], c="red", s=12)
    lim = 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title(solid.name)


def visualize_platonic_solids() -> None:
    solids = [cube(), octahedron(), icosahedron()]
    fig = plt.figure(figsize=(12, 4))
    for i, solid in enumerate(solids, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        _plot_solid(ax, solid)
    fig.suptitle("Platonic Solids (subset)")
    plt.tight_layout()


def visualize_cube(ax: plt.Axes) -> None:
    _plot_solid(ax, cube())


def visualize_octahedron(ax: plt.Axes) -> None:
    _plot_solid(ax, octahedron())


def visualize_icosahedron(ax: plt.Axes) -> None:
    _plot_solid(ax, icosahedron())


def visualize_torus(ax: plt.Axes) -> None:
    x, y, z = torus()
    ax.plot_surface(x, y, z, cmap="viridis", linewidth=0, alpha=0.85)


def demo_all(output_prefix: str = "sacred") -> None:
    """Render all requested 2D patterns and selected 3D plots to image files."""
    visuals = {
        "seed_of_life": visualize_seed_of_life,
        "flower_of_life": visualize_flower_of_life,
        "vesica_piscis": visualize_vesica_piscis,
        "tree_of_life": visualize_tree_of_life,
        "metatrons_cube": visualize_metatrons_cube,
        "sri_yantra": visualize_sri_yantra,
        "infinity": visualize_infinity,
        "labyrinth": visualize_labyrinth,
        "hexagram": visualize_hexagram,
    }

    for name, draw in visuals.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        draw(ax)
        ax.axis("off")
        fig.savefig(f"{output_prefix}_{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    visualize_cube(ax)
    fig.savefig(f"{output_prefix}_cube.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    visualize_octahedron(ax)
    fig.savefig(f"{output_prefix}_octahedron.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    visualize_icosahedron(ax)
    fig.savefig(f"{output_prefix}_icosahedron.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    visualize_torus(ax)
    fig.savefig(f"{output_prefix}_torus.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    visualize_platonic_solids()
    plt.show()
