"""Deterministic, dependency-free hollow metal part assets, in metres.

Visuals are closed triangle meshes centered vertically at the origin. Bores are
polygonal (64 sides for bushings); gear teeth use a straight-sided approximation,
not an involute. Collision meshes are exact convex angular wedges, split at the
flange height where necessary. They have disjoint interiors and leave the bore
open. Mass uses the actual polygonal mesh volume, not ideal-cylinder volume.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import struct
import zlib


def _sections(spec):
    """Return angular outer radii and height intervals (local, centered z)."""
    family = spec["family"]
    if family not in ("bushing", "flanged_bushing", "gear"):
        raise ValueError("family must be bushing, flanged_bushing, or gear")
    outer = float(spec["outer_radius_m"])
    inner = float(spec["inner_radius_m"])
    height = float(spec["height_m"])
    if not all(math.isfinite(v) and v > 0 for v in (outer, inner, height)):
        raise ValueError("radii and height must be finite and positive")
    if inner >= outer:
        raise ValueError("inner_radius_m must be less than outer_radius_m")
    if family == "gear":
        teeth = spec["teeth"]
        if isinstance(teeth, bool) or not isinstance(teeth, int) or teeth < 3:
            raise ValueError("teeth must be an integer of at least three")
        root = outer * 0.82
        if inner >= root * math.cos(math.pi / (2 * teeth)):
            raise ValueError("bore intersects the gear root")
        radii = [root, outer, outer, root] * teeth
    else:
        radii = [outer] * 64
    lo, hi = -height / 2, height / 2
    if family == "flanged_bushing":
        flange_r = float(spec["flange_radius_m"])
        flange_h = float(spec["flange_height_m"])
        if not math.isfinite(flange_r) or flange_r <= outer:
            raise ValueError("flange_radius_m must exceed outer_radius_m")
        if not math.isfinite(flange_h) or not 0 < flange_h < height:
            raise ValueError("flange_height_m must be between zero and height")
        return inner, radii, [(lo, lo + flange_h, [flange_r] * 64),
                              (lo + flange_h, hi, radii)]
    return inner, radii, [(lo, hi, radii)]


def _annular_area(inner, radii):
    n = len(radii)
    sine = math.sin(2 * math.pi / n)
    return 0.5 * sine * (sum(radii[i] * radii[(i + 1) % n]
                             for i in range(n)) - n * inner * inner)


def _volume(spec):
    inner, _, sections = _sections(spec)
    return sum((hi - lo) * _annular_area(inner, radii)
               for lo, hi, radii in sections)


def generate_specs(count=50, seed=7):
    """Return reproducible dimension/material specifications with unique IDs.

    Families cycle so all three are represented for count >= 3. mass_scale is a
    small independent inertia/density uncertainty, not a change to mesh geometry.
    Density is either aluminium-like 2700 or steel-like 7850 kg/m^3.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    rng = random.Random(seed)
    specs = []
    for i in range(count):
        family = ("bushing", "flanged_bushing", "gear")[i % 3]
        outer = round(rng.uniform(0.016, 0.028), 7)
        height = round(rng.uniform(0.014, 0.028), 7)
        dynamic = round(rng.uniform(0.15, 0.45), 5)
        spec = {
            "object_id": f"part_{i:03d}",
            "family": family,
            "outer_radius_m": outer,
            "inner_radius_m": round(rng.uniform(0.005, 0.010), 7),
            "height_m": height,
            "density_kg_m3": rng.choice((2700, 7850)),
            "mass_scale": round(rng.uniform(0.95, 1.05), 6),
            "static_friction": round(dynamic + rng.uniform(0.05, 0.20), 5),
            "dynamic_friction": dynamic,
            "restitution": round(rng.uniform(0.0, 0.1), 5),
            "metallic": round(rng.uniform(0.95, 1.0), 5),
            "roughness": round(rng.uniform(0.06, 0.35), 5),
        }
        if family == "flanged_bushing":
            spec["flange_radius_m"] = round(outer * rng.uniform(1.15, 1.35), 7)
            spec["flange_height_m"] = round(height * rng.uniform(0.20, 0.32), 7)
        if family == "gear":
            spec["teeth"] = rng.randint(12, 24)
        spec["volume_m3"] = _volume(spec)
        spec["mass_kg"] = (spec["volume_m3"] * spec["density_kg_m3"]
                           * spec["mass_scale"])
        specs.append(spec)
    return specs


def _convex_wedge(inner, ra, rb, angle_a, angle_b, lo, hi):
    """Extrude one convex annular trapezoid; return outward triangles."""
    ca, sa = math.cos(angle_a), math.sin(angle_a)
    cb, sb = math.cos(angle_b), math.sin(angle_b)
    planar = [(inner * ca, inner * sa), (ra * ca, ra * sa),
              (rb * cb, rb * sb), (inner * cb, inner * sb)]
    points = [(x, y, z) for z in (lo, hi) for x, y in planar]
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7)]
    for a in range(4):
        b = (a + 1) % 4
        faces.extend(((a, b, b + 4), (a, b + 4, a + 4)))
    return {"points": points, "faces": faces}


def build_geometry(spec):
    """Build visual and convex collider meshes with face-varying UVs.

    Keys: points (xyz triples), faces (index triples), uv (one uv pair per face
    corner), colliders (list of points/faces dictionaries), volume_m3 (float).
    The top/bottom lie at +/-height_m/2; the flange is at the bottom.
    """
    inner, outer_radii, sections = _sections(spec)
    n = len(outer_radii)
    lo, hi = sections[0][0], sections[-1][1]
    inner_radii = [inner] * n
    if spec["family"] == "flanged_bushing":
        mid = sections[0][1]
        flange_radii = sections[0][2]
        profile = [(inner_radii, lo), (flange_radii, lo),
                   (flange_radii, mid), (outer_radii, mid),
                   (outer_radii, hi), (inner_radii, hi)]
    else:
        profile = [(inner_radii, lo), (outer_radii, lo),
                   (outer_radii, hi), (inner_radii, hi)]
    points = [(radii[i] * math.cos(2 * math.pi * i / n),
               radii[i] * math.sin(2 * math.pi * i / n), z)
              for radii, z in profile for i in range(n)]
    faces, uv = [], []
    m = len(profile)
    for j in range(m):
        jj = (j + 1) % m
        for i in range(n):
            ii = (i + 1) % n
            a, b, c, d = j * n + i, j * n + ii, jj * n + ii, jj * n + i
            faces.extend(((a, b, c), (a, c, d)))
            u0, u1, v0, v1 = i / n, (i + 1) / n, j / m, (j + 1) / m
            uv.extend(((u0, v0), (u1, v0), (u1, v1),
                       (u0, v0), (u1, v1), (u0, v1)))
    colliders = []
    for bottom, top, radii in sections:
        for i in range(n):
            colliders.append(_convex_wedge(
                inner, radii[i], radii[(i + 1) % n],
                2 * math.pi * i / n, 2 * math.pi * (i + 1) / n,
                bottom, top))
    return {"points": points, "faces": faces, "uv": uv,
            "colliders": colliders, "volume_m3": _volume(spec)}


def write_obj(path, geometry):
    """Write a triangulated OBJ with explicit per-corner texture coordinates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# binpick_sdg; units metres; Z up; centered vertically",
             "# Straight-sided approximate gears, not involute tooth profiles."]
    lines.extend("v " + " ".join(f"{v:.12g}" for v in p)
                 for p in geometry["points"])
    lines.extend("vt " + " ".join(f"{v:.12g}" for v in p)
                 for p in geometry["uv"])
    lines.append("s off")
    for i, face in enumerate(geometry["faces"]):
        lines.append("f " + " ".join(f"{v + 1}/{3 * i + k + 1}"
                                      for k, v in enumerate(face)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _png_chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def write_roughness_png(path, seed, base_roughness, size=128):
    """Write a deterministic grayscale roughness map (linear data, no gamma).

    Scratches are directional roughness variation, not invented physical normal
    displacement. The file is PNG color type 0, 8-bit, filter 0, zlib compressed.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer >= 2")
    if not math.isfinite(base_roughness) or not 0 <= base_roughness <= 1:
        raise ValueError("base_roughness must lie in [0, 1]")
    rng = random.Random(seed)
    rows = []
    for y in range(size):
        grain = 0.018 * math.sin(2 * math.pi * y / 7)
        rows.append([base_roughness + grain + rng.uniform(-0.018, 0.018)
                     for _ in range(size)])
    for _ in range(max(8, size // 3)):
        y0 = rng.randrange(size)
        x0 = rng.randrange(size)
        length = rng.randint(max(1, size // 5), size)
        slope = rng.uniform(-0.07, 0.07)
        delta = rng.choice((-1, 1)) * rng.uniform(0.025, 0.11)
        for dx in range(length):
            # Wrapping makes texture repeats less conspicuous.
            y = round(y0 + slope * dx) % size
            x = (x0 + dx) % size
            rows[y][x] += delta
    raw = b"".join(b"\x00" + bytes(max(0, min(255, round(v * 255)))
                                    for v in row) for row in rows)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    payload = (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
               + _png_chunk(b"IDAT", zlib.compress(raw, 9))
               + _png_chunk(b"IEND", b""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    specs = generate_specs(args.count, args.seed)
    for index, spec in enumerate(specs):
        object_id = spec["object_id"]
        geometry = build_geometry(spec)
        write_obj(args.output / f"{object_id}.obj", geometry)
        write_roughness_png(args.output / f"{object_id}_roughness.png",
                            args.seed + index, spec["roughness"])
        (args.output / f"{object_id}.json").write_text(
            json.dumps({"spec": spec, "geometry": geometry}, indent=2) + "\n",
            encoding="utf-8")
    (args.output / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "units": "metres", "specs": specs}, indent=2)
        + "\n", encoding="utf-8")
    print(f"Exported {len(specs)} parts to {args.output.resolve()}")


if __name__ == "__main__":
    main()
