"""CPU-only scene planning and export validation. No Isaac imports here."""
from __future__ import annotations

import math
import random


def harness_specs(count: int, seed: int) -> list[dict]:
    if not 1 <= count <= 500:
        raise ValueError("count must be between 1 and 500")
    rng = random.Random(seed)
    specs = []
    for i in range(count):
        dynamic = rng.uniform(0.30, 0.70)
        specs.append({
            "object_id": f"harness_{i:03d}",
            "family": "branched_rigid_link_harness",
            "radius_m": rng.uniform(0.0018, 0.0028),
            "segments": rng.randint(8, 12),
            "segment_length_m": rng.uniform(0.014, 0.018),
            "linear_density_kg_m": rng.uniform(0.06, 0.12),
            "bend_stiffness_nm_rad": rng.uniform(0.003, 0.012),
            "twist_stiffness_nm_rad": rng.uniform(0.001, 0.005),
            "damping_nm_s_rad": rng.uniform(0.00003, 0.00009),
            "connector_mass_kg": rng.uniform(0.008, 0.020),
            "static_friction": dynamic + rng.uniform(0.03, 0.12),
            "dynamic_friction": dynamic,
            "restitution": rng.uniform(0.0, 0.06),
            "roughness": rng.uniform(0.45, 0.85),
            "metallic": 0.0,
            "color": rng.choice([(0.025, 0.03, 0.035), (0.08, 0.11, 0.16),
                                  (0.17, 0.06, 0.025)]),
        })
    return specs


def scene_plan(mode: str, count: int, seed: int) -> dict:
    """Nonintersecting initial envelopes, not teleportation into a tangled pile.

    Metal envelope is a sphere of radius < 0.043 m. Harness helper guarantees
    x +/-0.16, y +/-0.14, z +/-0.02 m. No global harness rotation here.
    """
    if mode not in ("metal", "harness"):
        raise ValueError("mode must be metal or harness")
    if not 1 <= count <= 500:
        raise ValueError("count must be between 1 and 500")
    rng = random.Random(seed ^ 0x719AD)
    if mode == "metal":
        nx, ny, pitch, layer = 6, 4, 0.096, 0.095
        bin_size = (0.66, 0.47, 0.25)
        bottom = 0.12
    else:
        nx, ny, pitch, layer = 4, 3, 0.40, 0.065
        bin_size = (1.68, 1.29, 0.38)
        bottom = 0.085
    placements = []
    for i in range(count):
        ix = i % nx
        iy = (i // nx) % ny
        iz = i // (nx * ny)
        jitter = 0.003 if mode == "metal" else 0.005
        p = ((ix - (nx - 1) / 2) * pitch + rng.uniform(-jitter, jitter),
             (iy - (ny - 1) / 2) * pitch + rng.uniform(-jitter, jitter),
             bottom + iz * layer)
        if mode == "metal":
            # Uniform SO(3), quaternion serialized WXYZ. Uniform Euler angles
            # would overweight some orientations.
            u1, u2, u3 = rng.random(), rng.random(), rng.random()
            q = (math.sqrt(u1) * math.cos(2 * math.pi * u3),
                 math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
                 math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
                 math.sqrt(u1) * math.sin(2 * math.pi * u3))
        else:
            q = (1.0, 0.0, 0.0, 0.0)
        placements.append({"position_m": p, "quaternion_wxyz": q})
    return {"bin_inner_size_m": bin_size, "placements": placements,
            "physics_hz": 240, "units": "meters, kilograms, seconds"}


def pose_motion(previous: dict, current: dict, elapsed_s: float) -> tuple[float, float]:
    """Maximum translational/angular speed inferred from all body poses.

    These sampled speeds cannot detect motion that returns between samples.
    This is a settle heuristic, not penetration/entanglement certification.
    """
    if elapsed_s <= 0 or set(previous) != set(current):
        raise ValueError("positive time and identical body sets required")
    max_linear = max_angular = 0.0
    for key, (p1, q1) in current.items():
        p0, q0 = previous[key]
        if not all(math.isfinite(v) for v in (*p0, *p1, *q0, *q1)):
            return math.inf, math.inf
        max_linear = max(max_linear, math.dist(p0, p1) / elapsed_s)
        dot = abs(sum(a * b for a, b in zip(q0, q1)))
        angle = 2 * math.acos(min(1.0, max(0.0, dot)))
        max_angular = max(max_angular, angle / elapsed_s)
    return max_linear, max_angular


def bounds_errors(poses: dict, bin_size: tuple) -> list[str]:
    """Reject nonfinite transforms, floor escapes, and escaped body origins.

    This deliberately does not claim to test mesh penetration.
    """
    bx, by, _ = bin_size
    reasons = []
    for path, (position, quaternion) in poses.items():
        if not all(math.isfinite(x) for x in (*position, *quaternion)):
            reasons.append(f"nonfinite transform: {path}")
        elif abs(position[0]) > bx / 2 or abs(position[1]) > by / 2 or position[2] < -0.005:
            reasons.append(f"body origin outside bin: {path}")
    return reasons
