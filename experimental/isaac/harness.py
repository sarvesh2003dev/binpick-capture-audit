"""Procedural, branched harness geometry using regular PhysX rigid-body joints.

This is a segmented rigid approximation, not a calibrated cable constitutive
model: it has no axial stretch, conductor slip, jacket crushing, or tape
hysteresis. Import this module AFTER creating Isaac Sim's SimulationApp.
"""

from __future__ import annotations

import math
from typing import Any

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade


def _v3(values):
    return Gf.Vec3d(*(float(value) for value in values))


def _qf(quaternion):
    return Gf.Quatf(
        float(quaternion.GetReal()),
        Gf.Vec3f(quaternion.GetImaginary()),
    )


def _bind(prim, visual_material, physics_material):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(visual_material)
    binding.Bind(
        physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
    )


def _body(stage, path, position, orientation, mass):
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(position)
    body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(orientation)
    prim = body.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    properties = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    properties.CreateSolverPositionIterationCountAttr(32)
    properties.CreateSolverVelocityIterationCountAttr(4)
    properties.CreateEnableCCDAttr(True)
    # Requires enableCCD on the PhysX scene as well. Sweep CCD does not catch
    # every fast angular crossing, so this does NOT guarantee cable topology.
    properties.CreateLinearDampingAttr(0.02)
    properties.CreateAngularDampingAttr(0.02)
    return prim


def _collider(prim, radius, visual_material, physics_material):
    UsdPhysics.CollisionAPI.Apply(prim)
    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    # Default meter-scale contact offsets can dwarf a millimeter-scale cable.
    collision.CreateContactOffsetAttr(float(0.2 * radius))
    collision.CreateRestOffsetAttr(0.0)
    _bind(prim, visual_material, physics_material)


def _box(
    stage, path, position, orientation, dimensions, mass, radius,
    visual_material, physics_material,
):
    _body(stage, path, position, orientation, mass)
    shape = UsdGeom.Cube.Define(stage, path + "/geometry")
    shape.CreateSizeAttr(1.0)
    # Keep scale on the collision child; the rigid-body/joint frame is unscaled.
    shape.AddScaleOp().Set(_v3(dimensions))
    _collider(shape.GetPrim(), radius, visual_material, physics_material)


def _capsule(
    stage, path, start, end, radius, mass, visual_material, physics_material,
):
    delta = end - start
    length = delta.GetLength()
    orientation = Gf.Rotation(Gf.Vec3d(1, 0, 0), delta / length).GetQuat()
    center = (start + end) * 0.5
    _body(stage, path, center, orientation, mass)
    shape = UsdGeom.Capsule.Define(stage, path + "/geometry")
    shape.CreateAxisAttr("X")
    shape.CreateRadiusAttr(float(radius))
    shape.CreateHeightAttr(float(length))
    # Capsule height is cylindrical length. Tip-to-tip is length + 2*radius;
    # adjacent endcaps intentionally overlap and their joint filters the pair.
    _collider(shape.GetPrim(), radius, visual_material, physics_material)
    return center, orientation


def _joint(
    stage, path, body0_path, body1_path, pose0, pose1, anchor,
    frame_orientation, spec, fixed=False,
):
    schema = UsdPhysics.FixedJoint if fixed else UsdPhysics.Joint
    joint = schema.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    # Disable ONLY this jointed pair. All nonadjacent links retain collision.
    joint.CreateCollisionEnabledAttr(False)

    for index, (position, orientation) in enumerate((pose0, pose1)):
        inverse = orientation.GetInverse()
        local_position = Gf.Rotation(inverse).TransformDir(anchor - position)
        local_orientation = inverse * frame_orientation
        getattr(joint, f"CreateLocalPos{index}Attr")().Set(
            Gf.Vec3f(local_position)
        )
        getattr(joint, f"CreateLocalRot{index}Attr")().Set(
            _qf(local_orientation)
        )

    if fixed:
        return

    # For generic USD D6 joints LOW > HIGH means locked, not low == high.
    for axis in ("transX", "transY", "transZ"):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)

    # The frame X axis follows the cable: rotX twists, rotY/rotZ bend.
    # PhysX shares its two swing drive coefficients: Y/Z MUST be identical.
    per_degree = math.pi / 180.0
    for axis in ("rotX", "rotY", "rotZ"):
        stiffness = spec[
            "twist_stiffness_nm_rad" if axis == "rotX"
            else "bend_stiffness_nm_rad"
        ]
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr("force")
        # USD angular coefficients are per degree, our inputs are per radian.
        drive.CreateStiffnessAttr(float(stiffness * per_degree))
        drive.CreateDampingAttr(float(spec["damping_nm_s_rad"] * per_degree))
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
    # No imposed angular limits: real curvature limits need measurement.
    # Nearly reversed adjacent links can still make this coarse model singular.


def create_harness(
    stage, path: str, spec: dict[str, Any], origin,
    visual_material: UsdShade.Material, physics_material: UsdShade.Material,
) -> list[str]:
    """Create a Y harness and return every rigid-body prim path.

    ``stage`` must use meters and kilograms. ``path`` is a NEW absolute prim
    path below ancestors with identity transforms. ``origin`` is an XYZ world
    position in meters. The caller may label the root with object semantics.
    Both materials must be valid UsdShade.Material objects on this stage.

    Required spec keys (all physical properties are uncalibrated seed values):
      radius_m: cable radius, 0 < r <= .0035 m; example .002
      segment_length_m: cylindrical/centerline length, 2*r < L <= .020 m
      segments: integer stem-link count, 6..24, with segments*L <= .240 m
      linear_density_kg_m: cable mass per centerline meter; example .080
      bend_stiffness_nm_rad: per-joint bending coefficient; example .005
      twist_stiffness_nm_rad: per-joint torsional coefficient; example .002
      damping_nm_s_rad: per-joint angular damping coefficient; example .00005
      connector_mass_kg: mass of EACH of three end connectors; example .015
    Optional keys:
      branch_segments: integer length of EACH of two branches, 2 or 3; default3
      hub_mass_kg: rigid taped junction mass; default .006

    Initial geometry lies inside origin + ([-.16,.16],[-.14,.14],[-.02,.02]).
    It starts without nonadjacent overlaps; it is not a pre-generated tangle.
    Use low-height, sequential drops to generate piles. The caller must set
    scene CCD and a suitably small physics timestep, and reject failed scenes.
    The coefficients are per discrete joint: changing link resolution requires
    recalibration (for a simple rod, k_bend scales approximately as EI/L).
    """
    required = (
        "radius_m", "segment_length_m", "segments", "linear_density_kg_m",
        "bend_stiffness_nm_rad", "twist_stiffness_nm_rad",
        "damping_nm_s_rad", "connector_mass_kg",
    )
    for name in required:
        if name not in spec:
            raise ValueError(f"Missing harness specification: {name}")
        value = float(spec[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    radius = float(spec["radius_m"])
    length = float(spec["segment_length_m"])
    stem_count = int(spec["segments"])
    branch_count = int(spec.get("branch_segments", 3))
    hub_mass = float(spec.get("hub_mass_kg", 0.006))
    if stem_count != spec["segments"] or not 6 <= stem_count <= 24:
        raise ValueError("segments must be an integer from 6 through 24")
    if branch_count != spec.get("branch_segments", 3) or branch_count not in (2, 3):
        raise ValueError("branch_segments must be 2 or 3")
    if not 0 < radius <= 0.0035 or not 2 * radius < length <= 0.020:
        raise ValueError("Require 0 < radius <= .0035 and 2*radius < length <= .020")
    if stem_count * length > 0.240 + 1e-12:
        raise ValueError("Stem centerline length exceeds .240 m spawn envelope")
    if not math.isfinite(hub_mass) or hub_mass <= 0:
        raise ValueError("hub_mass_kg must be finite and positive")
    origin = _v3(origin)
    if not all(math.isfinite(float(component)) for component in origin):
        raise ValueError("origin must have three finite components")
    sdf_path = Sdf.Path(path)
    if not sdf_path.IsAbsolutePath() or not sdf_path.IsPrimPath():
        raise ValueError("path must be an absolute USD prim path")
    if stage.GetPrimAtPath(sdf_path):
        raise ValueError(f"Harness path already exists: {path}")
    if not visual_material or not physics_material:
        raise ValueError("Both visual_material and physics_material are required")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        raise ValueError("Harness stage must use metersPerUnit=1.0")
    if abs(UsdPhysics.GetStageKilogramsPerUnit(stage) - 1.0) > 1e-9:
        raise ValueError("Harness stage must use kilogramsPerUnit=1.0")

    UsdGeom.Xform.Define(stage, sdf_path)
    UsdGeom.Scope.Define(stage, path + "/joints")
    identity = Gf.Quatd(1.0)
    hub_position = origin + Gf.Vec3d(0.100, 0.0, 0.0)
    hub_path = path + "/hub"
    _box(
        stage, hub_path, hub_position, identity, (0.012, 0.016, 0.008),
        hub_mass, radius, visual_material, physics_material,
    )
    body_paths = [hub_path]
    # The branch origins are on DIFFERENT hub faces. Starting both at one
    # vertex would make sibling capsules overlap without a collision filter.
    routes = (
        ("stem", Gf.Vec3d(-0.006, 0, 0), Gf.Vec3d(-1, 0, 0), stem_count),
        ("upper", Gf.Vec3d(0.004, 0.008, 0), Gf.Vec3d(0.5, math.sqrt(3)/2, 0), branch_count),
        ("lower", Gf.Vec3d(0.004, -0.008, 0), Gf.Vec3d(0.5, -math.sqrt(3)/2, 0), branch_count),
    )
    mass_per_link = float(spec["linear_density_kg_m"]) * length
    connector_length = 0.012
    connector_width = max(0.007, 3.0 * radius)
    connector_height = max(0.006, 3.0 * radius)

    for route_name, offset, direction, count in routes:
        start = hub_position + offset
        previous_path = hub_path
        previous_pose = (hub_position, identity)
        for index in range(count):
            end = start + direction * length
            link_path = f"{path}/{route_name}_link_{index:02d}"
            pose = _capsule(
                stage, link_path, start, end, radius, mass_per_link,
                visual_material, physics_material,
            )
            _joint(
                stage, f"{path}/joints/{route_name}_{index:02d}",
                previous_path, link_path, previous_pose, pose, start,
                pose[1], spec,
            )
            body_paths.append(link_path)
            previous_path, previous_pose, start = link_path, pose, end

        connector_path = f"{path}/{route_name}_connector"
        connector_position = start + direction * (0.5 * connector_length)
        connector_pose = (connector_position, previous_pose[1])
        _box(
            stage, connector_path, *connector_pose,
            (connector_length, connector_width, connector_height),
            float(spec["connector_mass_kg"]), radius,
            visual_material, physics_material,
        )
        _joint(
            stage, f"{path}/joints/{route_name}_connector", previous_path,
            connector_path, previous_pose, connector_pose, start,
            previous_pose[1], spec, fixed=True,
        )
        body_paths.append(connector_path)

    return body_paths
