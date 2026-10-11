"""Isaac Sim 6.1.0.0: procedural bin piles -> RGB, metric depth, instances.

Run with the dedicated Python 3.12 Isaac environment; see README.md.
This is a perception dataset generator, not an RL grasping environment.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from assets import build_geometry, generate_specs, write_roughness_png
from pipeline import bounds_errors, harness_specs, pose_motion, scene_plan


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metal", "harness"), default="metal")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--frames", type=int, default=4, help="views of ONE settled pile")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--path-tracing", action="store_true")
    parser.add_argument("--max-settle-seconds", type=float, default=12.0)
    parser.add_argument("--physics-hz", type=int, default=240)
    parser.add_argument("--augment-depth", action="store_true",
                        help="also save explicitly uncalibrated depth augmentation")
    args = parser.parse_args()
    if not 1 <= args.count <= 500 or args.frames < 1:
        parser.error("count must be 1..500 and frames must be positive")
    if min(args.width, args.height) < 64 or args.physics_hz < 120:
        parser.error("image dimensions >=64 and physics-hz >=120 required")
    if args.max_settle_seconds < 2:
        parser.error("max-settle-seconds must be >=2")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; existing datasets are never overwritten")
    return args


def write_json(path, value):
    def convert(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"Unsupported metadata type: {type(item).__name__}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n", encoding="utf-8")


def run(args, app):
    # Kit must already be running BEFORE importing any omni/pxr modules.
    import carb.settings
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager
    from PIL import Image
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    args.output.mkdir(parents=True, exist_ok=True)
    texture_dir = args.output / "textures"
    texture_dir.mkdir()
    rng = random.Random(args.seed)
    rep.set_global_seed(args.seed)
    settings = carb.settings.get_settings()
    # USD writeback is essential for exported poses and our settling checks.
    # The setting path has NO '/persistent' prefix. Fabric bypasses USD unless
    # managed explicitly; CPU/USD is the intentional reference implementation.
    SimulationManager.enable_fabric(False)
    settings.set("/physics/updateToUsd", True)
    settings.set("/physics/updateVelocitiesToUsd", True)
    settings.set("/rtx/post/dlss/execMode", 2)
    if args.path_tracing:
        settings.set("/rtx/pathtracing/spp", 128)
        settings.set("/rtx/pathtracing/totalSpp", 128)
    rep.orchestrator.set_capture_on_play(False)
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    if SimulationManager.get_active_physics_engine() != "physx":
        if not SimulationManager.switch_physics_engine("physx"):
            raise RuntimeError("Could not select the PhysX engine")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/Objects")

    physics = UsdPhysics.Scene.Define(stage, "/World/Physics")
    physics.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
    scene_api.CreateTimeStepsPerSecondAttr(args.physics_hz)
    scene_api.CreateEnableCCDAttr(True)
    scene_api.CreateEnableGPUDynamicsAttr(False)
    scene_api.CreateBroadphaseTypeAttr("MBP")
    scene_api.CreateSolverTypeAttr("TGS")

    def visual_material(path, color, roughness, metallic=0.0, texture=None):
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        rough_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        rough_input.Set(float(roughness))
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        if texture:
            reader = UsdShade.Shader.Define(stage, path + "/uv")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(stage, path + "/roughness_texture")
            tex.CreateIdAttr("UsdUVTexture")
            # Absolute paths work while running; convert to relative at export.
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture.resolve())))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
            rough_input.ConnectToSource(tex.ConnectableAPI(), "r")
        return mat

    def physics_material(path, mu_s, mu_d, restitution):
        mat = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(mu_s))
        api.CreateDynamicFrictionAttr(float(mu_d))
        api.CreateRestitutionAttr(float(restitution))
        physx = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        # Effective pair coefficient is average(object, bin), not object alone.
        physx.CreateFrictionCombineModeAttr("average")
        physx.CreateRestitutionCombineModeAttr("min")
        return mat

    def collider(prim, mat, contact_offset=0.0005):
        UsdPhysics.CollisionAPI.Apply(prim)
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(contact_offset)
        api.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    def cube(path, position, scale, visual, physical):
        mesh = UsdGeom.Cube.Define(stage, path)
        mesh.CreateSizeAttr(1.0)
        mesh.AddTranslateOp().Set(Gf.Vec3d(*position))
        mesh.AddScaleOp().Set(Gf.Vec3d(*scale))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual)
        collider(mesh.GetPrim(), physical)
        return mesh

    plan = scene_plan(args.mode, args.count, args.seed)
    plan["physics_hz"] = args.physics_hz
    bx, by, bh = plan["bin_inner_size_m"]
    bin_color = [rng.uniform(0.10, 0.30) for _ in range(3)]
    bin_vis = visual_material("/World/Looks/bin_visual", bin_color, 0.65)
    bin_phys = physics_material("/World/Looks/bin_physical", 0.5, 0.4, 0.02)
    # Five static boxes; an enclosing solid cube would fill the bin interior.
    wall = 0.02
    cube("/World/Bin/floor", (0, 0, -wall/2), (bx+2*wall, by+2*wall, wall), bin_vis, bin_phys)
    for sign in (-1, 1):
        side = "negative" if sign < 0 else "positive"
        cube(f"/World/Bin/x_{side}", (sign*(bx+wall)/2, 0, bh/2),
             (wall, by+2*wall, bh), bin_vis, bin_phys)
        cube(f"/World/Bin/y_{side}", (0, sign*(by+wall)/2, bh/2),
             (bx, wall, bh), bin_vis, bin_phys)

    def mesh_from_geometry(path, geometry, with_uv=False):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in geometry["points"]])
        mesh.CreateFaceVertexCountsAttr([len(f) for f in geometry["faces"]])
        mesh.CreateFaceVertexIndicesAttr([i for face in geometry["faces"] for i in face])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        if with_uv:
            uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                       UsdGeom.Tokens.faceVarying)
            uv.Set([Gf.Vec2f(*v) for v in geometry["uv"]])
        return mesh

    specs = generate_specs(args.count, args.seed) if args.mode == "metal" else harness_specs(args.count, args.seed)
    object_records = []
    body_paths = []
    for index, (spec, placement) in enumerate(zip(specs, plan["placements"])):
        name = spec["object_id"]
        path = "/World/Objects/" + name
        phys_mat = physics_material(f"/World/Looks/{name}_physics", spec["static_friction"],
                                    spec["dynamic_friction"], spec["restitution"])
        if args.mode == "metal":
            texture = texture_dir / f"{name}_roughness.png"
            write_roughness_png(texture, args.seed * 1000 + index, spec["roughness"])
            vis_mat = visual_material(f"/World/Looks/{name}_visual", (0.65, 0.67, 0.70),
                                      spec["roughness"], spec["metallic"], texture)
            obj = UsdGeom.Xform.Define(stage, path)
            obj.AddTranslateOp().Set(Gf.Vec3d(*placement["position_m"]))
            w, x, y, z = placement["quaternion_wxyz"]
            obj.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, Gf.Vec3d(x,y,z)))
            UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
            UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr(spec["mass_kg"])
            dynamics = PhysxSchema.PhysxRigidBodyAPI.Apply(obj.GetPrim())
            dynamics.CreateSolverPositionIterationCountAttr(16)
            dynamics.CreateSolverVelocityIterationCountAttr(4)
            dynamics.CreateEnableCCDAttr(True)
            dynamics.CreateLinearDampingAttr(0.01)
            dynamics.CreateAngularDampingAttr(0.01)
            geometry = build_geometry(spec)
            visual = mesh_from_geometry(path + "/visual", geometry, with_uv=True)
            UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(vis_mat)
            # One convex hull would CLOSE THE BORE. Convex radial wedges retain
            # the hole; collision children MUST NOT have their own RigidBodyAPI.
            for j, piece in enumerate(geometry["colliders"]):
                collision_mesh = mesh_from_geometry(f"{path}/colliders/c_{j:03d}", piece)
                collision_mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                UsdPhysics.MeshCollisionAPI.Apply(collision_mesh.GetPrim()).CreateApproximationAttr("convexHull")
                collider(collision_mesh.GetPrim(), phys_mat)
            objects_bodies = [path]
        else:
            from harness import create_harness
            # Constant jacket material: capsules have no authored UVs. Do not
            # pretend a PNG exists on a surface without the mapping to display it.
            vis_mat = visual_material(f"/World/Looks/{name}_visual", spec["color"], spec["roughness"])
            objects_bodies = create_harness(stage, path, spec, placement["position_m"], vis_mat, phys_mat)
        rep.functional.modify.semantics(stage.GetPrimAtPath(path), {"class": spec["family"]}, mode="add")
        body_paths.extend(objects_bodies)
        object_records.append({"prim_path": path, "spec": spec, "initial_pose": placement,
                               "body_paths": objects_bodies})
    print(f"Built {len(specs)} objects, {len(body_paths)} rigid bodies", flush=True)

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))
    # Raw USD lens attributes use TENTHS of a stage unit. With metersPerUnit=1,
    # 1 attribute unit = 100 mm. Ratios alone would render correctly even with
    # a 100x lens-scale error, hiding the mistake until depth of field is used.
    camera.CreateHorizontalApertureAttr(36.0 / 100.0)
    camera.CreateVerticalApertureAttr(36.0 / 100.0 * args.height / args.width)
    camera.CreateFocusDistanceAttr(1.0)
    camera.CreateFStopAttr(0.0)  # Ideal pinhole; no uncalibrated lens blur.
    camera_op = camera.AddTransformOp()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/dome")
    lights = []
    for i in range(3):
        light = UsdLux.RectLight.Define(stage, f"/World/Lights/panel_{i}")
        light.CreateEnableColorTemperatureAttr(True)
        lights.append((light, light.AddTransformOp()))

    def randomize_view(frame_id):
        focal = rng.uniform(19.0, 24.0)
        elevation = max(1.0, max(bx, by*args.width/args.height) * 1.6)
        eye = (rng.uniform(-0.12, 0.12)*bx, rng.uniform(-0.12, 0.12)*by, elevation)
        target = (0.0, 0.0, 0.04)
        transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0,1,0)).GetInverse()
        camera_op.Set(transform)
        camera.GetFocalLengthAttr().Set(focal / 100.0)
        dome_intensity = rng.uniform(70.0, 250.0)
        dome.CreateIntensityAttr(dome_intensity)
        light_info = []
        for light, op in lights:
            position = (rng.uniform(-bx,bx), rng.uniform(-by,by), elevation*rng.uniform(0.6,0.95))
            intensity, temperature = rng.uniform(1200,4500), rng.uniform(3500,6500)
            width, height = rng.uniform(.15,.55)*elevation, rng.uniform(.07,.25)*elevation
            light.CreateIntensityAttr(intensity)
            light.CreateColorTemperatureAttr(temperature)
            light.CreateWidthAttr(width)
            light.CreateHeightAttr(height)
            op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*position), Gf.Vec3d(0,0,0), Gf.Vec3d(0,1,0)).GetInverse())
            light_info.append({"position_m": position, "intensity_usd": intensity,
                               "color_temperature_k": temperature, "size_m": [width,height]})
        # USD's matrix uses row-vector convention. Transpose for familiar
        # column-vector T_world_from_camera exported here. Camera axes: X right,
        # Y up, -Z forward. ROS optical coordinates require diag(1,-1,-1).
        matrix = np.asarray(transform, dtype=float).T
        fx = focal / 36.0 * args.width
        return {"frame": frame_id, "resolution_wh": [args.width,args.height],
                "K": [[fx,0,args.width/2], [0,fx,args.height/2], [0,0,1]],
                "principal_point_convention": "continuous image edges; pixel centers at i+0.5",
                "T_world_from_usd_camera": matrix.tolist(),
                "usd_camera_from_ros_optical": [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]],
                "focal_length_mm": focal, "horizontal_aperture_mm": 36.0,
                "depth_kind": "ideal optical-axis Z in meters, not Euclidean range",
                "dome_intensity_usd": dome_intensity, "lights": light_info}

    randomize_view(0)
    product = rep.create.render_product(str(camera.GetPath()), (args.width, args.height))
    rgb_ann = rep.annotators.get("rgb")
    depth_ann = rep.annotators.get("distance_to_image_plane")
    instance_ann = rep.annotators.get("instance_segmentation", init_params={"colorize": False})
    for ann in (rgb_ann, depth_ann, instance_ann):
        ann.attach(product)

    def poses():
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        values = {}
        for path in body_paths:
            transform = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            p = transform.ExtractTranslation()
            q = transform.ExtractRotationQuat().GetNormalized()
            values[path] = (tuple(p), (float(q.GetReal()), *tuple(q.GetImaginary())))
        return values

    manifest = {"schema_version": 1, "seed": args.seed, "mode": args.mode,
                "isaacsim_target": "6.1.0.0", "python": sys.version,
                "platform": platform.platform(), "arguments": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "status": "building", "plan": plan, "objects": object_records,
                "bin_friction_static_dynamic": [0.5,0.4], "bin_color": bin_color,
                "calibration_status": "UNCALIBRATED synthetic example",
                "physics_model": "PhysX CPU rigid bodies; harnesses are jointed rigid links",
                "renderer": "PathTracing" if args.path_tracing else "RaytracedLighting",
                "capture_frames_are_views_of_one_pile": True}
    try:
        manifest["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                   "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        manifest["gpu"] = "nvidia-smi unavailable"
    write_json(args.output / "manifest.json", manifest)

    # Physics-only stepping avoids spending render time while objects fall.
    # Initialize after constructing all bodies; do NOT also run timeline.play()
    # or app.update() in this loop, which could add unintended physics steps.
    SimulationManager.set_device("cpu")
    SimulationManager.enable_fabric(False)
    settings.set("/physics/updateToUsd", True)
    settings.set("/physics/updateVelocitiesToUsd", True)
    SimulationManager.initialize_physics()
    previous = poses()
    initial_body_poses = previous
    start_wall = time.perf_counter()
    chunk_steps = max(1, args.physics_hz // 20)
    interval = chunk_steps / args.physics_hz
    stable_s = simulated_s = 0.0
    settled = False
    stats = {}
    while simulated_s < args.max_settle_seconds:
        SimulationManager.step(steps=chunk_steps)
        current = poses()
        simulated_s += interval
        linear, angular = pose_motion(previous, current, interval)
        stable_s = stable_s + interval if linear < 0.002 and angular < 0.05 else 0.0
        previous = current
        if stable_s >= 0.5 and simulated_s >= 1.5:
            settled = True
            break
    reasons = bounds_errors(previous, plan["bin_inner_size_m"])
    if not settled:
        reasons.append("settle timeout: bodies did not remain below speed thresholds")
    # A frozen initial stack must never pass as a settled pile if writeback or
    # stepping failed. Verify gravity actually moved at least one body.
    largest_drop = max(initial_body_poses[path][0][2] - value[0][2]
                       for path, value in previous.items())
    if not math.isfinite(largest_drop) or largest_drop < 0.02:
        reasons.append("no observed gravity drop; inspect physics initialization/USD writeback")
    stats.update({"simulated_seconds": simulated_s, "wall_seconds": time.perf_counter()-start_wall,
                  "max_linear_speed_m_s": linear if math.isfinite(linear) else None,
                  "max_angular_speed_rad_s": angular if math.isfinite(angular) else None,
                  "settled": settled, "rejection_reasons": reasons})
    manifest["settling"] = stats
    manifest["body_poses_world"] = previous if not any("nonfinite" in r for r in reasons) else None
    manifest["status"] = "rejected" if reasons else "capturing"
    write_json(args.output / "manifest.json", manifest)
    if reasons:
        stage.GetRootLayer().Export(str(args.output / "rejected_scene.usda"))
        raise RuntimeError("Scene rejected; inspect manifest.json: " + "; ".join(reasons[:4]))
    print(f"Settled in {simulated_s:.2f} simulated seconds; capturing {args.frames} views", flush=True)

    try:
        for frame in range(args.frames):
            metadata = randomize_view(frame)
            omni.usd.get_context().reset_renderer_accumulation()
            rep.orchestrator.step(delta_time=0.0, rt_subframes=8, pause_timeline=True, wait_for_render=True)
            rgb = np.asarray(rgb_ann.get_data())
            depth = np.asarray(depth_ann.get_data(), dtype=np.float32).reshape(args.height,args.width)
            instances = instance_ann.get_data()
            mask = np.asarray(instances["data"], dtype=np.uint32).reshape(args.height,args.width)
            if rgb.shape[:2] != (args.height,args.width) or rgb.size == 0:
                raise RuntimeError(f"RGB annotator returned unexpected shape {rgb.shape}")
            valid = np.isfinite(depth) & (depth > 0)
            known_instances = {int(key): value for key,value in instances["info"]["idToLabels"].items()
                               if isinstance(value,str) and value.startswith("/World/Objects/")}
            visible_instances = sorted(set(int(v) for v in np.unique(mask)) & set(known_instances))
            if not valid.any() or not visible_instances:
                raise RuntimeError("No valid depth or labeled objects; inspect camera and annotation setup")
            depth = np.where(valid, depth, np.nan).astype(np.float32)
            prefix = args.output / f"frame_{frame:04d}"
            Image.fromarray(rgb[:,:,:3].astype(np.uint8)).save(str(prefix)+"_rgb.png")
            np.save(str(prefix)+"_depth_gt_m.npy", depth)
            np.save(str(prefix)+"_instances.npy", mask)
            Image.fromarray((valid*255).astype(np.uint8)).save(str(prefix)+"_depth_valid.png")
            # Ground truth remains unchanged; this optional file is explicitly
            # NOT a calibrated structured-light/ToF camera model.
            if args.augment_depth:
                noise_rng = np.random.default_rng(args.seed * 10000 + frame)
                sigma = 0.001 * np.square(np.nan_to_num(depth, nan=0.0))
                noisy = depth + noise_rng.normal(0.0, sigma, depth.shape)
                drop = noise_rng.random(depth.shape) < 0.02
                noisy[drop | ~valid | (noisy <= 0)] = np.nan
                np.save(str(prefix)+"_depth_aug_UNCALIBRATED_m.npy", noisy.astype(np.float32))
                metadata["depth_augmentation"] = {"sigma_m": "0.001 * depth_m^2", "random_dropout": 0.02,
                                                   "status": "uncalibrated augmentation, no reflection model"}
            metadata["instance_info"] = instances["info"]
            metadata["visible_object_instance_ids"] = visible_instances
            metadata["valid_depth_fraction"] = float(valid.mean())
            write_json(Path(str(prefix)+"_metadata.json"), metadata)
            print(f"Saved frame {frame+1}/{args.frames}", flush=True)
        # Replace texture paths with paths relative to this saved layer. This
        # makes the scene portable together with its textures subdirectory.
        for prim in stage.Traverse():
            attr = prim.GetAttribute("inputs:file")
            if attr and attr.Get():
                value = attr.Get()
                attr.Set(Sdf.AssetPath("textures/" + Path(value.path).name))
        stage.GetRootLayer().Export(str(args.output / "scene.usda"))
        manifest["status"] = "complete"
        manifest["frames_written"] = args.frames
        write_json(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest["status"] = "capture_failed"
        manifest["error"] = str(error)
        write_json(args.output / "manifest.json", manifest)
        raise
    finally:
        for ann in (rgb_ann, depth_ann, instance_ann):
            ann.detach(product)
        rep.orchestrator.wait_until_complete()
        product.destroy()


def main():
    args = parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Isaac Sim 6.1.0.0 requires Python 3.12. Use the environment in README.md; asset generation works separately.")
    try:
        version = importlib.metadata.version("isaacsim")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit("Isaac Sim is not installed in this interpreter. Follow README.md.")
    if version != "6.1.0.0":
        raise SystemExit(f"Expected isaacsim 6.1.0.0, found {version}. Use the pinned dedicated environment.")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless,
                         "renderer": "PathTracing" if args.path_tracing else "RaytracedLighting",
                         "width": args.width, "height": args.height})
    try:
        run(args, app)
    finally:
        app.close()


if __name__ == "__main__":
    main()
