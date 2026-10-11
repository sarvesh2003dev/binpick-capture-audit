# Bin-picking synthetic data generator — delivered source

Implementation and local verification status. GPU execution remains unverified.

## Scope

- 50 procedural metal-part variants: bushings, flanged bushings, simplified gears.
- Separate harness mode: rigid capsule links with spring-driven joints and connectors.
- Physics settling, RTX RGB capture, ideal metric depth, per-object instance masks.
- Recorded seeds, geometry, materials, physical parameters, camera calibration and poses.
- No downloaded meshes are required.

## Completed work

1. Procedural geometry and scratch-texture generator: executed for 50 variants.
2. Isaac Sim 6.1.0.0 scene/capture runner: source written; APIs checked against documentation.
3. Harness joint helper: source written; no physics execution claim.
4. Installation commands and technical blueprint: README.md and BLUEPRINT.md.
5. Local tests: 12 passing; all Python source parses; CLI help checked.
6. Real generated assets: sample_assets/; CPU mesh preview: geometry_preview.png.
7. Dataset validator: source written; awaits actual Isaac capture output.

## Remaining GPU acceptance work

- Run the one-part capture and confirm RGB/depth/instance alignment and units.
- Qualify the 50-metal-part scene and inspect collision/settling behavior.
- Qualify two harnesses before ten and fifty; check joints and self collisions.
- Repeat physical comparisons with smaller timesteps and cable segments.
- Calibrate appearance, sensor failures and mechanics against real measurements.

No Isaac frames or simulated pick-success labels are included in this delivery.

## Hardware and execution boundary

Detected locally: RTX 2050 Laptop GPU, 4 GB VRAM, driver 566.07, Python 3.13.2.
The target Isaac Sim installation uses Python 3.12 and requires a substantially
more capable RTX host. Source generation and CPU-only asset checks can run here.
Do not interpret the source files or asset previews as evidence of a completed
Isaac simulation or successful transfer to a real robot.
