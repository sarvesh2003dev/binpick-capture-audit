# Bin-picking SDG foundation

Actual Python source for procedural assets and an Isaac Sim capture pipeline.
The asset generator has been executed locally. The Isaac renderer and harness
dynamics have **not** been executed on a suitable GPU here. This is a foundation
for a perception dataset, not a validated robotics product or a trained policy.

## Run something useful on this laptop now

Open a terminal in this folder. No third-party dependencies are needed for these:

```powershell
python assets.py --count 50 --seed 7 --output my_parts
python -m unittest discover -s . -p "test_*.py" -v
```

Each part produces an OBJ mesh, a PNG roughness texture, and JSON containing
geometry, convex colliders, dimensions, mass and material parameters. A generated
set is already included in `sample_assets/`. OBJ has UVs but no MTL texture
binding; the Isaac runner creates the actual roughness material graph.

`geometry_preview.png` displays the generated geometry. It is a CPU preview,
not a photograph, physics result, or Isaac render. Re-create it separately:

```powershell
python -m venv .venv-preview
.\.venv-preview\Scripts\python.exe -m pip install -r requirements-preview.txt
.\.venv-preview\Scripts\python.exe preview_assets.py --output geometry_preview.png
```

## GPU host installation

Target **Python 3.12 + Isaac Sim 6.1.0.0**, using NVIDIA's current documented
installation sequence. Keep this environment separate from other ML projects.
The older World-based tutorials are not the API foundation used here.
[NVIDIA installation instructions](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html).

Your checked laptop has an RTX 2050, 4 GB VRAM, driver 566.07 and Python 3.13.2.
NVIDIA's published x86 requirements list 32 GB RAM, RTX 4080/16 GB VRAM and 50 GB
SSD. Tested drivers are Windows 595.97 and Linux 595.58.03. Use a compatible RTX
workstation or remote host for the commands below. A100/H100 are not substitutes
for the RTX renderer. No cloud resources have been rented or charged.
[Requirements](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/requirements.html).

Windows 11 PowerShell, after installing Python 3.12 with the Python launcher:

```powershell
py -3.12 -m venv .venv-isaac
.\.venv-isaac\Scripts\python.exe -m pip install --upgrade pip
.\.venv-isaac\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-isaac\Scripts\python.exe -m pip install -r requirements-isaac.txt
```

If Python 3.12 is missing on Windows, install it first:

```powershell
winget install --exact --id Python.Python.3.12 --source winget
```

Ubuntu 22.04/24.04 with Python 3.12 and its venv support installed:

```bash
python3.12 -m venv .venv-isaac
.venv-isaac/bin/python -m pip install --upgrade pip
.venv-isaac/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-isaac/bin/python -m pip install -r requirements-isaac.txt
```

The first Isaac import presents NVIDIA's license prompt. The code does not
silently accept it. NumPy, OpenUSD, PhysX and Replicator come from the Isaac
bundle; do not independently upgrade those packages. Record the resolved
environment after installing:

```powershell
.\.venv-isaac\Scripts\python.exe -m pip freeze > installed-environment.txt
```

## Execute the simulation

First qualify one object, then the required 50. Each output directory must be new
or empty; the runner refuses to overwrite an existing dataset.

```powershell
.\.venv-isaac\Scripts\python.exe generate_dataset.py --mode metal --count 1 --frames 1 --seed 7 --output runs\metal_smoke
.\.venv-isaac\Scripts\python.exe generate_dataset.py --mode metal --count 50 --frames 4 --seed 7 --headless --output runs\metal_50
.\.venv-isaac\Scripts\python.exe validate_dataset.py runs\metal_50
```

For a high-quality reflection comparison, rerun with `--path-tracing` and a new
output folder. The code sets 128 samples per pixel. Compare with real images;
more samples do not calibrate the material or the sensor.

The harness mode creates a Y-shaped cable from jointed rigid capsules, a junction
hub and three connectors. Start with two before scaling:

```powershell
.\.venv-isaac\Scripts\python.exe generate_dataset.py --mode harness --count 2 --frames 2 --seed 7 --max-settle-seconds 20 --output runs\harness_smoke
.\.venv-isaac\Scripts\python.exe generate_dataset.py --mode harness --count 50 --frames 4 --seed 7 --max-settle-seconds 30 --headless --output runs\harness_50
```

This mode may reject scenes or need parameter changes on its first GPU runs;
50 harnesses create roughly a thousand rigid bodies and many joints. No runtime
or stability claim has been established. It generates flexible configurations,
not a certified entanglement-force model.

`--frames 4` means four camera/light configurations of **one settled pile**.
To generate independent piles, run different seeds into different folders:

```powershell
7..16 | ForEach-Object {
    .\.venv-isaac\Scripts\python.exe generate_dataset.py --mode metal --count 50 --frames 4 --seed $_ --headless --output "runs\metal_seed_$_"
    if ($LASTEXITCODE -ne 0) { throw "Rejected or failed seed $_; inspect its manifest" }
}
```

This simple batch launches Isaac for each pile. Reusing a simulator process is
a later throughput optimization, after scene teardown/reinitialization is tested.

## What the capture writes

| Output | Meaning |
|---|---|
| `frame_0000_rgb.png` | Rendered RGB, uint8 |
| `frame_0000_depth_gt_m.npy` | float32 optical-axis depth in meters; invalid pixels NaN |
| `frame_0000_depth_valid.png` | 255 where the ground-truth depth is valid |
| `frame_0000_instances.npy` | uint32 visible instance IDs |
| `frame_0000_metadata.json` | Intrinsics, camera pose, lights, instance-to-prim mapping |
| `manifest.json` | Seed, geometry/material parameters, initial/final poses, settling status |
| `scene.usda` + `textures/` | Portable authored stage and its texture dependencies |

Depth is geometry ground truth. **Shiny pixels can have perfectly valid ground
truth even when a real depth camera would return holes or false ranges.** Optional
`--augment-depth` creates a separate, explicitly uncalibrated noisy depth file.
It does not simulate structured-light reconstruction, multipath, or stereo
matching. The unmodified ground truth is retained.

Camera matrices use column vectors in metadata. USD camera coordinates are
X-right/Y-up/-Z-forward; the exported conversion to ROS optical coordinates is
included. Read each frame's instance mapping; never infer a persistent object
identity from an integer mask value. Hidden objects need not appear in the mask.

## Files and current evidence

| File | Responsibility |
|---|---|
| `assets.py` | Deterministic hollow geometry, convex colliders, physical specs, scratch PNGs |
| `harness.py` | PhysX D6 cable approximation and connector joints |
| `pipeline.py` | Scene placement, quaternion sampling, settling/bounds helpers |
| `generate_dataset.py` | Isaac initialization, scene, materials, capture and manifest |
| `validate_dataset.py` | Checks real exported frame shapes, types, depth masks and IDs |
| `BLUEPRINT.md` | Architecture, parameter ranges, validation and RL boundary |

Local verification: asset generation, the 12 CPU unit tests, source parsing and
CLI help. NOT locally verified: Isaac startup, shader compilation, annotation
capture, PhysX stability, harness behavior, runtime throughput, or Sim2Real
performance. `BUILD_STATUS.md` records the same boundary.
