"""Create clearly synthetic RGB-D PNGs to demonstrate the customer manifest CLI."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def create_demo(output: Path) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")
    (output / "depth").mkdir(parents=True)
    (output / "rgb").mkdir()
    (output / "mask").mkdir()
    frames = []
    for index, invalid_fraction in enumerate((0.10, 0.25, 0.40), start=1):
        height, width = 40, 48
        yy, xx = np.mgrid[:height, :width]
        depth = (850 + xx * 4 + yy * 3 + index * 11).astype(np.uint16)
        mask = ((xx - 23) ** 2 + (yy - 19) ** 2 <= 11 ** 2).astype(np.uint8) * 255
        invalid_count = round(depth.size * invalid_fraction)
        order = np.random.default_rng(index).permutation(depth.size)
        depth.reshape(-1)[order[:invalid_count]] = 0
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[..., 0] = 90 + index * 30
        rgb[..., 1] = xx * 3
        rgb[..., 2] = yy * 4
        filename = f"{index:04d}.png"
        Image.fromarray(depth).save(output / "depth" / filename)
        Image.fromarray(rgb).save(output / "rgb" / filename)
        Image.fromarray(mask).save(output / "mask" / filename)
        frames.append({"id": f"demo-{index:02d}", "depth": f"depth/{filename}",
                       "rgb": f"rgb/{filename}", "mask": f"mask/{filename}",
                       "mask_alignment": "depth"})
    manifest = {
        "schema_version": 1,
        "camera": {"name": "synthetic demonstration only", "invalid_values": [0],
                   "depth_scale_mm": 0.25, "registration_status": "unknown"},
        "frames": frames,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for generated demo images and manifest")
    args = parser.parse_args(argv)
    manifest = create_demo(args.output)
    print(f"Created synthetic demonstration data: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
