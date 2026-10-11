"""Validate completed Isaac captures; never mistakes asset previews for data."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def validate(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Not a completed simulation dataset (assets-only exports do not qualify)")
    width, height = manifest["arguments"]["width"], manifest["arguments"]["height"]
    if not (folder / "scene.usda").is_file():
        raise ValueError("Missing portable scene.usda")
    summaries = []
    for frame in range(manifest["frames_written"]):
        prefix = folder / f"frame_{frame:04d}"
        rgb = np.asarray(Image.open(str(prefix)+"_rgb.png"))
        depth = np.load(str(prefix)+"_depth_gt_m.npy", allow_pickle=False)
        instances = np.load(str(prefix)+"_instances.npy", allow_pickle=False)
        valid_png = np.asarray(Image.open(str(prefix)+"_depth_valid.png")) != 0
        metadata = json.loads(Path(str(prefix)+"_metadata.json").read_text(encoding="utf-8"))
        if rgb.shape != (height,width,3) or rgb.dtype != np.uint8:
            raise ValueError(f"Frame {frame}: invalid RGB shape/type")
        if depth.shape != (height,width) or depth.dtype != np.float32:
            raise ValueError(f"Frame {frame}: invalid depth shape/type")
        if instances.shape != (height,width) or instances.dtype != np.uint32:
            raise ValueError(f"Frame {frame}: invalid instance shape/type")
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any() or not np.array_equal(valid,valid_png):
            raise ValueError(f"Frame {frame}: depth/mask mismatch")
        ids = metadata["visible_object_instance_ids"]
        if not ids or any(not np.any(instances == i) for i in ids):
            raise ValueError(f"Frame {frame}: no declared visible object pixels")
        if not np.isfinite(np.array(metadata["K"])).all():
            raise ValueError(f"Frame {frame}: invalid intrinsics")
        summaries.append({"frame":frame,"visible_instances":len(ids),
                          "depth_min_m":float(depth[valid].min()),
                          "depth_max_m":float(depth[valid].max())})
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.folder),indent=2))
