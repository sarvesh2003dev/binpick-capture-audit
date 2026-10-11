"""Validate and summarize customer supplied PNG depth captures conservatively."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


MAX_FRAMES = 60
MAX_IMAGE_PIXELS = 40_000_000
MAX_FILE_BYTES = 100_000_000
REGISTRATION = {"unknown", "depth_native", "registered_to_rgb"}


class AuditError(ValueError):
    """Input is missing, ambiguous, unsafe, or outside the supported format."""


def _contained_file(root: Path, text: object, label: str) -> Path:
    if not isinstance(text, str) or not text.strip():
        raise AuditError(f"{label} must be a non-empty relative file path")
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AuditError(f"{label} must remain inside the manifest directory")
    try:
        resolved = (root / candidate).resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        raise AuditError(f"{label} is missing or escapes the manifest directory") from error
    if not resolved.is_file():
        raise AuditError(f"{label} is not a file")
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise AuditError(f"{label} exceeds the 100 MB file limit")
    return resolved


def _load_png(path: Path, label: str, *, integer: bool = False) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise AuditError(f"{label} must be PNG")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise AuditError(f"{label} exceeds the 40 MP image limit")
            image.load()
            array = np.asarray(image)
    except (OSError, ValueError) as error:
        if isinstance(error, AuditError):
            raise
        raise AuditError(f"Could not decode {label} as PNG") from error
    if array.ndim not in (2, 3):
        raise AuditError(f"{label} must have two spatial dimensions")
    if integer:
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise AuditError(f"{label} must be a single-channel integer PNG")
        if array.size == 0 or int(array.min()) < 0 or int(array.max()) > 65535:
            raise AuditError(f"{label} values must fit unsigned 16-bit depth data")
    return array


def _summary(depth: np.ndarray, invalid_values: list[int], mask: np.ndarray | None) -> dict:
    selected = np.ones(depth.shape, dtype=bool) if mask is None else mask
    area = int(selected.sum())
    if area == 0:
        raise AuditError("ROI mask contains no selected pixels")
    invalid = selected & np.isin(depth, invalid_values)
    observed = depth[selected & ~np.isin(depth, invalid_values)]
    positive = observed[observed > 0]
    quantiles = (np.quantile(positive, [0.01, 0.5, 0.99]).tolist()
                 if positive.size else None)
    return {
        "pixels": area,
        "invalid_pixels": int(invalid.sum()),
        "invalid_fraction": float(invalid.sum() / area),
        "non_invalid_pixels": int(area - invalid.sum()),
        "positive_pixels": int(positive.size),
        "positive_raw_quantiles_01_50_99": quantiles,
    }


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise AuditError("Frame id must use 1–80 ASCII letters, digits, dot, dash or underscore")
    return value


def run_audit(manifest_path: Path) -> dict:
    """Read one manifest and return measurements without copying input paths."""
    manifest_path = manifest_path.resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 2_000_000:
        raise AuditError("Manifest must be a file no larger than 2 MB")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError("Manifest must be valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise AuditError("Expected manifest schema_version 1")

    camera = manifest.get("camera")
    if not isinstance(camera, dict):
        raise AuditError("camera must be an object")
    name = camera.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise AuditError("camera.name must be a short, non-empty label")
    invalid = camera.get("invalid_values")
    if (not isinstance(invalid, list) or not invalid
            or any(type(value) is not int or not 0 <= value <= 65535 for value in invalid)
            or len(set(invalid)) != len(invalid)):
        raise AuditError("camera.invalid_values must explicitly list unique uint16 sentinel values")
    scale = camera.get("depth_scale_mm")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))
                              or not np.isfinite(scale) or scale <= 0):
        raise AuditError("camera.depth_scale_mm must be null or a positive finite number")
    registration = camera.get("registration_status")
    if not isinstance(registration, str) or registration not in REGISTRATION:
        raise AuditError(f"camera.registration_status must be one of {sorted(REGISTRATION)}")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise AuditError(f"frames must contain 1–{MAX_FRAMES} entries")
    root = manifest_path.parent
    seen_ids: set[str] = set()
    output_frames = []
    total = {"whole": {"pixels": 0, "invalid_pixels": 0},
             "roi": {"pixels": 0, "invalid_pixels": 0}}
    for index, entry in enumerate(frames):
        if not isinstance(entry, dict):
            raise AuditError(f"frames[{index}] must be an object")
        frame_id = _safe_id(entry.get("id"))
        if frame_id in seen_ids:
            raise AuditError(f"Duplicate frame id: {frame_id}")
        seen_ids.add(frame_id)
        depth_path = _contained_file(root, entry.get("depth"), f"{frame_id}.depth")
        depth = _load_png(depth_path, f"{frame_id}.depth", integer=True)

        rgb_path = None
        rgb_shape = None
        if "rgb" in entry and entry["rgb"] is not None:
            rgb_path = _contained_file(root, entry["rgb"], f"{frame_id}.rgb")
            rgb = _load_png(rgb_path, f"{frame_id}.rgb")
            if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise AuditError(f"{frame_id}.rgb must have 3 or 4 channels")
            rgb_shape = [int(rgb.shape[0]), int(rgb.shape[1])]
            del rgb
        if registration == "registered_to_rgb":
            if rgb_shape is None or rgb_shape != [int(depth.shape[0]), int(depth.shape[1])]:
                raise AuditError(f"{frame_id}: registered_to_rgb requires RGB and matching RGB/depth grids")

        frame_mask = None
        if "mask" in entry and entry["mask"] is not None:
            alignment = entry.get("mask_alignment")
            if not isinstance(alignment, str) or alignment not in {"depth", "rgb"}:
                raise AuditError(f"{frame_id}: mask_alignment must explicitly be depth or rgb")
            if alignment == "rgb" and registration != "registered_to_rgb":
                raise AuditError(f"{frame_id}: RGB-aligned mask requires declared registered_to_rgb depth")
            if alignment == "rgb" and rgb_shape is None:
                raise AuditError(f"{frame_id}: RGB-aligned mask requires an RGB image")
            mask_path = _contained_file(root, entry["mask"], f"{frame_id}.mask")
            raw_mask = _load_png(mask_path, f"{frame_id}.mask", integer=True)
            if raw_mask.shape != depth.shape:
                raise AuditError(f"{frame_id}: mask and depth grids must match")
            frame_mask = raw_mask > 0
            del raw_mask
        elif "mask_alignment" in entry:
            raise AuditError(f"{frame_id}: mask_alignment is invalid without a mask")

        whole = _summary(depth, invalid, None)
        roi = _summary(depth, invalid, frame_mask) if frame_mask is not None else None
        if scale is not None:
            for measurement in (whole, roi):
                if measurement is not None:
                    raw_quantiles = measurement["positive_raw_quantiles_01_50_99"]
                    measurement["positive_quantiles_mm_01_50_99"] = (
                        None if raw_quantiles is None
                        else [float(value * scale) for value in raw_quantiles])
        for key in ("pixels", "invalid_pixels"):
            total["whole"][key] += whole[key]
            if roi is not None:
                total["roi"][key] += roi[key]
        output_frames.append({
            "id": frame_id,
            "depth_width": int(depth.shape[1]),
            "depth_height": int(depth.shape[0]),
            "rgb_width": None if rgb_shape is None else rgb_shape[1],
            "rgb_height": None if rgb_shape is None else rgb_shape[0],
            "whole_image": whole,
            "roi": roi,
        })
        del depth, frame_mask

    def aggregate(item: dict) -> dict:
        pixels, invalid_count = item["pixels"], item["invalid_pixels"]
        return {"pixels": pixels, "invalid_pixels": invalid_count,
                "invalid_fraction": invalid_count / pixels}

    return {
        "schema_version": 1,
        "camera": {
            "name": name,
            "invalid_values": invalid,
            "depth_scale_mm": scale,
            "registration_status": registration,
        },
        "frame_count": len(output_frames),
        "assumptions": [
            "Invalid codes and scale are supplied by the user; the CLI does not verify calibration.",
            "Non-invalid depth values are not certified accurate.",
            "ROI measurements depend on caller-supplied mask alignment and labels.",
            "This audit does not establish material causation or robot pick success.",
        ],
        "pooled_whole_image": aggregate(total["whole"]),
        "pooled_roi": aggregate(total["roi"]) if total["roi"]["pixels"] else None,
        "frames": output_frames,
    }


def write_report(result: dict, output: Path) -> None:
    """Write a JSON and Markdown summary into a fresh local directory."""
    output = output.resolve()
    if output.exists():
        raise AuditError("Output directory already exists; choose a new private output path")
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# RGB-D capture validity audit", "",
        f"Camera label: {result['camera']['name']}",
        f"Frames: {result['frame_count']}",
        f"Declared invalid raw codes: {', '.join(map(str, result['camera']['invalid_values']))}",
        f"Declared registration: {result['camera']['registration_status']}",
        "Depth scale (mm/raw unit): " + (
            "not supplied" if result["camera"]["depth_scale_mm"] is None
            else str(result["camera"]["depth_scale_mm"])),
        "",
        "## Pooled counts", "",
        f"- Whole images: {result['pooled_whole_image']['invalid_pixels']:,} of "
        f"{result['pooled_whole_image']['pixels']:,} invalid "
        f"({result['pooled_whole_image']['invalid_fraction']:.2%}).",
    ]
    if result["pooled_roi"] is not None:
        roi = result["pooled_roi"]
        lines.append(f"- Caller-provided ROIs: {roi['invalid_pixels']:,} of "
                     f"{roi['pixels']:,} invalid ({roi['invalid_fraction']:.2%}).")
    lines += ["", "## Per-frame counts", "",
              "| Frame id | Whole invalid | ROI invalid | Positive raw median |",
              "|---|---:|---:|---:|"]
    for frame in result["frames"]:
        whole = frame["whole_image"]
        roi_text = "—" if frame["roi"] is None else (
            f"{frame['roi']['invalid_pixels']:,}/{frame['roi']['pixels']:,} "
            f"({frame['roi']['invalid_fraction']:.2%})")
        q = whole["positive_raw_quantiles_01_50_99"]
        if q is None:
            median = "—"
        elif result["camera"]["depth_scale_mm"] is None:
            median = f"{q[1]:.4g} raw units"
        else:
            mm = q[1] * result["camera"]["depth_scale_mm"]
            median = f"{q[1]:.4g} raw; {mm:.4g} mm (declared scale)"
        lines.append(f"| {frame['id']} | {whole['invalid_pixels']:,}/{whole['pixels']:,} "
                     f"({whole['invalid_fraction']:.2%}) | {roi_text} | {median} |")
    lines += ["", "## Limits", ""]
    lines.extend(f"- {assumption}" for assumption in result["assumptions"])
    lines += ["", "No source image file paths are included in this report.",
              "Keep the report with the customer's access and retention controls.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
