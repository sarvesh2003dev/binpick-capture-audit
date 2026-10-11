# Customer manifest format, version 1

The CLI counts invalid values in original integer PNG depth images. It does not
perform CAD registration, identify failure causes, certify nonzero depth as
accurate, or overwrite input files.

Create a UTF-8 manifest.json in a private data directory:

```json
{
  "schema_version": 1,
  "camera": {
    "name": "camera name and configuration label",
    "invalid_values": [0],
    "depth_scale_mm": null,
    "registration_status": "unknown"
  },
  "frames": [
    {
      "id": "case-a-0001",
      "depth": "depth/0001.png",
      "rgb": "rgb/0001.png",
      "mask": "masks/0001.png",
      "mask_alignment": "depth"
    }
  ]
}
```

- `invalid_values` is required and explicit. Common convention: [0]. Include
  additional sentinel values only if the camera/export format specifies them.
- `depth_scale_mm` may be null. Without verified units, the CLI only reports raw
  integer values and valid/invalid counts.
- `registration_status` is `unknown`, `depth_native`, or `registered_to_rgb`.
  It is provenance supplied by the user, not validated calibration.
- Optional masks are 8-bit/16-bit PNGs; pixels greater than zero are the ROI.
  `mask_alignment` must be `depth` or `rgb`. RGB-aligned masks are accepted only
  when depth is declared registered to RGB and all grids have matching size.
- RGB PNG is optional; if included, dimensions are recorded. RGB/depth alignment
  is never inferred from equal dimensions.
- Every image path must be relative to the manifest directory and remain inside
  that directory. Absolute paths and `..` traversal are rejected.
- Up to 60 frames per manifest. One manifest represents one claimed camera
  configuration; do not mix camera settings in a single manifest.

The report gives pooled, pixel-weighted and per-frame invalid fractions for
whole images and optional ROIs, plus raw valid-depth quantiles. Quantiles in
millimetres are included only when a positive `depth_scale_mm` is declared;
that scale remains a user-provided assumption.
All output goes into a new directory. Keep customer inputs and reports local.

Example generator: `python -m customer_audit.demo --output /some/private/path`.
