"""Command line entry point for the local RGB-D capture audit."""

import argparse
from pathlib import Path
import sys

from .core import AuditError, run_audit, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="UTF-8 JSON manifest stored beside relative image paths")
    parser.add_argument("--output", type=Path, required=True,
                        help="new local directory for report.md and metrics.json")
    args = parser.parse_args(argv)
    try:
        result = run_audit(args.manifest)
        write_report(result, args.output)
    except AuditError as error:
        print(f"Input/output rejected: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Could not read inputs or write report: {error}", file=sys.stderr)
        return 2
    print(f"Audit complete: {args.output.resolve()}")
    print(f"Frames: {result['frame_count']}; "
          f"whole-image invalid: {result['pooled_whole_image']['invalid_fraction']:.2%}")
    if result["pooled_roi"] is not None:
        print(f"Caller-provided ROI invalid: {result['pooled_roi']['invalid_fraction']:.2%}")
    if result["camera"]["depth_scale_mm"] is None:
        print("Depth scale absent: metric values withheld; raw counts only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
