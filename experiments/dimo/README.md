# DIMO reproduction

This directory contains the runnable, CPU-only experiment for DIMO's selected
real L515 and D415 scenes. DIMO source RGB/depth images and CAD files are not
committed. Download selected files with the scripts in this directory; keep
the published CC BY 4.0 attribution and resulting provenance manifests.

The `reference_results/` directory contains only derived JSON summaries and
the attribution-bearing diagnosis; see the image at `../../docs/images/`.
Current reported result: 99.70% stored zeros in L515 CAD-derived part interiors
and 77.28% in D415 part interiors. Possible L515 range cutoff remains a
hypothesis because saved depth scale and acquisition settings are missing.
