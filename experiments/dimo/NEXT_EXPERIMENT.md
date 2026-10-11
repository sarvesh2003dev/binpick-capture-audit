# Next experiment: distinguish capture failure from material failure

The existing experiment demonstrates an audit method. It does not demonstrate
that synthetic data improves a perception model or that a customer will pay.

## Immediate decision

Keep the original dropout helper only for testing how software handles nearly
absent depth. Do not fit a material model, train a depth-completion model, or
buy a GPU based on this sample's missingness rate.

## Smallest controlled capture

Ask a camera-equipped robotics integrator for a reflective machined part and a
matte reference at the same range and lighting. Use at least three independently
rearranged piles, several angles, and two distances safely beyond the configured
minimum range. This is a diagnostic pilot, not sufficient model validation.

Capture and preserve:

- Original RGB and uint16 depth, with frame timestamps.
- Exact camera model, firmware, preset, exposure, gain, projector power,
  confidence/filter settings and configured minimum range.
- Saved depth units, native image dimensions and intrinsics.
- RGB/depth extrinsics and whether/how depth was registered to RGB.
- Part CAD, dimensions and the annotation/calibration procedure.
- One concrete downstream failure: missed detection, wrong pose, unsafe grasp
  candidate or missing-depth crash; include the baseline output.

Compare missingness and depth error only after verifying units and alignment.
Use a matte target to check range/configuration, then metallic targets at the
same geometry. Preserve raw depth; never evaluate inpainted data as measured
ground truth. Split later training and tests by physical arrangement.

## Small deliverable worth testing with a buyer

A capture-and-perception failure audit: attempt to reproduce the supplied
failure, measure supported observations, test hypotheses and specify the
smallest follow-up capture that separates unresolved explanations. Deliver
evidence plus a regression case. Agree on this scope and its acceptance
criterion before expanding the simulator; a root cause is not guaranteed.

Only offer synthetic-data improvement as a later deliverable when a held-out
real test set demonstrates improvement over the customer's existing baseline.
Public datasets cannot establish that buyer demand exists. No outreach has been
sent, customer agreement obtained or revenue validated.
