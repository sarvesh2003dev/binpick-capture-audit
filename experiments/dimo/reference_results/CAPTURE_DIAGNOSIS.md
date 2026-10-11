# Capture diagnosis

This report separates stored-data observations from a sensor-configuration hypothesis.

| Camera | Views | Zero inside CAD interiors | Zero across image |
|---|---:|---:|---:|
| real_l515 | 13 | 99.70% | 96.41% |
| real_d415 | 13 | 77.28% | 41.00% |

L515 hypothesis: 12 views share raw minimum 1960. **If** each stored unit represents 0.25 mm, that boundary is 490.0 mm. Every projected object interior is closer than that boundary. A minimum-range setting is a plausible explanation, not a confirmed cause.

[The L515 guide](https://support.realsenseai.com/hc/en-us/article_attachments/360081846174) documents that Min Distance sets nearer values to zero. Actual dataset settings and scale are missing.

**Decision: HOLD_FOR_SENSOR_METADATA.** Do not fit a material-specific dropout model from these captures yet.

Required before calibration: stored depth units, capture preset/minimum range, exposure/confidence settings, verified RGB/depth registration, and multiple independent object arrangements.

The previous constant-dropout profile remains useful only as an almost-total-depth-loss software stress test.

DIMO: Peter De Roovere, Steven Moonen, Nick Michiels, Francis Wyffels. CC BY 4.0. https://pderoovere.github.io/dimo/usage/ . Counts, masks, crops and overlays are derived outputs.
