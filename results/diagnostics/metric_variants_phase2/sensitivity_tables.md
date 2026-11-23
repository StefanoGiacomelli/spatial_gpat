**Table A -- metric-variant sensitivity, ExpA (STARSS23 test).** Same predictions, different evaluation protocols (first row = headline). `tau` = 0.5 (fixed) and the post-hoc test-optimal tau of each row (diagnostic, not an operating point). S/D/I = per-segment substitution/deletion/insertion rates of the DCASE-style rows.

| variant | ER | F | LE (deg) | LR | SELD | S / D / I | best tau (post-hoc) | SELD at best tau |
|---|---|---|---|---|---|---|---|---|
| A  repository protocol: 10-ms frames, windows pooled (4x overlap), gated 20 deg  [headline] | 0.816 | 0.235 | 12.48 | 0.108 | 0.635 |  | 0.30 | 0.607 |
| B  windows stitched (reference-free), each frame once, same gated metric | 0.814 | 0.241 | 12.47 | 0.109 | 0.633 |  | 0.30 | 0.596 |
| B2 as B, exact 100-ms label bins (no float artefact) | 0.814 | 0.241 | 12.47 | 0.109 | 0.633 |  | 0.30 | 0.596 |
| C  as B, tPIT-aligned slots (upper bound, as main_demo.py) | 0.811 | 0.247 | 12.47 | 0.110 | 0.631 |  | 0.30 | 0.592 |
| D  stitched, 100-ms frames, gated frame-level metric | 0.813 | 0.242 | 12.44 | 0.109 | 0.633 |  | 0.30 | 0.596 |
| S1 100-ms segments, DCASE-style (ungated LE/LR, macro-F, S/D/I; public-code arithmetic) [re-impl.] | 0.752 | 0.141 | 44.81 | 0.273 | 0.647 | 0.15 / 0.40 / 0.21 | 0.40 | 0.630 |
| S2 1-s segments, repository components (gated LE/LR, pooled F) | 0.832 | 0.269 | 12.63 | 0.125 | 0.627 | 0.37 / 0.41 / 0.05 | 0.45 | 0.627 |
| S3 1-s segments, gated LE/LR, macro-F | 0.832 | 0.150 | 12.63 | 0.125 | 0.657 | 0.37 / 0.41 / 0.05 | 0.50 | 0.657 |
| S4 1-s segments, DCASE-style (ungated LE/LR, macro-F, S/D/I; public-code arithmetic) [re-impl.] | 0.735 | 0.151 | 45.70 | 0.325 | 0.628 | 0.18 / 0.30 / 0.25 | 0.45 | 0.625 |
| S4s S4 with stricter bookkeeping (never-co-active class = miss + spurious; macro over classes with references) | 0.735 | 0.150 | 45.70 | 0.330 | 0.627 | 0.18 / 0.30 / 0.25 | 0.45 | 0.624 |
| S4m S4 with segment-mean-direction instance angles (sensitivity) | 0.726 | 0.150 | 45.94 | 0.322 | 0.627 | 0.17 / 0.31 / 0.25 | 0.45 | 0.623 |

**Table B -- SELD score at the fixed threshold 0.5 under each protocol.**

| variant | ExpA (STARSS23 test) | mixed_focal reference (STARSS23 test) | ExpB (STARSS23 test) | TAU2019-ExpA (TAU2019 eval) | NIGENS2021-ExpA (NIGENS2021 test) | NIGENS2020-ExpA (NIGENS2020 test) |
|---|---|---|---|---|---|---|
| A  repository protocol: 10-ms frames, windows pooled (4x overlap), gated 20 deg  [headline] | 0.635 | 0.666 | 0.663 | 0.142 | 0.537 | 0.576 |
| B  windows stitched (reference-free), each frame once, same gated metric | 0.633 | 0.659 | 0.658 | 0.138 | 0.523 | 0.566 |
| B2 as B, exact 100-ms label bins (no float artefact) | 0.633 | 0.659 | 0.658 | 0.138 | 0.523 | 0.566 |
| C  as B, tPIT-aligned slots (upper bound, as main_demo.py) | 0.631 | 0.659 | 0.655 | 0.137 | 0.521 | 0.563 |
| D  stitched, 100-ms frames, gated frame-level metric | 0.633 | 0.658 | 0.658 | 0.134 | 0.523 | 0.565 |
| S1 100-ms segments, DCASE-style (ungated LE/LR, macro-F, S/D/I; public-code arithmetic) [re-impl.] | 0.647 | 0.642 | 0.695 | 0.136 | 0.497 | 0.535 |
| S2 1-s segments, repository components (gated LE/LR, pooled F) | 0.627 | 0.671 | 0.657 | 0.130 | 0.509 | 0.547 |
| S3 1-s segments, gated LE/LR, macro-F | 0.657 | 0.696 | 0.688 | 0.133 | 0.515 | 0.550 |
| S4 1-s segments, DCASE-style (ungated LE/LR, macro-F, S/D/I; public-code arithmetic) [re-impl.] | 0.628 | 0.633 | 0.679 | 0.118 | 0.466 | 0.498 |
| S4s S4 with stricter bookkeeping (never-co-active class = miss + spurious; macro over classes with references) | 0.627 | 0.631 | 0.679 | 0.116 | 0.464 | 0.498 |
| S4m S4 with segment-mean-direction instance angles (sensitivity) | 0.627 | 0.630 | 0.674 | 0.121 | 0.464 | 0.497 |

Segment-based rows (S*) are a re-implementation of DCASE-style scoring: the arithmetic of S1/S4 was verified against the public DCASE-2022/2023 baseline implementation on random inputs (maximum difference 2e-8; `analysis/diagnostics/check_against_public_dcase.py`), the frame preparation (10-ms outputs pooled to 100-ms frames, thresholding, slot order) is this repository's; they are not the official evaluation pipeline (definitions: `analysis/diagnostics/seld_variants.py`, `docs/CONVENTIONS.md` section 4.2).
