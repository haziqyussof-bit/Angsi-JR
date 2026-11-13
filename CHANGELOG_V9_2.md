# ANGSI JR. V9.2 changelog

- **Appearance** sidebar: Dark / Light / Custom palettes with text and accent colours.
- **Overview:** no interpreted seismic horizon overlay (well tops remain independently selectable).
- **Seismic QC:** GR left / RT right controls, without changing the existing fence/section layout.
- **Synthetic tie QC:** LAS curve inspection, manual real sonic/density mnemonic, unit validation, own-checkshot prerequisite, Ricker synthetic/trace plot and CSV export. Cannot synthesize from missing sonic data.
- **3D:** contours, contour interval, colour table, invert, cmin/cmax percentiles; vertical guides from zero to TD−RT with explicit caveat.
- **Well Correlation fluid track:** oil green, gas red, water muted grey.
- **Analysis:** horizon extraction footprint expanded to real survey paired cells, distinguishes observed reflectors and model-only trends; removes well-buffer island clipping, does NOT validate geology by smooth extrapolation.
- **Reservoir screening:** optional GR-derived shale index explicitly uncalibrated, independent-well validation enforced, optional descriptive no-label full-footprint seismic clusters.
- **Opportunities:** horizon confidence flags and evidence gap disclosure remain visible.

Tested with synthetic inputs; not verified on HPC Angsi volumes.
