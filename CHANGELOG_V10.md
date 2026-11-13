# V10 release notes — 12 November 2026

Baseline: ANGSI JR V9.3.1 FACIES, including original user's organiser Gemma GeoCopilot and full-survey Analysis/Opportunities.

Changes:
- Facies colors: bright yellow, intermediate green, dim grey.
- Extended E–W fence beyond end wells using actual paired seismic station lookups.
- GeoCopilot connection tester, optional explicit private-organiser-LAN HTTP mode (without silently weakening general HTTPS restrictions), basic inference check; current model evidence controls preserved.
- Configurable well correlation track widths and per-curve colors/thickness, fixed X, only oil green and gas red shaded, multiple AI interpreted horizon toggles.
- Synthetic seismic-to-well tie now includes a true separate inline seismic visualization with synthetic wiggle and positive variable-area fill, with legends above both panels. Existing QC unchanged.
- Opportunities: facies background + 4D adjustable transparency, top 10 candidate labels default, top 3 summary, bubble map based on candidate centers and optional affine-geometric footprint area.
- Volumetric engine: explicit reservoir/phase/scenario inputs, STOIIP MMstb, GIIP Bscf, in-place MMboe, illustrative P90/P50/P10 distributions; no fabricated reserves.
- Jupyter launch workflow port 8517, README and V10 synthetic test coverage.

Limits: 4D detection ≠ production evidence, facies proxy ≠ confirmed lithology, and 4D polygon area ≠ geological trap closure. Runtime test with Angsi HPC SEG-Y remains pending.

V10 entrypoint: `angsi_jr_v10.py`; `angsi_jr_v9.py` retained as compatibility copy in source bundle.
