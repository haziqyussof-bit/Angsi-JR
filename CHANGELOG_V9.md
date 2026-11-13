# ANGSI JR. V9.0 — Analysis Pipeline

Based on uploaded `ANGSI_JR_V8_3_FULL(1).zip` (30 files). V8.3 source retained.

- NEW `analysis_engine.py`: finite-sample/mute-aware 17 seismic attribute computations; extraction from actual paired SEG-Y, corrected difference only on QC PASS, window perturbation diagnostic.
- NEW `analysis_dashboard.py`: Workflow 1 seismic attribute 3-way maps, time/depth/horizon extraction, per-cell QC and export; Workflow 2 independent well-log reservoir screening and QC; V9 opportunities overlay with polygons and ranking uncertainty.
- NEW `angsi_jr_v9.py`: original V8.3 UI plus six ordered pages, renamed AI Horizon, linked Analysis and Opportunities.
- MODIFIED `fence.py`: display-only vertical exaggeration for E–W fence.
- CARRIED FORWARD all original V8.3 scientific modules and tests, branding images and favicon.
- UI: optional light mode, local seismic palettes, 1–5× Y-only display zoom, measured marker overlay and well availability QC.
- NEW tests: six synthetic tests including fake paired SEG-Y extraction.

Scientific blockers: no confirmed production/pressure/dynamic data, missing well-3 seismic coverage, incomplete marker/checkshot control, sparse well labels, uncertain velocity/deviation, horizon uncertainty and survey XY validation. Missing evidence is never silently replaced with inferred oil or gas.
