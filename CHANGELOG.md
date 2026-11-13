## ANGSI JR. V8.3 — help and explanatory parameter tooltips

- Added `?` help tooltips to every supported Streamlit input across main dashboard, seismic QC, horizon reviewer, and opportunity screen, including dynamic ranking-component weights.
- Added dedicated domain definitions for TWT/TVDSS, sample/grid spacing, NRMS/correction gates, mute thresholds, horizon polarity/search, EPSG and candidate scoring.
- Added data-editor per-column help for all nine conditional HCPV scenario parameters.
- Added visible caption for the GeoCopilot chat input because Streamlit does not provide a native `help` parameter for this widget.
- Preserved V8.2 processing engine and parameter defaults.
- Updated launch name and documentation.

# ANGSI JR. V8 changelog

## V7 workflows retained

- The original V7 archive and extracted source are preserved in a separate folder.
- The five dashboard pages, navy/cyan style, ANGSI JR. artwork, event badge, right-hand GeoCopilot, paired seismic palettes and section-to-map selection are retained.
- The real SEG-Y, XLSX, LAS, marker and checkshot loaders remain the basis of the application.
- Fixed-X well tracks retain vertical pan, zoom, flattening, exaggeration, geological tie lines and model-derived fluid tracks.

## V8 changes

- Added QC 4.1–4.10, explicit reference and reservoir windows, documented early mute and suspected-padding masks, held-out NRMS gates, shift/gain/spectral trials, and raw versus corrected plots.
- Stacked the East–West sections with TWT/TVDSS switching, well paths, logs, provided tops and multiple marker-guided reflector overlays. Reversed the GR fill to the right of the curve.
- Added polarity-aware reflector previews, discontinuity review, horizon sensitivity envelopes, checkshot QC, held-out well validation and a synthetic-to-baseline tie.
- Added fixed and conformable 4D extraction, sampled-cell polygons, verified-CRS area, weighted ranking with missing evidence, and candidate score/rank sensitivity.
- Added versioned processing manifests, CSV and figure exports, conditional WGS84 GeoJSON, and endpoint-specific GeoCopilot evidence authorization.

## Verification

Syntax compilation passed for every project Python file. All 14 automated tests passed on Python 3.12 and Streamlit 1.67, including navigation through all five pages and fixed/conformable screening on test-only SEG-Y and well inputs.

The test suite covers exact header registration, invalid samples and live zeros, NRMS and predictability, disjoint window validation, shifts and spectral QC, checkshot interpolation, marker parsing, picking, petrophysics, polygon area, ranking, uncertainty, figure layout, and Streamlit startup with and without synthetic inputs.

## HPC validation required

- Confirm actual field file parsing and performance, well and SEG-Y coordinate datum/units, monitor geometry, true mute headers, survey coverage and sample axis.
- Review non-reservoir reference windows, spatial consistency of estimated corrections and transferability into the reservoir.
- Confirm checkshot and deviation datums, log units, tie wavelet and polarity, fault geometry, marker order, well-tie residuals and cutoff suitability.
- Validate the model connection on the approved HPC endpoint. Local tests use no external model requests.

## Missing field evidence

- Production, pressures, fluid contacts, independent saturation calibration and validated drainage polygons are unavailable in this workspace.
- Deviation file schema and coordinate conventions require verification. Displayed well paths are vertical assumptions.
- Candidate cells indicate local 4D screening classes. Their scores describe available evidence and sensitivity. This release makes no field discovery, fluid, reserves or remaining-oil claims.


## V8.2 — Candidate Area, Volume Sensitivity & Uncertainty (2026-11-11)

**Additions without rebuilding V8.1 UI**

- Friendly candidate map and evidence-card names `OPP-01`, `OPP-02`, ... determined by current weighted opportunity score; persistent `C-...` identifiers retained in exports to track changing filters/ranks.
- Separate, reproducible **overall opportunity rank**, **verified physical area rank (km²)**, and **conditional hydrocarbon pore volume scenario rank**. A sample-cell-count ranking is explicitly labelled a sampling proxy when CRS/geometry validation fails.
- Always-visible survey scientific uncertainty inventory and per-candidate evidence/uncertainty table, including absent reservoir, horizon, production, drainage and physical-area evidence.
- Independent, editable per-candidate what-if inputs for net pay, effective porosity and water saturation. Blank by default; clearly *hypothetical* demo values only if the user opts in.
- Transparent scenario calculation of hydrocarbon pore volume, HCPV [million reservoir m³] = polygon area [km²] × net pay [m] × effective porosity × (1−Sw).
- Seeded bounded triangular what-if Monte Carlo; show P90/P50/P10 *scenario exceedance values*, spread, relative spread and scenario top-rank share. Missing physical area or input parameters produce missing output, not fabricated numbers.
- Export scenario assumptions, ranked area/volume candidates, uncertainty and settings as CSV and JSON.
- Plot all polygon parts robustly and show concise `OPP-nn` labels instead of long internal hashes.

**Scientific limitations**

- What-if volumes are **not** actual Angsi STOIIP, GIIP, remaining hydrocarbons, recoverable reserves or a calibrated probability distribution.
- Polygon area requires user-confirmed projected-metre EPSG and an affine IL/XL→XY geometric validation. If this fails, physical areas and volumes remain unavailable.
- Candidate numbers are display order and change if filters/weights change; internal hash IDs preserve auditability.
- Shared hypothetical assumptions across polygons make volume ordering correlate mainly with physical area. Candidate-specific reservoir properties and production/pressure/well-test inputs would be required for field-defensible volumetrics and depletion risking.
- V8.1 QC and horizon assumptions remain unchanged. Actual HPC SEG-Y run, field data QC, and user acceptance testing are still required.

**Verification run in release environment**

- V8.2 module: eight new isolated synthetic-fixture tests passed.
- All project Python files compiled successfully.
- Full legacy Streamlit/SEG-Y integration tests could not run here because `streamlit` and `segyio` are unavailable and package installation could not reach the index. Run them on the user's HPC using `python -m pytest -q`.
