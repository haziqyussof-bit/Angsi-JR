# ANGSI JR. V8 · Reservoir Time Machine

Streamlit dashboard for the APGCE Challenge 3 baseline and monitor surveys, well logs, markers and checkshots. The application opens the supplied files read-only. Synthetic fixtures are created only by the tests.

## Install and run

Use Python 3.11 or 3.12. From the V8 folder:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export ANGSI_DATA_ROOT=/data/epic20/APGCE/input_data/challenge3
python -m streamlit run angsi_jr_v8.py --server.address 127.0.0.1 --server.port 8508
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and set the root with `$env:ANGSI_DATA_ROOT = 'C:\path\to\challenge3'`. The HPC default is the path in the command above. Access port 8508 through the HPC remote desktop or the site's configured JupyterHub proxy.

The V7 loader expects one baseline and one monitor SEG-Y in `volumes/`. Put the well list XLSX under `well_data/`, with the headers on Excel row 3. The loader expects `las/Well-N.las`, `checkshots/Well-N_Checkshot.txt` and `markers/Well-N_Markers.txt` inside `well_data/`. Review source datum, units, and marker headers before interpreting. The checkshot parser expects positive-down TVDSS in metres and TWT in milliseconds.

## Workflow

- **Overview and Seismic QC.** The inline and crossline panels share baseline/monitor amplitude limits. The selected trace links to its map position. QC modules 4.1–4.10 cover header geometry, paired keys, phase, repeatability, reference windows, lag, spectral matching, validity, before/after results, and correction review. Survey-wide maps are in **Opportunities**.
- **Horizon 2D & 3D.** Choose provided marker names for maps and surfaces. Review abrupt changes, competing peaks and the range across wider-search, smoothed and unsmoothed picks. Check TWT/TVDSS interpolation and use leave-one-well-out review for markers with three independent controls. The synthetic tie requires a validated compressional-slowness mnemonic, `DENS_P`, nearby baseline coverage and the well's own checkshot.
- **Well Correlation.** Fixed horizontal log tracks support vertical pan, zoom, marker flattening, and exaggeration. East–West seismic uses three stacked panels, multiple marker-guided reflectors, a TWT/TVDSS selector and projected well overlays. Vertical paths and MDKB–RT log depths are assumptions.
- **Opportunities.** Choose a fixed target or two provided markers with a correctly ordered interpreted interval. The latter uses the same baseline-conformable window for both surveys. Inspect signed monitor-minus-baseline, RMS, NRMS, relative RMS, predictability, shift, phase, reference quality, and sensitivity. Move a selected map location back to a section. Review the candidate map, ranked table and individual cards.
- **GeoCopilot.** Set `ANGSI_MODEL_ENDPOINT`, `ANGSI_MODEL_NAME` and, if required, `ANGSI_MODEL_API_KEY` in the process environment. The panel also accepts session settings. Choose and inspect a bounded read-only record before authorizing transfer to the endpoint. The request contains the current question and selected evidence. Raw traces, files and previous chat messages are excluded.

## Scientific methods and reproducibility

**Registration and masking.** Use exact IL/XL header pairs with duplicate rejection. Baseline CDP coordinates are scaled per SEG-Y header. Monitor XY is unresolved in the inherited loader. The QC footprint is the sampled paired-key distribution. NaNs and infinities are invalid. Isolated zero amplitude remains valid. The optional consecutive zero-run rule flags suspected padding, including internal runs. Set it to zero when the source has validated live zero runs. Independently validated mute-header interpretation is pending.

**Repeatability.** NRMS is `200 × RMS(M−B)/(RMS(B)+RMS(M))` on shared valid samples. Predictability uses squared normalized zero-lag inner product × 100, so it loses polarity sign. Signed reference phase uses analytic signals where the window is contiguous. The raw reservoir/reference NRMS ratio supplies local contrast and is unavailable when the denominator is zero or the reference fails quality gates. A rejected correction can still have raw metrics. [Kragh and Christie, 2002](https://doi.org/10.1190/1.1497316)

**Correction.** Confirm that the reference is non-reservoir and separated from the target by the required lag/filter guard. Default windows and thresholds are editable screening settings. Split the reference into fit and evaluation portions, search integer sample lags with common support, estimate scalar gain by least squares, and optionally apply a regularized nine-tap zero-phase amplitude filter. A positive shift samples the monitor later. A correction must satisfy lag agreement within one sample across the split, score separation, reference correlation, gain range and held-out NRMS criteria. The comparison uses common sample support. Corrections are applied in memory. Spatial coherence of the corrections and transferability across time require HPC review. [Rickett and Lumley, 2001](https://sep.stanford.edu/oldsep/james/papers/pdf/xequal.GP2001.pdf)

**Horizons.** Candidate reflectors use a well-guided trend and local extrema. Polarity can be constrained in the held-out review. Well ties based on borrowed checkshots and vertical depth assumptions are visible in the preview workflow. Conformable extraction requires two independent controls and rejects missing or crossing picks. Discontinuity flags identify abrupt changes for review. A depth-converted surface uses one reference checkshot. The displayed depth-conversion span compares overlapping own-checkshot curves at the same top TWT.

**Screening.** Four-connected sampled grid cells become union polygons, with holes retained. The polygons represent sample-cell support. Stable IDs are derived from class, stride and grid keys. Area and WGS84 export require a user-verified metre CRS and an affine IL/XL-to-XY fit that passes a 5 m residual gate. Areas are planimetric in that projected CRS. Relative TWT is a structural-position proxy. Nearby measured interval support uses valid fractional PHIE and VSHALE and a minimum coverage requirement. Proximity can be scored as distance to well control after CRS confirmation. Production and fluid evidence are missing. Low change alone provides no estimate of remaining hydrocarbons.

**Ranking and uncertainty.** The score is a weighted mean over available positive-weight components. The table reports the total weight of available evidence. Default quality cutoffs are PHIE ≥ 0.10 and VSHALE ≤ 0.40. Their sensitivity uses ±0.02 and ±0.05, respectively. The seismic component scores contrast within each screening class and discounts high reference NRMS. Horizon confidence is a heuristic derived from tracking sensitivity. Weight scenarios use 300 fixed-seed multiplicative draws over 0.8–1.2 and normalize each draw to 100%. Reported rank ranges describe sensitivity to those weights. Horizon envelopes, one-sample window perturbations, split-window lag difference, and checkshot spread are sensitivity diagnostics. The app produces no oil probability or probabilistic geological quantiles.

Use the CSV and processing JSON downloads together. The JSON records calculation version, selected controls and file path/size/mtime where applicable. For cross-machine replication, use the same inputs and Python environment. Cache keys for current loaders include source file stamps. GeoJSON uses WGS84 longitude/latitude.

## Verification

```sh
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m pytest -q
```

See [CHANGELOG.md](CHANGELOG.md) for test coverage and HPC validation still required.

## V8.1 extraction-mode patch (2026-11-11)

The **Opportunities** workflow now defaults to **Time window (TWT), 1500–1700 ms, 10 IL/XL**.
Other user-selectable modes are **Depth window (TVDSS)**, converted to time through a chosen well's own checkshot without extrapolation, and **Horizon-bounded**, requiring distinct, ordered valid interpreted surfaces.
Raw 4D attributes use the shared target window for both surveys.

The **Calculate survey diagnostics** control displays a coverage table *even when no valid cells remain*, reporting sampled paired/nonpaired IL/XL keys, missing/crossing horizons, reference-overlap guard, missing target samples and reference-QC exclusions. Download per-cell classification as `angsi_extraction_cell_qc.csv` and a processing manifest with mode, windows and exclusions. Screening-eligible means sufficient target/reference support and defined local contrast; the final opportunity region detector further requires correction `PASS` under its original strict gates.

The advanced reference window defaults to 700–950 ms as a **user-reviewable QC assumption**, not a confirmed non-reservoir interval. Survey acquisition mutes, spatial velocity validity, marker polarity, and production attribution remain unverified on the user's HPC.

Run on the HPC using `python -m pytest -q tests/test_extraction_qc.py` and compare map outputs before and after this patch. Source files were not modified in-place; the previous V8 folder can remain installed.

## V8.2 — ranked area and conditional volume workflow

The V8.2 main application filename remains **`angsi_jr_v8.py`** for compatibility. The new module is `opportunity_intelligence.py`.

1. In **Opportunities**, run an existing survey extraction and candidate screen, as in V8.1.
2. Enter a **verified projected-metre EPSG** from survey metadata. Physical polygon areas in km² are shown only if baseline XY geometry fits that projected CRS within the existing affine tolerance.
3. Review separate ranked tables: `Overall_rank` is evidence-weighted; `Area_rank` is mapped physical footprint size. If area cannot be verified, only sampled-cell count is reported, not an area estimate.
4. Under **Optional What-if Volume Potential & Sensitivity**, tick `Enable user-defined hydrocarbon pore-volume what-if analysis`. All property inputs start **blank**. Enter low / mode / high for net pay (m), porosity (fraction), and Sw (fraction) for each candidate. For a training/demo scenario only, explicitly tick `Populate clearly HYPOTHETICAL example inputs` and edit the values.
5. Review `Volume_rank`, `HCPV_P90_Mm3`, `HCPV_P50_Mm3`, `HCPV_P10_Mm3`, `HCPV_relative_spread` and `Volume_scenario_top_rank_share_pct`. Export the what-if table and processing manifest.
6. Use **Scientific uncertainty inventory** and **Opportunity uncertainty / evidence coverage** to see which facts are missing, including production, pressure, horizon control, physical area and local reservoir data.

### Interpretation

The model computes **HCPV** (hydrocarbon-filled pore volume) in *million reservoir m³*, using scenario assumed net pay, effective porosity and water saturation. It does not convert into barrels of oil, recovery factors, production forecasts or recoverable reserves; no Bo/Bg, PVT or field production calibration is assumed.

`P90/P50/P10` are *scenario* exceedance percentiles of artificial bounded triangular parameter ranges, **not statistically validated geological probabilities**. Different candidates are sampled independently and model covariance is absent. With equal scenarios for every polygon, candidate ranking by volume will closely track area rank. Actual well-constrained net pay, saturation, compartmentalization and production are needed before making field decisions.

### Release-specific checks

```bash
python -m compileall -q .
PYTHONPATH=. python -m pytest -q tests/test_v82_opportunity_intelligence.py
PYTHONPATH=. python -m pytest -q
```

The first two checks were executed in the release environment (eight new tests passed). The full integration suite requires the HPC Python environment with `streamlit` and `segyio` installed.

Suggested separate V8.2 port on HPC: **8511**. Keep V8.1 running on its existing port if desired.


## V8.3 — In-app parameter explanation tooltips

Every supported interactive Streamlit input, dropdown, slider, checkbox, toggle, radio, and action button now carries contextual **`help=`** hover text (the native **?** icon beside input labels). The explanations give the relevant geology or geophysics concept, expected units, parameter consequences, sensitivity, evidence checks, and caveats. All help is maintained in [`ui_help.py`](ui_help.py), which is safe to edit without changing scientific computation.

The optional HCPV what-if scenario **data editor** uses column-header hover help for **all nine editable low/mode/high inputs** (net pay, effective porosity, and water saturation). Streamlit's `st.chat_input` has no `help` argument, so a caption explains its behaviour directly above the message box.

**No QC formulas, screening thresholds, model calls, data loaders, or visualizations were intentionally changed in V8.3.** Numerical outputs should be identical for the same V8.2 dataset and inputs. The input hover tooltip is documentation, not proof of field accuracy.

### JupyterLab quick launch

Upload `ANGSI_JR_V8_3_FULL.zip` to `/data/epic20/APGCE/apgce_team_13/`, unzip there, and use:

```python
%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V8_3_FULL/requirements.txt
```

Then launch in a **Python notebook cell** (do not run `streamlit` with bare Python syntax):

```python
import sys, os, subprocess
from pathlib import Path
folder = Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V8_3_FULL')
app = folder / 'angsi_jr_v8.py'
env = os.environ.copy()
env['ANGSI_DATA_ROOT'] = '/data/epic20/APGCE/input_data/challenge3'
with (folder / 'server_v83.log').open('a') as log:
    process = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(app),
        '--server.address=127.0.0.1', '--server.port=8512', '--server.headless=true',
        '--server.fileWatcherType=none'], cwd=folder, env=env, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
print('PID:', process.pid)
```

Open `http://127.0.0.1:8512` in a browser **on the same remote HPC machine**, or use a configured JupyterHub proxy route.
