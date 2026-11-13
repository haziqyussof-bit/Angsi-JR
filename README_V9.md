# ANGSI JR. V9 — Integrated Reservoir Analysis (V8.3 upgrade)

## What is this release?

This release **uses the actual user-supplied ANGSI_JR_V8_3_FULL ZIP as its baseline**. It does not replace the Angsi exploration/science modules with a standalone demo. Original V8.3 code is retained at `angsi_jr_v8.py`; the new entry point is `angsi_jr_v9.py`.

### Navigation

1. Overview — source data, 3 vintage panels, wells, markers and data availability.
2. Seismic QC — section/fence navigation, horizon review, existing QC 4.1–4.10.
3. **AI Horizon 2D & 3D Interpretation** — original V8.3 controlled horizon previews and 3D scene.
4. Well Correlation — fixed horizontal well positions; oil-model GREEN, gas-model RED, water-model BLUE; fluid type remains a model indication.
5. **Analysis (NEW)** — two-stage interpretation: seismic attribute extraction → well-conditioned reservoir/non-reservoir screening.
6. **Opportunities** — V9 integrated evidence map/true sampled-cell polygons and ranking/uncertainty ABOVE the intact independent V8.3 QC/opportunities module.

Existing 4D QC, polygons, HCPV what-if and sensitivity models remain in the copied V8.3 code. Original dataset is read-only.

## Analysis Workflow 1 — seismic attributes

1. Choose **Time (TWT), Depth (TVDSS), or Interpreted horizon pair**. The default is 1500–1700 ms at 10 IL/XL. The depth option uses a named, available checkshot only; no extrapolation.
2. For horizon-bounded extraction, select different top/base markers previously available in **AI Horizon**. The engine reuses the baseline horizon previews and their strict `confident` plus continuity masks. **Missing and crossed picks are excluded**. Monitor and baseline are extracted using *the same baseline-interpreted window* so independent peak selection does not manufacture production movement.
3. Optional advanced settings: 700–950 ms reference (user-reviewable, NOT verified non-reservoir), zero-padding heuristic, registration lag, strict held-out correction test, spectral matching, verified acquisition mute. For speed, use a coarse 15 IL/XL grid and disable optional full correction. 5 IL/XL can be expensive on the HPC.
4. Click **Calculate seismic attributes**. Results show baseline, monitor and raw trace-difference maps, and optionally corrected-difference attributes for traces passing V8.3 cross-equalization QC. A single attribute definition is applied identically to both surveys; baseline/monitor use a shared amplitude color scale.
5. Review *valid sampled paired cells*, excluded QC reasons, sample coverage, correctable traces, and source manifest. Download analysis attributes, cell QC and configuration JSON.

**Available actual trace attributes**: RMS, signed mean, mean absolute, maximum positive, most negative, absolute peak, trace energy, amplitude variance, mean/max envelope, circular instantaneous phase, dominant frequency, spectral amplitudes near 10/25/50 Hz, skewness, excess kurtosis. These are concrete algorithms from provided SEG-Y **not every proprietary attribute offered in Petrel**. Their units, window widths, and processing history matter. 4D **RMS of (monitor − baseline)** is not the same as `RMS(monitor) − RMS(baseline)`.

## Analysis Workflow 2 — well-log facies screening

This runs **only after Workflow 1**. It extracts per-well net-reservoir fraction from measured PHIE+VSHALE data (not invented log values). Markers used for the well training interval are independent of seismic predictions; depth/time conversion uses own checkshots where needed. One well supplies **one** training label, not hundreds of pseudo-independent depth samples. Well-3 is excluded from training if its nearest seismic trace remains beyond 100 m.

Input labels are RESERVOIR (fraction >=0.55), NON_RESERVOIR (<=0.20), or AMBIGUOUS. The illustrated cutoffs are **reviewable assumptions**, not field-validated net-pay definitions. A simple regularized baseline attribute similarity model uses B_RMS, B_EnvelopeMean and B_MostNegative, evaluated **leave-one-well-out**. It publishes exploratory classes only if >=2 independent wells of each class and cross-validation balanced accuracy >=0.60. Scores are not calibrated probabilities, and **cannot identify oil vs gas or remaining oil**. If source LAS has no VSHALE or available wells do not support both classes, the UI correctly reports **INSUFFICIENT/NOT VALIDATED** and withholds sand/non-sand labels.

## Opportunities — analysis-linked evidence

- First calculate Analysis; then open Opportunities. Use **Local_contrast, 1995 RMS, 2006 RMS, raw 4D RMS, signed 4D mean, most negative 4D amplitude, reference NRMS**, or exploratory well-conditioned reservoir similarity as the base map.
- Overlay candidate polygons, well labels, optionally valid facies markers, and optional three-way RMS maps.
- Polygons are *unions of real supported sampled grid cells* and preserve missing-data holes; **no geometry is invented outside surveyed/paired grid**.
- Ranking combines available reservoir similarity evidence, local 4D contrast, repeatability quality and proximity to well control. Missing components remain missing and reduce evidence coverage, not a false probability.
- **300 fixed-seed weight perturbations** quantify rank sensitivity (best/worst rank and first-place scenario share). **NOT hydrocarbon occurrence probability**; missing geological/production/PVT evidence explicitly listed. Opportunity polygon coordinates are IL/XL grid, not georeferenced km² without a validated projected CRS.
- The V8.3 opportunity code is retained under **V8.3 independent 4D QC and legacy opportunity screening**. It still has full existing ranked area + conditional HCPV scenarios when user provides verified geometry and hypothetical assumptions.

## V9.0 UI refinements

- Sidebar shows **V9.0**, not seismic color dropdown. The palette dropdown lives near each Overview / Seismic QC section; vertical display exaggeration 1–5× applies only to Y-axis, not sample times or lateral extents. E–W fence remains stacked **1995 / 2006 / Difference**.
- Well log preview can show **actual markers** (select multiple) and a measured-marker/checkshot table; Well-3's missing measured tops are not fabricated.
- Header top padding and an optional experimental **Light mode** toggle; seismic Plotly canvases remain dark for reflectivity contrast.
- Well correlation keeps actual E→W position, fixed X, right-hand GR fill, Oil-model GREEN / Gas-model RED. Fluid model labels are not well tests.

## Installation on the team's HPC / Jupyter

1. Upload `ANGSI_JR_V9_ANALYSIS_FULL.zip` to `/data/epic20/APGCE/apgce_team_13/`.
2. In a Python notebook cell:

```python
from pathlib import Path
import zipfile
root = Path('/data/epic20/APGCE/apgce_team_13')
with zipfile.ZipFile(root/'ANGSI_JR_V9_ANALYSIS_FULL.zip') as archive:
    archive.extractall(root)
app = root/'ANGSI_JR_V9_ANALYSIS'/'angsi_jr_v9.py'
print('App:', app.exists())
```

3. Install on that Jupyter kernel: `%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V9_ANALYSIS/requirements.txt`
4. Launch **via `sys.executable -m streamlit run`**, not `%run app.py`:

```python
import sys, os, subprocess
folder=app.parent
env=os.environ.copy()
env['ANGSI_DATA_ROOT']='/data/epic20/APGCE/input_data/challenge3'
with (folder/'server_v9.log').open('a') as logfile:
    proc=subprocess.Popen([sys.executable,'-m','streamlit','run',str(app),
        '--server.address=127.0.0.1','--server.port=8513','--server.headless=true',
        '--server.fileWatcherType=none'],cwd=folder,env=env,stdout=logfile,
        stderr=subprocess.STDOUT,start_new_session=True)
print('PID',proc.pid)
```

Open `http://127.0.0.1:8513` inside the **same HPC remote desktop**. For JupyterHub use `/user/apgce_team_13/proxy/8513/` only if Jupyter server proxy is enabled. Health-check 200 is not proof that external browser routing works.

## Validation / status

`python -m compileall -q .` and `PYTHONPATH=. pytest -q tests/test_v9_analysis.py` are the release-level non-HPC checks. Real-data performance, SEG-Y offsets, horizon continuity, facies calibration, surface QC, online GeoCopilot and full integration tests **must be verified on the user's HPC**. In this packaging environment Streamlit/segyio/lasio are unavailable (offline), so the actual web app is **not claimed to have been run**.

No actual Angsi SEG-Y or confidential LAS files are bundled. Do not disclose field metadata to an unapproved third-party AI endpoint. Standard ChatGPT Plus subscription is not an OpenAI API credential.
