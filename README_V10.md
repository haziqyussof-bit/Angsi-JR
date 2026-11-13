# ANGSI JR. V10 — Complete Streamlit Dashboard

**Source baseline:** `ANGSI_JR_V9_3_1_FACIES_FULL.zip` uploaded by the team.

V10 preserves the six existing navigation tabs, 4D QC, full-footprint attribute extraction, well correlation, AI horizon previews, facies proxy, opportunity ranking and GeoCopilot. It adds the most recent requested features rather than rewriting the app from scratch.

## New features

1. **Analysis facies colors** — Bright / sand-prone hypothesis = **yellow**, intermediate / uncertain = **green**, dim / shale-prone hypothesis = **grey**. Only display colors changed; P30/P70 percentile algorithms and full-survey sample masks are unchanged.
2. **East–West arbitrary fence extension** — default 1,250 m beyond BOTH outermost wells (slider 0–3,000 m). Extends route through **actual paired traces only**; does not invent data beyond seismic footprint. Original well stations remain correctly located along the longer fence.
3. **GeoCopilot organiser model diagnostics** — explicit private-LAN HTTP opt-in, model listing test and a basic text-only inference test. Organiser endpoint default `http://172.28.34.1:8000/v1`; model default `google/gemma-4-26B-A4B-it`. If organiser has changed these, update fields. Private HTTP is *not encrypted*. Only enable if organisers/team explicitly authorise use. No seismic data is sent by either test; selected evidence is only sent by a user chat after authorisation. This is Gemma, **not OpenAI ChatGPT**.
4. **Well Correlation display** — individual curve color pickers (GR/RT/RHOB/NPHI/PHIE/SWE), shared track-width control and curve thickness control. Keeps left-to-right horizontal well positions fixed. Fluid track now ONLY fills **oil-model in green, gas-model in red**, leaving water/uncertain/other model classes visually unfilled. These are log-derived hypotheses, not confirmed oil/gas tests.
5. **Well Correlation AI horizon overlays** — user can multi-select interpreted surfaces separately from measured marker tops. Surface-to-well projections are shown as tentative; lines use dashed style for extrapolation or uncertain checkshot. Restricted to real paired grid support with no invented out-of-footprint reflectors. Expensive horizon picking is only done when the toggle is on.
6. **Two-panel Synthetic Tie** — keeps the existing trace-vs-Ricker synthetic QC panel, adds **a 1995 seismic inline with variable-area synthetic wiggle at the selected well**. User controls wiggle scale, display time shift, polarity; legends on top avoid clipping. Requires actual sonic, density and OWN checkshot. Results survive other widget reruns within the session.
7. **Opportunities multi-layer map** — if Analysis Workflow 2 has created amplitude facies, `Facies background + 4D top` supports a 0–100% **4D transparency** slider; no false spatial resampling is performed because both maps use the same sampled IL/XL keys. Can still independently show/hide well controls and candidate outlines. Top candidate labels limited to 0/5/10/20/50 rather than hundreds of overlapping annotations.
8. **Opportunity bubble map** — candidate region centroids scaled by physical footprint area *only when* baseline trace XY headers pass an affine geometry consistency check; otherwise sized by sampled cell count. No hydrocarbon volumes inferred from bubble sizes.
9. **Top 3 evidence summary** — rank, relative evidence score, ranking sensitivity and a status for STOIIP/GIIP/BOE (not estimated until user inputs geological scenario).
10. **Volumetric scenarios** — explicit inputs for hydrocarbon-bearing area (km²), gross thickness, NTG, PHIE, Sw, Bo, Bg, fluid phase and gas-to-oil conversion. Computes GRV, HCPV, in-place **STOIIP (MMstb)**, **GIIP (Bscf)**, **in-place MMboe**, and illustrative P90/P50/P10 sensitivities from triangular distributions. Candidate polygon area is a 4D anomaly footprint and **not structural closure**; it is merely an input suggestion. User must acknowledge limits before calculation. NOT reserves or recoverable volumes. No numeric volumes are fabricated from raw 4D amplitude.

## JupyterLab HPC — every step until website opens

Use **the Jupyter Notebook running on the same HPC system**, not a local Python installation.

### 1. Upload

Upload `ANGSI_JR_V10_FULL.zip` into:

`/data/epic20/APGCE/apgce_team_13/`

### 2. Extract (new Python notebook cell)

```python
from pathlib import Path
import zipfile
ROOT = Path('/data/epic20/APGCE/apgce_team_13')
ZIP = ROOT / 'ANGSI_JR_V10_FULL.zip'
assert ZIP.is_file(), f'Please upload ZIP here: {ZIP}'
with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    assert bad is None, f'Corrupt archive: {bad}'
    z.extractall(ROOT)
APP_DIR = ROOT / 'ANGSI_JR_V10_FULL'
APP = APP_DIR / 'angsi_jr_v10.py'
assert APP.is_file(), f'Application missing: {APP}'
print('V10 extracted:', APP)
```

### 3. Install dependencies (new cell)

```python
%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_FULL/requirements.txt
```

If new Python modules were installed for the first time, restart the Jupyter kernel and proceed to Step 4. Don't use `%run angsi_jr_v10.py` — run it as a Streamlit server.

### 4. Launch on port 8517 (new cell)

```python
import os, sys, subprocess, py_compile
from pathlib import Path
APP_DIR = Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_FULL')
APP = APP_DIR / 'angsi_jr_v10.py'
py_compile.compile(str(APP), doraise=True)
LOG = APP_DIR / 'server_v10.log'
ENV = os.environ.copy()
ENV['ANGSI_DATA_ROOT']='/data/epic20/APGCE/input_data/challenge3'
with LOG.open('a') as f:
    process_v10 = subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', str(APP),
        '--server.address=127.0.0.1', '--server.port=8517',
        '--server.headless=true','--server.fileWatcherType=none'
    ], cwd=APP_DIR, env=ENV, stdout=f, stderr=subprocess.STDOUT,
       start_new_session=True)
print('ANGSI JR V10 PID', process_v10.pid, 'log', LOG)
```

### 5. Check local server (new cell)

```python
import urllib.request
from pathlib import Path
try:
    with urllib.request.urlopen('http://127.0.0.1:8517/_stcore/health', timeout=6) as r:
        print('HTTP',r.status,r.read().decode())
except Exception as e:
    print('Health check failed:', e)
    log = Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_FULL/server_v10.log')
    if log.exists():print(log.read_text(errors='replace')[-8000:])
```

### 6. Open website

From **Firefox on the SAME remote HPC desktop (RustDesk)**, open:

**http://127.0.0.1:8517/**

If the browser is elsewhere, `127.0.0.1` refers to that other host. If the JupyterHub proxy extension is enabled, try your JupyterHub base URL plus `/user/apgce_team_13/proxy/8517/`; otherwise use an IT-approved tunnel or a browser on the same HPC host. HTTP health 200 on a compute node is not proof that Firefox on a different machine can see it.

### 7. If it does not open

Run the health check again, then print the last 8,000 characters of `server_v10.log`. Errors such as missing `segyio`, `streamlit`, `lasio`, invalid `ANGSI_DATA_ROOT`, and network routing are diagnosed separately. The app is read-only on the seismic input data.

## V10 test plan after opening

1. **Analysis → Workflow 1**: TWT 1500–1700 ms, sampling 10, Run. Workflow 2: bright=P70/dim=P30, Run. Verify full-footprint categorical map: yellow bright, green intermediate, grey dim.
2. **Overview/Seismic QC → E→W fence**: adjust extension 0–3000 m, verify both ends extend only where paired traces exist and well marker stations are retained.
3. **AI Horizon interpretation → Synthetic Seismic Tie**: choose a real well with sonic/density and OWN checkshot; Run, inspect two independent panels, visible legends, variable-area wiggle, and signed correlation. There is no valid true synthetic without sonic.
4. **Well Correlation**: adjust colors, shared track width, line thickness. Enable interpreted horizon overlay. Test fixed horizontal positions with vertical scrolling. Confirm oil green, gas red, water/uncertain unfilled.
5. **Opportunities**: build candidate polygons using a whole-survey 4D attribute window, enable facies bottom/4D top map with transparency slider, inspect Top 3 and bubble map, then choose candidate and enter explicit volumetric sensitivity inputs.
6. **GeoCopilot**: organiser endpoint default is a private HTTP address. Only if authorised, check the private HTTP opt-in and general authorization boxes, Test Model Connection and basic chat. The test does not upload well/seismic evidence. Ask field-specific questions only after selecting a supported read-only evidence record and verifying organisational permission.

## Scientific scope / known limitations

- All seismic attributes run on **sampled** paired survey coverage as configured, not literally all original traces unless grid spacing is 1. Horizon mode can contain model-only/exploratory picks. Per-cell masks remain respected.
- Cross-equalization and classification are screening workflows; high 4D contrast does not automatically indicate production, and low contrast does not automatically indicate unswept hydrocarbons.
- Volumetric equations depend on assumed reservoir geometry, hydrocarbon contacts, petrophysics and PVT. **Anomaly candidate polygons are not mapped hydrocarbon closures.** The outputs are illustrative in-place scenarios only and deliberately require user confirmation. P90/P50/P10 are sensitivity percentiles based on assumed distributions, not probabilistic reserves assessments.
- Oil / gas fluid labels are interpreted log-model indicators, not definitive formation tests. Where data lacks interpreted VOIL/VGAS, no fake fluid indicator is painted.
- The internal organiser model is not OpenAI ChatGPT. Connection tests must be run from a network that can reach the endpoint. No API keys are stored in files or included in this ZIP.
- This package includes source code and visual assets but **no Angsi SEG-Y, wells or confidential data**.

## Verification performed for this package

- Python compilation of every `.py` file; static source checks.
- `pytest -q tests/test_v10_science.py` — 10 V10-specific tests passed locally.
- ZIP CRC/integrity test.
- No full Streamlit runtime and no real SEG-Y loading performed locally (the real data is on the team's HPC and the local test runtime has no `segyio` or `streamlit`). Follow the exact test plan on HPC and share a traceback if there is a startup issue.
