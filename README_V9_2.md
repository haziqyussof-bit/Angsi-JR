# ANGSI JR. V9.2 — Full Surface Extraction & Interface Patch

**Base:** ANGSI JR. V9.1. **Scope:** changes requested after V9.1; retains the other six-navigation workflow.

## What changed

1. Sidebar **Appearance** supports Dark, Light and Custom colours for page, text and accent. Seismic palettes remain independent.
2. Overview does **not** plot interpreted seismic horizon overlays. Geological interpretation is under **AI Horizon 2D & 3D Interpretation**.
3. Seismic QC adds a collapsible **Well log display controls** section: GR left of intersecting wells; logarithmic RT right. Existing well top, section/footprint and 4D controls preserved.
4. Seismic-to-well tie: user can inspect actual LAS mnemonics and log QC, choose a real sonic and density mnemonic, declare sonic slowness units, and build a Ricker normal-incidence synthetic **only when measured controls suffice**. No real sonic = no synthetic (GR is NOT a substitute). This is preliminary, without extracted wavelet, mistie optimization or phase calibration.
5. 3D horizon viewer: controllable contours/interval, colour palette, inversion and percentile clipping. Wells display **assumed vertical paths from zero down to documented TD−RT** (not measured deviations); outside-seismic wells use an XY-based projected position with warnings.
6. Fluid colours: **Oil-model green, Gas-model red, Water-model neutral grey**; these are *screening flags* based on supplied interpreted petrophysical logs, not tested fluids.
7. Analysis **horizon-bounded** extraction evaluates the **whole sampled valid paired survey footprint**; tentative reflector picks and trend-only extrapolations are clearly tagged `EXPLORATORY_REFLECTOR` / `MODEL_ONLY` rather than omitted around well buffers. Only `HIGH_CONFIDENCE` cells approach tie-supported status; even those need independent geological review.
8. Reservoir screening: actual well-log PHIE/VSHALE remains preferred; if VSHALE is missing, optional **uncalibrated GR index** can be used *with explicit flag*. If leave-one-well-out validation fails, the app refuses validated reservoir labels. Optional *unvalidated seismic facies clustering* is available whole-survey, but cannot be called sand/oil/gas.
9. QC summary and candidate screening report horizon confidence instead of falsely implying that a smooth, full map is interpreted geology.

**Do not interpret low 4D response as unproduced oil without production/pressure/rock-physics evidence.**

## Install on APGCE HPC/JupyterLab (step by step)

1. Upload `ANGSI_JR_V9_2_FULL_PATCH.zip` through JupyterLab to `/data/epic20/APGCE/apgce_team_13/`.
2. Open a new Python notebook in JupyterLab. Extract with:

```python
from pathlib import Path
import zipfile
ROOT=Path('/data/epic20/APGCE/apgce_team_13')
with zipfile.ZipFile(ROOT/'ANGSI_JR_V9_2_FULL_PATCH.zip') as archive:
    archive.extractall(ROOT)
APP=ROOT/'ANGSI_JR_V9_2_PATCH'/'angsi_jr_v9.py'
assert APP.is_file(), APP
print('App ready:', APP)
```

3. Install dependencies in a NEW notebook cell:

```python
%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V9_2_PATCH/requirements.txt
```

4. Use `JUPYTER_START_V9_2.ipynb` in the extracted folder, **or** start the server in a notebook code cell:

```python
import os, sys, subprocess
from pathlib import Path
APP=Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V9_2_PATCH/angsi_jr_v9.py')
LOG=APP.parent/'server_v92.log'
env=os.environ.copy()
env['ANGSI_DATA_ROOT']='/data/epic20/APGCE/input_data/challenge3'
with LOG.open('a') as handle:
    PROC=subprocess.Popen([sys.executable,'-m','streamlit','run',str(APP),
        '--server.address=127.0.0.1','--server.port=8515',
        '--server.headless=true','--server.fileWatcherType=none'],
        cwd=str(APP.parent),env=env,stdout=handle,
        stderr=subprocess.STDOUT,start_new_session=True)
print('Streamlit PID:',PROC.pid,'Log:',LOG)
```

5. Verify backend in the same notebook:

```python
import urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8515/_stcore/health',timeout=5) as response:
        print('Backend:',response.status,response.read().decode())
except Exception as exc:
    print(type(exc).__name__,exc)
    print(LOG.read_text(errors='replace')[-5000:])
```

6. On the **same remote HPC desktop** in RustDesk/Firefox, open `http://127.0.0.1:8515`.
   If Firefox is on a different computer, localhost resolves there, not on the HPC server. Try your actual JupyterHub origin plus `/user/apgce_team_13/proxy/8515/` **only if** JupyterServerProxy is enabled. A 200 health response from the notebook does *not* prove that browser routing works.

7. If a blank page or exception occurs, inspect `server_v92.log`, copy the final traceback, and do not execute the Streamlit script directly via `%run` in a notebook.

## Tests and scientific scope

- Source syntax and static regression tests run in the packaging environment.
- Actual HPC SEG-Y processing and Streamlit render **not tested outside the user's HPC**. This local environment cannot install Streamlit, segyio and lasio from the network.
- Tests are synthetic and do not prove horizon picks, reservoir classes, well ties or opportunity rankings on Angsi data.
- 3D boreholes are *visual vertical assumptions* until actual directional survey coordinates are validated.
- Survey-wide in this app means entire **sampled paired seismic trace footprint** (5/10/15 IL/XL), not every trace unless full-resolution processing is separately implemented.
