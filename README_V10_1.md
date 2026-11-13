# ANGSI JR. V10.1 — Automatic Preliminary In-Place Volumes

Baseline: V10 Redownload ZIP. This is a **full source package**, not an incremental replace-only patch.

## New functionality

- **Opportunities** automatically estimates all ranked candidate *screening-scenario* volumes **when you click `Build ranked opportunity screening polygons`**. Press `RECALCULATE ALL CANDIDATE VOLUMES` only after editing the advanced PVT/uncertainty assumptions.
- Candidate 4D footprint area is derived from the **actual IL/XL to XY affine fit of the 1995 SEG-Y header**. This is not closure area. For physically inconsistent XY, the volume is `NOT QUANTIFIABLE`.
- Well-local **TVDSS** intervals use the actual paired well tops in horizon mode or the **own checkshot** in fixed-time mode; no checkshot is invented or extrapolated. No own checkshot => well is excluded from this automatic analog pool unless valid marker constraints cover horizon mode.
- LAS data supply interval-specific **PHIE**, **SWE** (Sw), **VSHALE** or explicitly *unvalidated* GR-derived shale proxy, model-derived **VOIL / VGAS** phase proportions if available, and **NTG** from a documented PHIE >= 0.10, shale <= 0.40, Sw <= 0.70 cutoff. The controls are screening hypotheses, not calibrated petrophysical cutoffs.
- When trace-level top/base horizons exist, the candidate gross thickness uses the median candidate TWT picks converted with the nearest well's **own** checkshot; if impossible, falls back to well interval analog gross with an explicit `Thickness_source` caveat.
- Report gross rock volume, HCPV, STOIIP (MMstb), GIIP (Bscf), in-place BOE (MMboe), and P90/P50/P10 **conditional sensitivity values** from +/- assumed perturbation.
- If VOIL/VGAS is missing or unusable, report **separate oil-only STOIIP and gas-only GIIP end members**, with **no combined BOE figure**. These alternatives must not be added together.
- If a candidate is outside local well support, output explicitly states `DISTANT_FIELD_ANALOG_EXTRAPOLATION`. Ranking is independent of the automatically calculated volume and remains untouched.
- PVT **Bo=1.25** (res m³ per stock-tank m³) and **Bg=0.005** (res m³ per standard m³) are **ASSUMED EXAMPLE VALUES** because the project does not provide PVT. Gas equivalency defaults to 6,000 scf per BOE. All settings are visible/editable under Advanced.
- WGS84/EPSG:32648 is the **assumed geographic CRS** (public Angsi locality approx. 5.1957°N / 104.7093°E), and calculated spatial fit and survey median are shown for plausibility QC. *The authoritative survey CRS has not been verified*. Area is derived from XY in metres only when the affine residual is acceptable.

## Scientific warning

These are **illustrative hydrocarbon in-place screening scenarios**, NOT proven STOIIP, GIIP, 2P reserves, remaining oil or volumes you can book. 4D differences may represent noise, pressure/saturation change or acquisition; anomaly polygons **are not** mapped structural closures or demonstrated connected reservoir; hydrocarbon contacts, recovery factor, PVT and production/injection histories are not supplied. Never sum candidates into a field resource figure. Use actual geological limits, production and calibrated rock physics for reliable volumes.

## Full JupyterLab installation — remote HPC

Upload `ANGSI_JR_V10_1_AUTO_VOLUMETRICS.zip` into `/data/epic20/APGCE/apgce_team_13/`.

**Cell 1: Extract**
```python
from pathlib import Path
import zipfile
ROOT=Path('/data/epic20/APGCE/apgce_team_13')
ZIP=ROOT/'ANGSI_JR_V10_1_AUTO_VOLUMETRICS.zip'
assert ZIP.exists(), f'Upload file first: {ZIP}'
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None, 'Corrupt ZIP'
    z.extractall(ROOT)
APP_DIR=ROOT/'ANGSI_JR_V10_1_AUTO_VOLUMETRICS'
APP=APP_DIR/'angsi_jr_v10.py'
print('APP FOUND:',APP.exists(),'PATH:',APP)
```

**Cell 2: Dependencies**
```python
%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_1_AUTO_VOLUMETRICS/requirements.txt
```

**Cell 3: Start on PORT 8518 (V10 on 8517 remains untouched)**
```python
from pathlib import Path
import os,sys,subprocess,py_compile,urllib.request
APP_DIR=Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_1_AUTO_VOLUMETRICS')
APP=APP_DIR/'angsi_jr_v10.py'
py_compile.compile(str(APP),doraise=True)
LOG=APP_DIR/'server_v101.log'
ENV=os.environ.copy()
ENV['ANGSI_DATA_ROOT']='/data/epic20/APGCE/input_data/challenge3'
def healthy():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8518/_stcore/health',timeout=3) as r:
            return r.status==200
    except Exception:return False
if healthy():
    print('Port 8518 already has a healthy server. Verify its version in the website.')
else:
    with LOG.open('a') as f:
        proc=subprocess.Popen([sys.executable,'-m','streamlit','run',str(APP),
          '--server.address=127.0.0.1','--server.port=8518',
          '--server.headless=true','--server.fileWatcherType=none'],
          env=ENV,cwd=APP_DIR,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print('Launched PID:',proc.pid)
print('Log:',LOG)
```

**Cell 4: Health check and errors**
```python
import time,urllib.request
for k in range(10):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8518/_stcore/health',timeout=4) as r:
            print('HTTP',r.status,r.read().decode());break
    except Exception as err:
        if k==9:
            print('FAILED:',err)
            print(LOG.read_text(errors='replace')[-7000:] if LOG.exists() else 'No log')
        else:time.sleep(1)
```

**Cell 5: Website**

Open **http://127.0.0.1:8518/** in Firefox on the **same remote desktop host**. If notebook kernel and Firefox run on different HPC nodes, `localhost` won't route; use a configured JupyterHub proxy `/user/apgce_team_13/proxy/8518/` or your IT-approved route. HTTP health 200 is not a browser access guarantee.

**Workflow:** Analysis Workflow 1 → Extract 1500-1700 ms or horizon I-25 to I-68 → Opportunities → Build ranked polygons. Auto volume runs with that click. Open the `Inspect real LAS interval controls` expander for exclusions. If a candidate reports no data, it is not quantitatively estimated — do not fill with synthetic petrophysical measurements.

## Verification

Python syntax checks pass. Targeted synthetic tests for volumes, area/CRS plausibility, missing phase, and V10 existing science pass. Real Angsi SEG-Y and LAS can only be tested in your HPC environment; use logs if the app fails at runtime.
