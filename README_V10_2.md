# ANGSI JR. V10.2 — Multi-Reservoir Volume Scenarios & Evidence-Based GeoCopilot

**Baseline:** Full V10.1 app, independently deployed on **port 8519**. This is a complete package; it does not modify running V10.1 on port 8518.

## What's added

- **Opportunities — all candidate areas:** One click calculates facies-weighted STOIIP (MMstb) or GIIP (MMscf) screening scenarios for **all** candidate polygons in the selected analysis interval (provided a calibrated SEG-Y XY area and valid paired samples exist). Bubbles are green, labelled with candidate ID and numeric volume, and sized by calculated scenario volume; ranked cards sorted by volume, not 4D score.
- **Reservoir parameter logic:** `v10_auto_volumetrics.py` uses available interval-relevant LAS and actual marker/checkshot controls; V10.2 uses the **same median gross thickness, NTG, PHIE and Sw across all candidate polygons in each reservoir by default**, before applying **shared editable** yellow-bright/green-intermediate/grey-dim facies multipliers. An optional spatial-analog mode retains regional variation. Missing parameters get **clearly identified default assumptions**, not invented measured values. Gross-thickness cap is off by default, so measured intervals are not secretly truncated.
- **Batch multiple marker intervals:** Run all eligible **adjacent** interpreted-marker pairs or choose a subset, using whole paired SEG-Y spatial sampling, candidate segmentation and well-controlled assumptions. **Marker intervals do not automatically equal hydrocarbon-bearing reservoirs.** Some pairs may not produce eligible candidates. Session stores per-interval results. Display total volumes by stratigraphic interval; show an illustrative field sum **only for measured non-overlapping intervals** and consistent phase assumptions. Summed polygons are not mapped field reserves.
- **Volumes are NOT reserves:** GIIP, STOIIP and BOE are initial **in-place scenarios** within selected 4D anomaly footprints (not full structural closures). Display assumed gas/oil recoverable **scenario** separately. Remaining recoverable is not calculated without explicit verified cumulative gas (MMscf) or oil (MMstb) input; even then, pro-rata allocation over candidate polygons is hypothetical, not a production allocation.
- **Check consistent units:** gas shown as **MMscf** (million standard cubic feet), oil as **MMstb** (million stock-tank barrels), gas BoE uses explicitly assumed scf/BOE. `Bo=1.25`, `Bg=0.005`, oil recovery 25%, gas recovery 50% are **illustrative defaults**, NOT validated PVT/recovery factors. No 25 MMscf target is tuned into the results.
- **Seismic QC / E-W fence:** GR and resistivity line colours can be customized; E-W fence extension slider is now **beneath** the seismic plot.
- **GeoCopilot:** Curated local answer library for 4D, well ties, facies and volumetrics remains available without external endpoint. Public metadata search powered by Crossref is **opt-in and requires HPC internet**. It never sends SEG-Y/LAS; it does not claim to ingest full papers. GeoCopilot history is displayed within the page (no separate fixed-height scroll). An approved organiser endpoint can still be used when available.

## Installation in HPC JupyterLab — every step

### 1. Upload

Upload **`ANGSI_JR_V10_2_FULL.zip`** into
`/data/epic20/APGCE/apgce_team_13/` via JupyterLab's upload icon.

### 2. Open a Jupyter Python notebook and extract (CELL 1)

```python
from pathlib import Path
import zipfile
ROOT=Path('/data/epic20/APGCE/apgce_team_13')
ZIP=ROOT/'ANGSI_JR_V10_2_FULL.zip'
assert ZIP.is_file(), f'Upload ZIP here: {ZIP}'
with zipfile.ZipFile(ZIP) as z:
    assert z.testzip() is None, 'ZIP CRC check failed'
    z.extractall(ROOT)
APP_DIR=ROOT/'ANGSI_JR_V10_2_AUTO_VOLUMETRICS'
APP=APP_DIR/'angsi_jr_v10.py'
print('APP FOUND:',APP.is_file(),'APP:',APP)
print('New components:',(APP_DIR/'v10_2_reservoir_engine.py').is_file(),
      (APP_DIR/'v10_2_copilot.py').is_file())
assert APP.is_file(), 'Unexpected ZIP layout; print z.namelist() and inspect'
```

### 3. Install dependencies in the SAME Jupyter kernel (CELL 2)

```python
%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_AUTO_VOLUMETRICS/requirements.txt
```

Restart the notebook kernel if the package manager requests it, then rerun Cell 1 to restore Python variables.

### 4. Start Streamlit via Python executable (CELL 3)

```python
import sys, os, subprocess, py_compile, socket
from pathlib import Path
APP_DIR=Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_AUTO_VOLUMETRICS')
APP=APP_DIR/'angsi_jr_v10.py'
PORT=8519
LOG=APP_DIR/'server_v102.log'
assert APP.is_file(),f'Missing app: {APP}'
py_compile.compile(str(APP),doraise=True)
env=os.environ.copy()
env['ANGSI_DATA_ROOT']='/data/epic20/APGCE/input_data/challenge3'
with LOG.open('a') as output:
    process_v102=subprocess.Popen(
        [sys.executable,'-m','streamlit','run',str(APP),
         '--server.address=127.0.0.1',f'--server.port={PORT}',
         '--server.headless=true','--server.fileWatcherType=none'],
        cwd=str(APP_DIR),env=env,stdout=output,stderr=subprocess.STDOUT,
        start_new_session=True)
print('Process:',process_v102.pid,'on Jupyter kernel host:',socket.gethostname())
print('Server log:',LOG)
```

Never run the Streamlit app with `%run app.py`, `exec`, or directly as a Python notebook script; this causes missing ScriptRunContext.

### 5. Check Streamlit health (CELL 4)

```python
import urllib.request, time
from pathlib import Path
LOG=Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_AUTO_VOLUMETRICS/server_v102.log')
health='http://127.0.0.1:8519/_stcore/health'
healthy=False
for attempt in range(12):
    try:
        with urllib.request.urlopen(health,timeout=4) as response:
            print('HTTP STATUS:',response.status,'RESPONSE:',response.read().decode())
            healthy=(response.status==200)
            break
    except Exception as exc:
        if attempt==11:print('Health check failed:',repr(exc))
        else:time.sleep(1)
if not healthy:
    print('\n=== LAST 8000 CHARACTERS OF SERVER LOG ===')
    print(LOG.read_text(errors='replace')[-8000:] if LOG.is_file() else 'No log file')
```

**HTTP 200 / ok** confirms the app is listening on the **Jupyter kernel host**. It does not prove that a different desktop browser can connect.

### 6. Open web dashboard

- In **Firefox running on the SAME remote host as the Streamlit process**, open `http://127.0.0.1:8519/`.
- If your Firefox is running on a different host, localhost is the WRONG host; use the active JupyterHub base origin + `/user/apgce_team_13/proxy/8519/` **only if the proxy extension works**, or use HPC-supported SSH port forwarding.
- If JupyterHub says 404 for proxy URL, contact the HPC/Jupyter administrator about `jupyter-server-proxy` configuration; localhost health 200 does not imply proxy routing.

### 7. If browser won't connect (CELL 5)

```python
import socket, subprocess
from pathlib import Path
print('JUPYTER KERNEL HOST:',socket.gethostname())
p=subprocess.run(['ss','-ltn'],capture_output=True,text=True)
print('PORT 8519 LISTENER:')
print('\n'.join(x for x in p.stdout.splitlines() if ':8519' in x) or '(none)')
log=Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_AUTO_VOLUMETRICS/server_v102.log')
print('\n=== LOG ===')
print(log.read_text(errors='replace')[-8000:] if log.exists() else 'No log')
```

### 8. Exercise V10.2 features

1. **Analysis → Workflow 1:** run 1995/2006 time interval (e.g. 1500–1700 ms) with sample spacing 10, or a supported marker pair. Use full paired survey footprint; inspect valid/excluded-cell QC.
2. **Analysis → Workflow 2:** calculate amplitude-facies proxy; classes are **yellow bright**, **green intermediate**, **grey dim** and NOT validated sand/shale.
3. **Opportunities:** `Build ranked opportunity screening polygons`. Existing V10.1 auto-volume will attempt area/analogs automatically.
4. Under **V10.2 All Candidates & Multi-Reservoir Screening Volumetrics**, choose fluid scenario, uniform log medians, optional user cutoffs, and click **CALCULATE ALL CURRENT CANDIDATE VOLUMES & BUBBLE RANKING**. Review total current interval, Top 3 by VOLUME, bubble map, CSV, provenance and QC.
5. Optionally open **Batch run MULTIPLE interpreted reservoir intervals**. Select measured adjacent marker pairs and run across whole seismic footprint. If `MODEL_ONLY` dominates, horizon QC is needed before any prospectivity claim. Review interval totals and only enable an illustrative field sum after independent strata QC.
6. **Seismic QC:** choose GR/RT curve colours and adjust E-W extension beneath the seismic plot.
7. **GeoCopilot:** ask a general 4D/petrophysics question with or without organiser live endpoint. Optional Crossref lookup requires external internet, and public topic-only queries are sent.

**Never represent the output as discovered reserves, STOIIP or remaining resources without independent geologic closure, fluid contacts, PVT, production history and uncertainty calibration.**

## Verification status

Synthetic Python tests are in `tests/test_v102_reservoir_volumes.py`. Run via a Jupyter terminal:

```bash
cd /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_AUTO_VOLUMETRICS
python -m pytest -q tests/test_v102_reservoir_volumes.py
```

Local syntax checks and selected synthetic tests pass; **actual Angsi SEG-Y/LAS and browser interaction have not been tested on your HPC from this chat**.
