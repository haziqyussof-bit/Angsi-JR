# ANGSI JR. V9.1 — Whole Survey Analysis + Opportunities

**Base:** User's ANGSI JR. V8.3 → V9 Analysis. V9.1 changes only `analysis_engine.py`, `analysis_dashboard.py` and app version labels. All original Overview, Seismic QC, Horizon Interpretation, Well Correlation and GeoCopilot code is retained.

## What changed

- **Analysis** time/depth mode evaluates target-valid, baseline/monitor *paired* traces across their entire common seismic footprint at user-selected IL/XL spacing (5/10/15). It does **not** buffer near wells or restrict fixed-time maps to horizon control.
- **Opportunities** now has its own `Calculate FULL seismic coverage + 4D attributes` button, default TWT 1500–1700 ms and grid step 10. A user can run opportunities directly without first visiting Analysis.
- Preserve descriptive raw attributes where target seismic is valid but overburden/reference window fails quality criteria. Such cells are labelled `RAW_ONLY` and must not be presented as validated dynamic reservoir changes.
- **Two screening modes:** `Raw RMS percentile (whole target-valid survey)` detects relatively high/low 4D difference energy throughout the paired survey; `Reference-supported local 4D contrast (strict QC)` excludes cells failing reference checks. Neither mode proves produced/unswept hydrocarbons.
- Connected grid polygons and relative candidate ranking operate across **all eligible sampled seismic cells**, never limited by well-distance buffers. Wells are overlaid as geological/contextual control only. Candidate scores disclose unverified production and pressure context.
- QC metrics: paired, target-valid, reference-supported, RAW_ONLY, excluded; per-cell QC CSV/export in Analysis.

## Instructions on HPC JupyterLab

1. Upload `ANGSI_JR_V9_1_FULL_COVERAGE.zip` into `/data/epic20/APGCE/apgce_team_13/`.
2. Extract in a Jupyter notebook:

```python
from pathlib import Path
import zipfile
root=Path('/data/epic20/APGCE/apgce_team_13')
with zipfile.ZipFile(root/'ANGSI_JR_V9_1_FULL_COVERAGE.zip') as z:
    z.extractall(root)
app=root/'ANGSI_JR_V9_1_FULL_COVERAGE'/'angsi_jr_v9.py'
print(app, app.exists())
```
3. In the same notebook, `%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V9_1_FULL_COVERAGE/requirements.txt`.
4. Open `ANGSI_JR_V9_1_FULL_COVERAGE/JUPYTER_START_V9_1.ipynb` and run its launcher cells; or use the launch snippet below. The new port is **8514**.
5. Open `http://127.0.0.1:8514` **in a browser on the same remote HPC host**, not your personal laptop. Check JupyterHub proxy `/user/apgce_team_13/proxy/8514/` only if the server proxy is enabled.

### Direct launch cell
```python
import os,sys,subprocess
from pathlib import Path
app=Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V9_1_FULL_COVERAGE/angsi_jr_v9.py')
env=os.environ.copy()
env['ANGSI_DATA_ROOT']='/data/epic20/APGCE/input_data/challenge3'
with (app.parent/'server.log').open('a') as log:
    proc=subprocess.Popen([sys.executable,'-m','streamlit','run',str(app),
        '--server.address=127.0.0.1','--server.port=8514','--server.headless=true'],
        cwd=str(app.parent),stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
print('PID:',proc.pid)
```

## Scientific boundaries and limitations

- This is **full spatial-footprint sampling**, not every trace: e.g. stride 10 selects every tenth inline and crossline; choose 5 for more detail at higher compute cost. Baseline and monitor must have the same IL/XL key for a defensible difference. Unpaired footprint sections cannot have a measured 4D difference.
- Horizon-bounded maps legitimately remain restricted to locations with both valid interpreted reflector picks. To see whole footprint, select a fixed TWT window or use Opportunities' full survey button.
- `RAW_ONLY` means *data exist*, not that seismic time-lapse change passed reference repeatability QC. Raw RMS percentile is relative spatial anomaly rank only, not hydrocarbon probability.
- Well-conditioned facies classification still depends on sufficient measured well-log labels and leave-one-well-out validation; it will not fabricate full-survey oil/gas labels.
- Confirm SEG-Y geometry co-registration, acquisition mutes, CRS, amplitudes and processing before geological interpretation. Additional production/pressure/PVT data are required to infer unswept oil.

## Verification

- Syntax compilation: PASS.
- Nine focused synthetic tests: PASS (`tests/test_v9_analysis.py` and `tests/test_v91_full_coverage.py`).
- Full pre-existing tests and live SEG-Y end-to-end run: **NOT completed here** because `segyio`, `streamlit` and actual HPC SEG-Y data are not available in this runtime. Run on HPC and inspect `server.log` if startup fails.
