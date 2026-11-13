# ANGSI JR. V9.3.1 — Workflow 2 Amplitude Facies Proxy

**Built from the exact user-uploaded `ANGSI_JR_V9_3_GEOCOPILOT_FULL.zip`.** This package contains all V9.3 application code and assets; only Workflow 2 and its Opportunities connection are changed, plus version label and launcher.

## What is new

- **Analysis → Workflow 2 → Screening pathway** provides two independent choices:
  1. **Amplitude Facies Proxy — full seismic coverage / no training wells** (default): 1995 baseline RMS, P30 dim, P70 bright; also baseline envelope mean or absolute peak. Uses every target-valid paired sampled cell from Workflow 1, never near-well buffers. Color labels: green = BRIGHT_SAND_PRONE hypothesis, grey = DIM_SHALE_PRONE hypothesis, yellow = INTERMEDIATE_UNCERTAIN. No oil, gas, or confirmed sand/shale labels.
  2. **Well-calibrated reservoir screening**: unchanged conservative measured-well training + leave-one-well-out gate, can still return INSUFFICIENT.
- Strict no-contrast guard: flat amplitude grids cannot be labelled bright/dim.
- Categorical 2D map with well overlays; per-cell raw-only / reference QC and model-only horizon confidence flags, counts, downloadable CSV and manifest.
- **Opportunities** inherits amplitude proxy results for a categorical map background or optional hypothesized bright/dim overlay. Optional candidate polygon filter `No facies restriction` / `Bright hypothesis` / `Dim hypothesis`. Entire shared sampled seismic survey is still evaluated. Ranking table includes `Bright_amplitude_proxy_fraction` and explicitly `UNVALIDATED` evidence status, but proxy is not treated as a calibrated oil/reservoir probability.
- Switching back to Workflow 1 and recalculating resets stale Workflow 2/Opportunities products.

## Geological / quality restrictions

Bright=potential sand and dim=potential shale are *hypotheses*, not universally valid. Reflectivity also depends on acoustic impedance, polarity, tuning, saturation, gas, pressure, acquisition, phase, processing, and local stratigraphic context. `B_RMS` measures unsigned amplitude *strength*, not peak polarity. Only 1995 baseline strength establishes proxy bins; 4D difference does not label lithology. QC failures remain flagged; a large displayed footprint is not proof of reservoir. Whole cube means **every matched cell on your chosen IL/XL sampling grid**, not every physical trace unless spacing=1. Horizon mode still requires top/base predictions per sampled location and distinguishes MODEL_ONLY from HIGH_CONFIDENCE. Not a validated geological map; don't infer remaining oil volumes.

## Install and launch in HPC JupyterLab

1. Upload `ANGSI_JR_V9_3_1_FACIES_FULL.zip` into `/data/epic20/APGCE/apgce_team_13/`.
2. New notebook cell:

```python
from pathlib import Path
import zipfile
root=Path('/data/epic20/APGCE/apgce_team_13')
with zipfile.ZipFile(root/'ANGSI_JR_V9_3_1_FACIES_FULL.zip') as z:
    z.extractall(root)
app=root/'ANGSI_JR_V9_3_1_FACIES'/'angsi_jr_v9.py'
print('APP FOUND',app.is_file(),app)
```

3. Open `ANGSI_JR_V9_3_1_FACIES/JUPYTER_START_V9_3_1.ipynb` and execute the cells. The notebook has dependency installation, syntax preflight, launch on port **8516**, health and logs.
4. On the *same remote HPC desktop* open `http://127.0.0.1:8516`. If the browser is elsewhere use IT-authorised port forwarding or JupyterHub server proxy (e.g. the authenticated `/user/apgce_team_13/proxy/8516/` route, if enabled). A health response on Jupyter kernel does not guarantee browser access.
5. **Analysis** → Workflow 1: choose `Time window (TWT)`, 1500–1700 ms, grid 10 IL/XL; run **Calculate seismic attributes**. Workflow 2: default **Amplitude Facies Proxy**, P30/P70, press **Calculate amplitude facies proxy**. See complete map and QC. Then **Opportunities**, choose optional facies map/background and build candidate polygons. For horizon extraction instead, run Workflow 1 `Interpreted horizon pair`, noting MODEL_ONLY caveats.
6. If the site fails: check `server_v931.log` in this folder; if `ModuleNotFoundError`, use the notebook `%pip install -r .../requirements.txt` cell. For terminal usage, run `python -m streamlit run angsi_jr_v9.py --server.address 127.0.0.1 --server.port 8516 --server.headless true` in the app folder.

## Verification completed here

- New tests: `tests/test_v931_amplitude_facies.py` — 6 passed. Existing tests `tests/test_v9_analysis.py` — 6 passed. **12 targeted tests total**.
- Python compilation and ZIP CRC integrity passed.
- Full legacy test suite and actual HPC SEG-Y run not performed: `segyio` and `streamlit` are unavailable in this local verification environment. Run the full test suite in HPC after dependency installation.
- No confidential well/SEG-Y data included. Files are read-only analyses of the user's data root when launched.
