# ANGSI JR V9.3 — GeoCopilot scientific-brain integration

This is a full standalone copy of V9.2. The non-Copilot pages and datasets are unchanged.

## Additions
- `geoscience_brain.py`: teammate's scientific knowledge, 4D interpretation rules, guardrails, recommendation patterns, and context-driven questions.
- Integrated system instructions with existing endpoint and explicit consent gate.
- Suggested questions in GeoCopilot; buttons submit questions only after a click.
- Improved safe offline explanations for 4D, NRMS, faults, amplitude, saturation and potential unswept reservoir.
- No invented Angsi field values, well outcomes, reserves or literature citations.
- `tests/test_geocopilot_brain.py`: targeted smoke checks.

## Launch in JupyterLab / remote HPC
1. Upload ZIP to `/data/epic20/APGCE/apgce_team_13/` using JupyterLab Upload.
2. Open a JupyterLab terminal and run:

```bash
cd /data/epic20/APGCE/apgce_team_13
unzip -o ANGSI_JR_V9_3_GEOCOPILOT_FULL.zip
cd ANGSI_JR_V9_3_GEOCOPILOT
python -m pip install -r requirements.txt
python -m pytest -q tests/test_geocopilot_brain.py
```

3. Open `JUPYTER_START_V9_3.ipynb` inside the unpacked folder and run all cells. Alternatively start using:

```bash
export ANGSI_DATA_ROOT=/data/epic20/APGCE/input_data/challenge3
python -m streamlit run angsi_jr_v9.py --server.address 127.0.0.1 --server.port 8515 --server.headless true --server.fileWatcherType none 2>&1 | tee server_v93.log
```

4. Check logs and health:

```bash
curl -I http://127.0.0.1:8515/_stcore/health
tail -n 50 server_v93.log
```

5. In JupyterLab use the existing remote port proxy for port 8515 (often `/proxy/8515/` from the authenticated JupyterLab URL), or the port forwarding URL supplied by your HPC. Do not assume plain `localhost:8515` on your laptop will reach the HPC.
6. In GeoCopilot, select read-only evidence. Suggested questions work in offline science-help mode. For model-based reasoning, expand Model connection, supply an **approved** OpenAI-compatible endpoint and model ID, enter the session key, and explicitly authorize sending derived records. Without that permission nothing is sent to an external model.

Note: this upgrade adds a scientific reasoning *prompt*, not a locally hosted trained AI model. No field-specific data are automatically sent or analyzed without explicit evidence selection and an authorized model endpoint.
