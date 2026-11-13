# ANGSI JR. V10.2.1 — Working organiser AI chat replacement

## Scope

**Baseline:** ANGSI JR. V10.2 — `ANGSI_JR_V10_2_AUTO_VOLUMETRICS`.
**Change:** Replace only the right-hand `copilot_panel()` with an OpenAI-compatible **organiser Gemma chat**, implemented in `ai_chat.py`. All seismic QC, horizon, well correlation, analysis, opportunities and V10.2 volumetrics modules are preserved unchanged.

The user-supplied pasted text described the collaborator's working `ai_chat.py` and zip, **but did not include the actual `ai_chat.py` source**. Accordingly this is a new implementation matching its documented behaviour, **not a byte-for-byte replacement** of the collaborator's original script.

### Chat features

- Default endpoint: `http://172.28.34.1:8000/v1`
- Model ID: `google/gemma-4-26B-A4B-it`
- Python `openai.OpenAI` client, `token-not-needed` default placeholder, configurable through environment variables.
- Last 20 messages; strictly alternating Gemma `user/assistant` roles. Instructions and optional dashboard context are prepended to the **first user message** instead of creating a second user role or unsupported `system` role.
- Streaming responses with non-streaming fallback if the endpoint lacks SSE support.
- A real **Test connection** button (including a one-line inference check), **Clear chat**, full message history in session.
- General questions like “hi” or “What is 4D seismic?” can be sent without selecting records. No scripted rule-based answer impersonates the model.
- Optional **Include derived dashboard evidence in AI chat** checkbox, **OFF by default**: names of wells, selected IL/XL, top/checkshot summaries, bounded QC, candidates, and calculated V10.2 volume totals. This shares only summaries — no raw SEG-Y or LAS arrays.

**Data/security:** The organiser IP is on a private network, but requests are plain HTTP and unencrypted. Only use it where your team and project policies authorise sending messages and derived data. The optional evidence checkbox grants additional context sharing, not full-volume upload. Chat history stays in the current Streamlit session; the model server may have its own logging/retention.

## JupyterLab install — each cell in order

### 1. Upload

Upload `ANGSI_JR_V10_2_1_WORKING_AI_CHAT.zip` to:

`/data/epic20/APGCE/apgce_team_13/`

### 2. Extract (Python notebook cell)

```python
from pathlib import Path
import zipfile
ROOT = Path('/data/epic20/APGCE/apgce_team_13')
archive = ROOT / 'ANGSI_JR_V10_2_1_WORKING_AI_CHAT.zip'
assert archive.exists(), f'Missing uploaded ZIP: {archive}'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, 'Corrupt ZIP'
    z.extractall(ROOT)
APP_DIR = ROOT / 'ANGSI_JR_V10_2_1_WORKING_AI_CHAT'
APP = APP_DIR / 'angsi_jr_v10.py'
print('App:', APP.exists(), APP)
print('Live AI module:', (APP_DIR/'ai_chat.py').exists())
print('Volumetrics module:', (APP_DIR/'v10_2_reservoir_engine.py').exists())
```

### 3. Install Python requirements (new notebook cell)

```python
%pip install -r /data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_1_WORKING_AI_CHAT/requirements.txt
```

Restart kernel if Jupyter requests it, then run the next cell. This install includes `openai` and `httpx`.

### 4. Start Streamlit on separate port 8520 (Python notebook cell)

```python
import os, sys, subprocess, py_compile
from pathlib import Path
APP_DIR = Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_1_WORKING_AI_CHAT')
APP = APP_DIR / 'angsi_jr_v10.py'
LOG = APP_DIR / 'server_v1021.log'
PORT = 8520
assert APP.is_file(), f'App not found: {APP}'
py_compile.compile(str(APP), doraise=True)
env = os.environ.copy()
env['ANGSI_DATA_ROOT'] = '/data/epic20/APGCE/input_data/challenge3'
env['LLM_BASE_URL'] = 'http://172.28.34.1:8000/v1'
env['LLM_MODEL'] = 'google/gemma-4-26B-A4B-it'
env['LLM_API_KEY'] = 'token-not-needed'
env['NO_PROXY'] = '127.0.0.1,localhost,172.28.34.1'
with LOG.open('a') as out:
    process_v1021 = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', str(APP),
         '--server.address=127.0.0.1', f'--server.port={PORT}',
         '--server.headless=true', '--server.fileWatcherType=none'],
        cwd=str(APP_DIR), env=env, stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True)
print('Process:', process_v1021.pid, 'LOG:', LOG)
```

### 5. Health check (Python notebook cell)

```python
import time, urllib.request, socket
print('Jupyter kernel host:', socket.gethostname())
for i in range(12):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8520/_stcore/health', timeout=4) as r:
            print('HTTP', r.status, 'response:', r.read().decode())
        break
    except Exception as exc:
        if i == 11:
            print('Health failed:', exc)
            print(LOG.read_text(errors='replace')[-7000:] if LOG.exists() else 'No log')
        else:
            time.sleep(1)
```

`HTTP 200 / ok` confirms Streamlit is running on the **kernel host**, not necessarily that Firefox can route to it.

### 6. Open website

- If Firefox runs **on the same remote HPC host**: `http://127.0.0.1:8520/`
- If browser runs through a **JupyterHub server with jupyter-server-proxy enabled**: `https://<your actual JupyterHub origin>/user/apgce_team_13/proxy/8520/`.
- If the proxy route returns 404, check the JupyterHub proxy extension with the HPC administrator; don't paste literal `<your actual JupyterHub origin>` in the browser.

You may create a clickable link from your Jupyter notebook:

```python
from IPython.display import display, HTML
from urllib.parse import quote
prefix = '/user/apgce_team_13/'
display(HTML(f'<a target="_blank" href="{prefix}proxy/8520/">OPEN ANGSI JR V10.2.1</a>'))
```

### 7. Test the AI

1. Open the dashboard. On the right, click **Test connection** (this sends `Say CONNECTION_OK` but **no project data**).
2. Type `Hi`. A real response should stream in.
3. Type `What is 4D seismic monitoring?`.
4. If authorised, enable **Include derived dashboard evidence in AI chat** after running the desired QC/volumes. Then ask `What reservoir interval and opportunity totals have actually been calculated?`.
5. If it says no current evidence, run the appropriate calculation first and ask again.

### 8. Diagnostics if the app is healthy but AI isn't

In Jupyter:

```python
import os
from pathlib import Path
APP_DIR = Path('/data/epic20/APGCE/apgce_team_13/ANGSI_JR_V10_2_1_WORKING_AI_CHAT')
print((APP_DIR/'server_v1021.log').read_text(errors='replace')[-7000:])

# Organiser server connectivity, from the SAME Python kernel; no data sent:
from urllib.request import urlopen
try:
    print('Model endpoint HTTP:', urlopen('http://172.28.34.1:8000/v1/models', timeout=5).status)
except Exception as exc:
    print(type(exc).__name__, str(exc)[:240])
```

A `/models` error does not necessarily mean inference is unavailable. Use **Test connection** for the actual one-message inference check. If the main app works but chat has `ModuleNotFoundError: openai`, rerun the requirements installation in the same Python environment used to launch Streamlit.

## Verification limits

Static Python compile and local mock-client tests have been completed. This environment cannot reach the organiser's private Gemma server or the Angsi SEG-Y files on your HPC. The first genuine connection and end-to-end dashboard test must occur from your team's Jupyter/HPC environment.
