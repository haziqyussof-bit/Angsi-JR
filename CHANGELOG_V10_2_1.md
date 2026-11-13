# V10.2.1 — Live organiser GeoCopilot replacement

- Based on **V10.2**; no change to volumetrics, seismic, horizon or opportunity computation modules.
- Replaces only `copilot_panel(data)` in `angsi_jr_v10.py` with `ai_chat.render_chat_panel(data, st.session_state)`.
- New `ai_chat.py`: OpenAI-compatible model endpoint, streaming, conversation memory, Gemma role alternation, bounded optional derived context, `Test connection`, `Clear chat`.
- Default organiser server and model; override with `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_TRUST_ENV`.
- Previous local rule-based copilot module and its knowledge library remain in package for compatibility but **are not invoked by the new chat panel**.
- Updated `requirements.txt` to include `openai` and `httpx`.
- Added `tests/test_v1021_live_ai_chat.py`, `tests/test_v1021_http_integration.py`, new Jupyter launch notebook and README.
- Unlike the earlier rule-based fallback, failed live inference shows a genuine connection error, not an invented answer.
- The uploaded pasted transcript described collaborator code but did not contain the original `ai_chat.py` bytes; V10.2.1 is a compatible fresh implementation, not an exact file transplant.
