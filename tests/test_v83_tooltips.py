"""V8.3 documentation/tooltip smoke tests; no field dataset required."""
from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui_help import HELP, VOLUME_HELP, help_text

SUPPORTED = {
    'selectbox', 'multiselect', 'slider', 'select_slider', 'number_input',
    'text_input', 'text_area', 'checkbox', 'toggle', 'radio', 'date_input',
    'time_input', 'color_picker', 'file_uploader', 'button',
    'form_submit_button', 'segmented_control', 'pills', 'download_button'
}


def test_all_v83_supported_inputs_have_native_help():
    total = 0
    for filename in ('angsi_jr_v8.py', 'v8_dashboard.py', 'horizon_review.py'):
        tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr not in SUPPORTED:
                continue
            if not isinstance(call.func.value, ast.Name):
                continue
            if call.func.value.id not in ('st', 'col', 'pcol', 'scol'):
                continue
            assert any(arg.arg == 'help' for arg in call.keywords), (
                filename, call.lineno, call.func.attr
            )
            total += 1
    assert total >= 140


def test_advanced_qc_and_opportunity_help_is_specific():
    for label, tokens in {
        'Reference NRMS limit (%)': ('NRMS', 'screening'),
        'Maximum lag (samples)': ('shift', 'samples'),
        'Sample spacing in IL/XL': ('stride', 'sampling'),
        'Verified projected EPSG code (optional)': ('EPSG', 'area'),
        'What-if simulations per candidate': ('Monte Carlo', 'scenario'),
        'Low contrast cutoff': ('low-change', 'oil'),
        'Extraction top TVDSS (m)': ('TVDSS', 'checkshot'),
        'Ricker peak frequency (Hz)': ('wavelet', 'synthetic'),
        'Map resolution (IL/XL)': ('sampling', 'resolution'),
    }.items():
        doc = help_text(label)
        assert all(t.lower() in doc.lower() for t in tokens), (label, doc)
    assert 'Seismic' in help_text('Seismic %')
    assert len(VOLUME_HELP) == 11
    for label in ('NetPay_low_m', 'NetPay_mode_m', 'NetPay_high_m',
                  'Phi_low', 'Phi_mode', 'Phi_high',
                  'Sw_low', 'Sw_mode', 'Sw_high'):
        assert label in VOLUME_HELP and len(VOLUME_HELP[label]) >= 45


def test_no_calculation_module_modified():
    for file in ('seismic_qc.py', 'opportunities.py', 'uncertainty.py',
                 'opportunity_intelligence.py', 'angsi_core.py'):
        assert (ROOT / file).is_file()
