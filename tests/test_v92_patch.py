"""V9.2 synthetic scientific/regression checks (does not need real Angsi SEG-Y)."""
import numpy as np
import pandas as pd
from analysis_engine import descriptive_seismic_facies, rank_candidates
from petrophysics import interval_support


def test_optional_gr_proxy_not_silent_reservoir_truth():
    z=np.arange(1000.,1200.)
    a=pd.DataFrame({'DEPTH':z,'PHIE':np.linspace(.04,.26,len(z)),
                    'GR_P':np.linspace(18,128,len(z))})
    without=interval_support(a,0,1020,1180,allow_gr_proxy=False)
    assert np.isnan(without['quality'])
    result=interval_support(a,0,1020,1180,allow_gr_proxy=True)
    assert np.isfinite(result['quality'])
    assert result['shale_source']=='GR_quantile_index_UNCALIBRATED'
    assert 0<=result['quality']<=1


def test_measured_vshale_prioritized_over_gr_proxy():
    z=np.arange(1000.,1200.)
    a=pd.DataFrame({'DEPTH':z,'PHIE':np.full(200,.2),
                    'GR_P':np.full(200,135.),'VSHALE':np.full(200,.12)})
    result=interval_support(a,0,1010,1190,allow_gr_proxy=True)
    assert result['shale_source']=='provided_VSHALE'
    assert result['quality']>0.90


def test_full_coverage_faciese_unvalidated_and_not_geo_labels():
    il=np.repeat(np.arange(400,450,5),10)
    xl=np.tile(np.arange(2000,2050,5),10)
    frame=pd.DataFrame({'IL':il,'XL':xl,'B_RMS':np.arange(len(il))+1.,
                       'B_EnvelopeMean':np.arange(len(il))+1.,
                       'B_MostNegative':-np.arange(len(il))-1.})
    out=descriptive_seismic_facies(frame)
    assert len(out)==len(frame)
    assert out['Seismic_facies_cluster'].notna().sum()==len(frame)
    assert set(out['Seismic_facies_cluster'].unique())=={1.,2.,3.}
    assert 'Reservoir_similarity_score' not in out


def test_palette_oil_green_gas_red():
    from pathlib import Path
    s=(Path(__file__).resolve().parents[1]/'angsi_jr_v9.py').read_text()
    assert '"Oil-model":(36,159,101,255)' in s
    assert '"Gas-model":(239,100,73,255)' in s
    assert '"Water model":(178,187,192,255)' in s


def test_no_well_radius_in_surface_extraction():
    from pathlib import Path
    s=(Path(__file__).resolve().parents[1]/'analysis_dashboard.py').read_text()
    assert 'V8.surface_dict(a)' not in s
    assert 'surface["has_data"]' in s
    assert 'MODEL_ONLY' in s
