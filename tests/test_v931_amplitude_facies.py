"""Scientific regression tests for V9.3.1 Workflow 2, no SEG-Y/network required."""
import numpy as np
import pandas as pd
import analysis_engine as A
from shapely.geometry import Polygon


def sample_frame():
    rows=[]
    for il in (100,110,120):
        for xl in (200,210,220):
            # Strongest cells far from wells; no spatial cropping.
            v=float((il-100)//10*3+(xl-200)//10+1)
            rows.append(dict(IL=il, XL=xl, B_RMS=v, B_EnvelopeMean=v*1.25,
                             B_AbsolutePeak=v*2, M_RMS=v*.9, D_RMS=v*.1,
                             Local_contrast=.3, Reference_NRMS=15.,
                             QC_screening_pass=True,
                             Horizon_support="NOT_HORIZON_MODE",
                             X=450000+il,Y=570000+xl, Window_RMS_span=.1))
    return pd.DataFrame(rows)


def test_full_survey_without_well_training():
    df=sample_frame()
    out, report=A.amplitude_facies_proxy(df,low_percentile=30,high_percentile=70)
    assert report['status']=='EXPLORATORY_UNVALIDATED'
    assert report['no_training_wells_required'] is True
    assert len(out)==len(df)==9
    assert (out['AmplitudeFaciesClass']=='BRIGHT_SAND_PRONE').any()
    assert (out['AmplitudeFaciesClass']=='DIM_SHALE_PRONE').any()
    assert int(out['AmplitudeFaciesClass'].value_counts().sum())==9
    # Never classify oil or gas; only hypothesis labels.
    assert not out['AmplitudeFaciesClass'].str.contains('OIL|GAS').any()
    assert np.allclose(df['B_RMS'],out['B_RMS'])


def test_amplitude_strength_uses_1995_only_not_4d_leakage():
    df=sample_frame()
    a,_=A.amplitude_facies_proxy(df)
    df['D_RMS']*=10000
    df['M_RMS']=0
    b,_=A.amplitude_facies_proxy(df)
    assert a['AmplitudeFaciesClass'].equals(b['AmplitudeFaciesClass'])


def test_missing_amplitude_and_reference_fail_keep_data_and_qc_labels():
    df=sample_frame()
    df.loc[0,'B_RMS']=np.nan
    df.loc[1,'QC_screening_pass']=False
    df.loc[2,'Horizon_support']='MODEL_ONLY'
    o,r=A.amplitude_facies_proxy(df)
    assert len(o)==9
    assert o.loc[0,'AmplitudeFaciesClass']=='UNCLASSIFIED'
    assert np.isnan(o.loc[0,'AmplitudeFaciesCode'])
    assert o.loc[1,'AmplitudeFaciesQC']=='RAW_ONLY_REFERENCE_QC_FAILED'
    assert o.loc[2,'AmplitudeFaciesQC']=='MODEL_ONLY_HORIZON_UNVALIDATED'
    assert r['excluded_amplitude_cells']==1


def test_constant_seismic_never_invents_sand_class():
    df=sample_frame()
    df['B_RMS']=12
    o,r=A.amplitude_facies_proxy(df)
    assert r['status']=='INSUFFICIENT'
    assert not (o.AmplitudeFaciesClass=='BRIGHT_SAND_PRONE').any()
    assert not (o.AmplitudeFaciesClass=='DIM_SHALE_PRONE').any()


def test_invalid_percentiles_and_nonbaseline_rejected():
    df=sample_frame()
    for args in [dict(low_percentile=70,high_percentile=30),dict(attribute='D_RMS')]:
        try:
            A.amplitude_facies_proxy(df,**args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid proxy parameters accepted')


def test_candidate_regions_can_filter_unvalidated_bright_proxy_without_well_radius():
    df=sample_frame()
    o,_=A.amplitude_facies_proxy(df)
    raw=A.candidate_regions(o,10,metric='Raw_RMS_percentile',low=.30,high=.70,min_cells=1)
    filtered=A.candidate_regions(o,10,metric='Raw_RMS_percentile',low=.30,high=.70,
                                 min_cells=1,facies_filter='BRIGHT_SAND_PRONE')
    assert raw
    assert filtered
    allowed=set(map(tuple,o.loc[o.AmplitudeFaciesClass=='BRIGHT_SAND_PRONE',['IL','XL']].to_numpy()))
    assert all(set(r['Keys']) <= allowed for r in filtered)
    assert sum(r['Cells'] for r in filtered) <= sum(r['Cells'] for r in raw)
    assert all(r['Facies_evidence_status']=='UNVALIDATED amplitude hypothesis' for r in filtered)
    ranked=A.rank_candidates(filtered,o,wells=None,scenarios=10)
    assert 'Bright_amplitude_proxy_fraction' in ranked
    assert 'Facies_evidence_status' in ranked
    assert (ranked['Production_status']=='Not supplied').all()
