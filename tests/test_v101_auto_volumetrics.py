"""Synthetic only: never claims these values are Angsi measurements."""
import sys
import types
import numpy as np
import pandas as pd
from shapely.geometry import box

import v10_auto_volumetrics as A


def fixture_geometry():
    ii,xx=np.meshgrid(np.arange(800,810),np.arange(2500,2510))
    il=ii.ravel().astype(float); xl=xx.ravel().astype(float)
    g=pd.DataFrame({'IL':il,'XL':xl,'X':468000.+12.5*(il-805),
                    'Y':574000.+12.5*(xl-2505)})
    return g


def test_xy_fit_and_crs_plausibility():
    fit=A.survey_crs_qc(fixture_geometry())
    assert fit['metric_geometry_valid']
    assert abs(fit['area_bin_m2']-156.25)<1e-6
    assert fit['epsg32648_plausible']


def test_well_analog_uses_input_las_not_arbitrary_defaults(monkeypatch):
    n=1001
    depth=np.linspace(1400,1900,n)
    log=pd.DataFrame({'DEPT':depth+22,'PHIE':np.ones(n)*.20,
         'SWE':np.ones(n)*.30,'VSHALE':np.ones(n)*.20,
         'VOIL':np.ones(n)*.60,'VGAS':np.ones(n)*.40})
    cs=pd.DataFrame({'TVDSS_m':[1000,1500,2000],
         'TWT_ms':[1000,1500,2000]})
    import types,sys
    fake=types.ModuleType('angsi_core')
    fake.las_log=lambda wd,w:log
    fake.read_checkshot=lambda wd,w:cs
    monkeypatch.setitem(sys.modules,'angsi_core',fake)
    wells=pd.DataFrame({'Well':['Well-1'],'X':[468000.],'Y':[574000.],'RT':[22.]})
    analog=A.well_analog_records(wells,'ignored',pd.DataFrame(),
                                {'Mode':'Time window (TWT)','Top_ms':1500,'Base_ms':1700})
    row=analog.iloc[0]
    assert row.QC=='LOG_DERIVED'
    assert abs(row.NTG-1)<.01 and abs(row.PHIE-.2)<1e-6
    assert abs(row.Sw-.3)<1e-6
    assert abs(row.Oil_fraction-.6)<1e-6
    # Check area and in-place units, synthetic values only
    regions=[dict(ID='H01',Cells=4,Geometry=box(803,2502,807,2506))]
    vols,_,_=A.auto_candidate_volumes(regions,fixture_geometry(),wells,'ignored',pd.DataFrame(),
                     {'Stride':10,'Mode':'Time window (TWT)','Top_ms':1500,'Base_ms':1700})
    r=vols.iloc[0]
    assert r.Status.startswith('NEARBY_LOG_ANALOG')
    assert r.Area_km2>0 and r.STOIIP_MMstb>0 and r.GIIP_Bscf>0
    assert abs(r.Inplace_MMboe-(r.STOIIP_MMstb+r.GIIP_Bscf*1000/6000))<1e-9
    assert r.P90_MMboe<=r.P50_MMboe<=r.P10_MMboe


def test_unknown_phase_does_not_double_count(monkeypatch):
    n=1001
    depth=np.linspace(1400,1900,n)
    log=pd.DataFrame({'DEPT':depth+22,'PHIE':np.ones(n)*.20,
         'SWE':np.ones(n)*.30,'VSHALE':np.ones(n)*.20})
    cs=pd.DataFrame({'TVDSS_m':[1000,1500,2000],
         'TWT_ms':[1000,1500,2000]})
    fake=types.ModuleType('angsi_core')
    fake.las_log=lambda wd,w:log
    fake.read_checkshot=lambda wd,w:cs
    monkeypatch.setitem(sys.modules,'angsi_core',fake)
    wells=pd.DataFrame({'Well':['Well-1'],'X':[468000.],'Y':[574000.],'RT':[22.]})
    regions=[dict(ID='H01',Cells=4,Geometry=box(803,2502,807,2506))]
    vols,_,_=A.auto_candidate_volumes(regions,fixture_geometry(),wells,'ignored',pd.DataFrame(),
                     {'Stride':10,'Mode':'Time window (TWT)','Top_ms':1500,'Base_ms':1700})
    r=vols.iloc[0]
    assert np.isnan(r.Inplace_MMboe)
    assert r.Oil_only_STOIIP_MMstb>0 and r.Gas_only_GIIP_Bscf>0
    assert 'PHASE UNKNOWN' in r.Status
