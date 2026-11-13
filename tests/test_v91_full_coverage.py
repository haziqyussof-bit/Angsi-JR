"""Full common seismic coverage must NOT be clipped by wells/reference support."""
import sys
import types
import numpy as np
import pandas as pd
import analysis_engine as A


def test_reference_invalid_still_preserves_raw_target_maps(monkeypatch):
    # Reference (700-950 ms) is missing; target (1500-1700 ms) is valid.
    t = np.arange(0,2602,2,dtype=float)
    wave = np.sin(t/29.0)
    a = np.stack([wave,2*wave,3*wave], axis=1)
    m = 1.1*a
    a[t<1000,:] = np.nan
    m[t<1000,:] = np.nan
    class FakeVol:
        def __init__(self,z): self.samples=t; self.trace=[z[:,i] for i in range(z.shape[1])]
        def __enter__(self): return self
        def __exit__(self,*a): pass
    vols={'base':FakeVol(a),'mon':FakeVol(m)}
    monkeypatch.setitem(sys.modules,'segyio',types.SimpleNamespace(open=lambda p,*aa,**bb:vols[p]))
    pairs = pd.DataFrame([dict(IL=100+10*i,XL=200,bi=i,mi=i,X=460000+i*20,Y=570000) for i in range(3)])
    frame,qc = A.compute_attribute_volume('base','mon',pairs,bounds=(1500,1700),reference=(700,950),zero_run=0,use_correction=False)
    assert len(frame)==3, "Do not crop raw map to reference-supported well buffers"
    assert (frame.D_RMS>0).all()
    assert (~frame.QC_screening_pass).all()
    assert (frame.QC_reference_status=='RAW_ONLY').all()
    assert (qc.included==True).sum()==3
    assert qc.reason.str.startswith('raw_only').all()


def test_raw_candidate_regions_cover_far_from_wells():
    rows=[]
    # 15 x 15 survey cells; no wells or distance buffers.
    for il in range(300,450,10):
        for xl in range(2300,2450,10):
            hi = il<350 and xl<2350
            rows.append(dict(IL=il,XL=xl,D_RMS=9. if hi else 1.+(il-300)/150,
                             Local_contrast=np.nan,Reference_NRMS=np.nan,
                             X=460000+il,Y=570000+xl))
    frame=pd.DataFrame(rows)
    regions=A.candidate_regions(frame,10,metric='Raw_RMS_percentile',min_cells=4,low=.20,high=.80)
    assert regions
    assert any(r['Category'].startswith('Higher raw') for r in regions)
    assert sum(r['Cells'] for r in regions)>10
    # Ranking works with missing reference/production as explicit evidence gaps.
    ranked=A.rank_candidates(regions,frame,None,scenarios=12)
    assert not ranked.empty
    assert ranked.Production_status.eq('Not supplied').all()
    assert ranked.Evidence_coverage_pct.le(100).all()


def test_v91_opportunity_page_independent_full_scan():
    from pathlib import Path
    src=(Path(__file__).parents[1]/'analysis_dashboard.py').read_text()
    assert 'Calculate FULL seismic coverage + 4D attributes' in src
    assert 'Raw RMS percentile (whole target-valid survey)' in src
    assert 'Reference-supported local 4D contrast (strict QC)' in src
