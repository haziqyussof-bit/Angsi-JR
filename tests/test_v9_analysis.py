"""Self-contained V9 scientific unit tests; real SEG-Y integration remains HPC-only."""
import sys
import types
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

import analysis_engine as E


def test_trace_attributes_sinusoid_and_missing_spectral():
    t=np.arange(0,500,2)/1000
    a=np.sin(2*np.pi*10*t)
    out=E.trace_attributes(a,dt_ms=2)
    assert abs(out['RMS']-.707)<.02
    assert abs(out['DominantHz']-10)<2
    assert out['MaxPositive']>.95
    assert out['MostNegative']<-.95
    bad=np.array([np.nan]*30+[.1]*6+[np.nan]*30)
    assert np.isnan(E.trace_attributes(bad,2)['RMS'])
    # Do not interpolate across acquisition mute in Hilbert domain
    gapped=np.r_[np.ones(9),np.nan*np.ones(20),np.ones(9)]
    assert np.isnan(E.trace_attributes(gapped,2)['EnvelopeMean'])


def test_reservoir_screen_withheld_when_few_or_unvalidated():
    df=pd.DataFrame({'IL':[1,2],'XL':[5,5], 'B_RMS':[1.,2.], 'B_EnvelopeMean':[2.,3.], 'B_MostNegative':[-1.,-2.]})
    labels=pd.DataFrame({'Well':['A','B'], 'Label':['RESERVOIR','NON_RESERVOIR'],'IL':[1,2],'XL':[5,5]})
    result, report=E.reservoir_screen(df,labels)
    assert report['status']=='INSUFFICIENT'
    assert result['Reservoir_similarity_score'].isna().all()
    assert (result['Reservoir_proxy_class']=='Unclassified').all()


def test_connected_regions_preserve_missing_cells():
    values=[]
    for il in [100,110,120]:
        for xl in [200,210,220]:
            if il==110 and xl==210:
                continue
            values.append(dict(IL=il,XL=xl,Local_contrast=.3,Reference_NRMS=25.,
                               X=460000+il,Y=570000+xl,Window_RMS_span=.1))
    frame=pd.DataFrame(values)
    regions=E.candidate_regions(frame,10,min_cells=3,low=.7)
    assert len(regions)==1
    assert regions[0]['Cells']==8
    assert isinstance(regions[0]['Geometry'],Polygon)
    assert len(regions[0]['Geometry'].interiors)==1


def test_rank_reports_uncertainty_not_reserves():
    items=[]
    vals=[]
    for j,contrast in [(100,.3),(200,.5)]:
        keys=[(j,300),(j,310),(j,320)]
        from shapely.geometry import box
        from shapely.ops import unary_union
        geom=unary_union([box(il-5,xl-5,il+5,xl+5) for il,xl in keys])
        items.append({'ID':f'L{j}', 'Category':'Low 4D response','Cells':3,
                      'Keys':keys,'Geometry':geom,'Mean_contrast':contrast,
                      'Mean_reservoir_score':np.nan})
        for il,xl in keys:
            vals.append(dict(IL=il,XL=xl,Local_contrast=contrast,Reference_NRMS=20.,X=460000+il,Y=570000+xl))
    ranked=E.rank_candidates(items,pd.DataFrame(vals),None,scenarios=35)
    assert ranked['Evidence_coverage_pct'].le(100).all()
    assert ranked['Evidence_coverage_pct'].lt(100).all()
    assert (ranked['Production_status']=='Not supplied').all()
    assert (ranked['Best_rank']<=ranked['Worst_rank']).all()
    assert np.all((ranked['First_rank_scenario_share']>=0)&(ranked['First_rank_scenario_share']<=1))


def test_fake_segy_attributes_raw_and_excluded(monkeypatch):
    import seismic_qc
    t=np.arange(0,2602,2,dtype=float)
    wave=np.sin(t/50.0)
    baseline=np.column_stack([wave, wave])
    monitor=np.column_stack([wave*1.1, wave*1.1])
    class Volume:
        def __init__(self,arr): self.arr=arr; self.samples=t; self.trace=[arr[:,i] for i in range(2)]
        def __enter__(self):return self
        def __exit__(self,*args):pass
    volumes={'baseline':Volume(baseline),'monitor':Volume(monitor)}
    monkeypatch.setitem(sys.modules,'segyio',types.SimpleNamespace(open=lambda key,*args,**kwargs:volumes[key]))
    pairs=pd.DataFrame([{'IL':10,'XL':20,'X':450000,'Y':570000,'bi':0,'mi':0},
                        {'IL':10,'XL':30,'X':450010,'Y':570000,'bi':1,'mi':1}])
    top={(10,20):1500.,(10,30):1900.}
    base={(10,20):1700.,(10,30):1800.}
    frame,qc=E.compute_attribute_volume('baseline','monitor',pairs,top_map=top,base_map=base,
                                         use_correction=False,zero_run=0)
    assert len(frame)==1
    assert frame.iloc[0]['D_RMS']>0
    assert len(qc)==2
    assert (qc['reason']=='crossing_or_zero_thickness_horizons').sum()==1
    assert (qc['included']==True).sum()==1


def test_v9_main_preserves_navigation():
    from pathlib import Path
    src=(Path(__file__).parents[1]/'angsi_jr_v9.py').read_text()
    for title in ('Overview', 'Seismic QC', 'AI Horizon 2D & 3D Interpretation',
                  'Well Correlation', 'Analysis', 'Opportunities'):
        assert title in src
    assert 'V9.opportunities_bridge(data)' in src
    assert 'V8.screening(data' in src
