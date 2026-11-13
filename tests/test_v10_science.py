import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v10_features import (facies_colorscale,fluid_only_grid,extend_route_xy,
                          affine_survey_xy,candidate_areas_km2,map_bubble_layer)
from v10_volumetrics import in_place,sensitivity_scenarios,STB_PER_M3
from v10_tie_viewer import qc_trace_figure,seismic_tie_figure,shifted_model
import copilot


def test_facies_colors_and_fluid_only():
    assert facies_colorscale()[0][1]=='#879099'
    assert facies_colorscale()[2][1]=='#28ab66'
    assert facies_colorscale()[-1][1]=='#f6d349'
    d=np.array([100.,110.,120.,130.,140.])
    fluid=['Water model','Oil-model','HC candidate','Gas-model','Uncertain']
    z=fluid_only_grid(d,fluid,top=100,bottom=140,rows=5,columns=3)
    assert np.isnan(z[0]).all() and np.isnan(z[2]).all() and np.isnan(z[4]).all()
    assert (z[1]==0).all() and (z[3]==1).all()


def test_fence_extension_preserves_stations():
    points=np.array([[10.,20.],[1010.,20.],[2010.,120.]])
    full,stations=extend_route_xy(points,500)
    assert len(full)==5 and np.allclose(full[1:-1],points)
    assert abs(stations[0]-.5)<1e-8
    assert np.all(np.diff(stations)>0)
    assert np.linalg.norm(full[0]-points[0])==pytest.approx(500)
    assert np.linalg.norm(full[-1]-points[-1])==pytest.approx(500)


def test_affine_xy_and_candidate_area():
    records=[]
    for il in range(10):
        for xl in range(10):
            records.append({'IL':il,'XL':xl,'X':450000+25*il+4*xl,'Y':560000-3*il+25*xl})
    b=pd.DataFrame(records)
    fit=affine_survey_xy(b)
    assert fit['valid'] and fit['rmse_m']<1e-6
    area=25*25+12
    assert fit['cell_area_m2']==pytest.approx(area)
    aa,_=candidate_areas_km2([{'ID':'H01','Cells':20}],b,5)
    assert aa['H01']==pytest.approx(20*25*area/1e6)


def test_volumetric_oil_and_gas_units():
    oil=in_place(1,10,1,.20,.25,bo=1.25,bg=.005,oil_fraction=1)
    assert oil['HCPV_rm3']==pytest.approx(1e6*10*.2*.75)
    assert oil['STOIIP_MMstb']==pytest.approx(oil['HCPV_rm3']/1.25*STB_PER_M3/1e6)
    assert oil['GIIP_Bscf']==0
    gas=in_place(1,10,1,.20,.25,bo=1.25,bg=.005,oil_fraction=0)
    assert gas['STOIIP_MMstb']==0 and gas['GIIP_Bscf']>0
    assert gas['Inplace_MMboe']==pytest.approx(gas['GIIP_Bscf']/6)
    mixed=in_place(1,10,1,.20,.25,bo=1.25,bg=.005,oil_fraction=.30)
    assert mixed['STOIIP_MMstb']==pytest.approx(oil['STOIIP_MMstb']*.3)
    assert mixed['GIIP_Bscf']==pytest.approx(gas['GIIP_Bscf']*.7)


def test_volumetric_sensitivity_repeatability():
    args=dict(area_km2=2,thickness_m=20,ntg=.7,porosity=.2,sw=.3,bo=1.3,bg=.005,oil_fraction=.8,n=100,seed=12)
    a=sensitivity_scenarios(**args);b=sensitivity_scenarios(**args)
    assert a==b
    for stats in a.values(): assert stats['P90']<=stats['P50']<=stats['P10']


def test_volumetric_invalid_inputs():
    with pytest.raises(ValueError):in_place(1,-2,.6,.2,.3)
    with pytest.raises(ValueError):in_place(1,20,1.1,.2,.3)
    with pytest.raises(ValueError):in_place(1,20,.5,.2,.3,bo=-1)


def test_tie_plot_and_shift():
    t=np.arange(0.,100.,2.)
    trace=np.sin(t*.1)
    model=np.sin(t*.1)
    moved=shifted_model(t,model,4.)
    assert np.isnan(moved[0]) and np.isfinite(moved[10])
    mask=np.isfinite(moved)
    fig=qc_trace_figure(t,trace,moved,mask)
    assert len(fig.data)==2 and fig.layout.legend.orientation=='h'
    panel=seismic_tie_figure(t,np.arange(50,60),np.ones((len(t),10)),55.,model,np.isfinite(model))
    assert len(panel.data)>=2
    assert panel.layout.yaxis.autorange=='reversed'


def test_private_http_requires_explicit_consent():
    with pytest.raises(ValueError):copilot.valid_endpoint('http://172.28.34.1:8000/v1')
    assert copilot.valid_endpoint('http://172.28.34.1:8000/v1',True).endswith('/v1')
    with pytest.raises(ValueError):copilot.valid_endpoint('http://8.8.8.8:8000/v1',True)
    with pytest.raises(ValueError):copilot.valid_endpoint('http://192.0.2.1:8000/v1',True)
    with pytest.raises(ValueError):copilot.valid_endpoint('http://172.28.34.1:8000/v1?token=123',True)
    assert copilot.valid_endpoint('https://example.com/v1').startswith('https')


def test_opportunity_bubbles_keep_physical_centroids():
    from shapely.geometry import box
    import plotly.graph_objects as go
    ranks=pd.DataFrame({'ID':['H01','L02'],'Cells':[40,12],'Category':['Higher raw 4D RMS response','Low raw 4D RMS response']})
    regions=[{'ID':'H01','Geometry':box(100,200,120,220)},
             {'ID':'L02','Geometry':box(300,400,310,410)}]
    fig=map_bubble_layer(go.Figure(),ranks,regions,max_labels=2)
    assert len(fig.data)==1
    assert fig.data[0].x[0]==210 and fig.data[0].y[0]==110
    assert fig.data[0].x[1]==405 and fig.data[0].y[1]==305


def test_v10_app_compiles_without_runtime_deps():
    import ast
    root=Path(__file__).resolve().parents[1]
    files=['angsi_jr_v9.py','analysis_dashboard.py','horizon_review.py','fence.py','angsi_core.py']
    for name in files:
        ast.parse((root/name).read_text())
    source=(root/'analysis_dashboard.py').read_text()
    assert 'v10_4d_transparency' in source and 'v10_opportunity_bubbles' in source
    assert 'Calculate STOIIP / GIIP / BOE scenarios' in source
    assert 'v10_composite_on' in source
