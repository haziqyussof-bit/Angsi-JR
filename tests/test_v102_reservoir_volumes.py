"""Small synthetic checks; NEVER actual Angsi reserve validation."""
import numpy as np
import pandas as pd
from shapely.geometry import box
import v10_2_reservoir_engine as eng
import v10_2_copilot as brain


def fake():
    regions=[dict(ID='A',Category='Higher 4D response',Keys=[(800,2500),(810,2500)],Geometry=box(795,2495,815,2505)),
             dict(ID='B',Category='Low 4D response',Keys=[(800,2510),(810,2510)],Geometry=box(795,2505,815,2515))]
    frame=pd.DataFrame({'IL':[800,810,800,810],'XL':[2500,2500,2510,2510],
           'AmplitudeFaciesClass':['BRIGHT_SAND_PRONE','BRIGHT_SAND_PRONE','INTERMEDIATE_UNCERTAIN','DIM_SHALE_PRONE']})
    volumes=pd.DataFrame({'ID':['A','B'],'Area_km2':[.12,.10],'Thickness_m':[25,25],'NTG':[.60,.60],
           'PHIE':[.20,.20],'Sw':[.30,.30],'Oil_fraction':[np.nan,np.nan],
           'Analog_wells':['Well-4','Well-5']})
    return regions,frame,volumes


def test_bright_scenario_exceeds_intermediate_dim_with_same_area():
    bright=eng.calculate_facies_volume(.1,20,.5,.2,.4,0.,{'BRIGHT_SAND_PRONE':1.})
    dim=eng.calculate_facies_volume(.1,20,.5,.2,.4,0.,{'DIM_SHALE_PRONE':1.})
    assert bright['GIIP_MMscf']>dim['GIIP_MMscf']>0
    assert np.isclose(bright['GIIP_MMscf']/1000+0,bright['Gas_only_MMscf']/1000)
    assert bright['STOIIP_MMstb']==0


def test_all_candidates_and_volume_rank_are_independent_of_evidence_score():
    r,f,a=fake()
    out=eng.enrich_candidate_volumes(r,f,a,reservoir='I-25 → I-35',phase='Gas scenario')
    assert len(out)==2
    assert out.GIIP_MMscf.notna().sum()==2
    assert set(out.ID)=={'A','B'}
    assert out.loc[out.ID.eq('A'),'GIIP_MMscf'].iloc[0]>out.loc[out.ID.eq('B'),'GIIP_MMscf'].iloc[0]
    sorted_volume=eng.volume_ranking(out,'Gas scenario')
    assert sorted_volume.ID.iloc[0]=='A'
    assert sorted_volume.Volume_rank.tolist()==[1,2]


def test_undefined_fluid_phase_is_not_double_counted():
    r,f,a=fake()
    out=eng.enrich_candidate_volumes(r,f,a,reservoir='I-25 → I-35',phase='Unknown phase · alternatives')
    assert out.GIIP_MMscf.isna().all()
    assert out.STOIIP_MMstb.isna().all()
    assert out.Oil_only_MMstb.gt(0).all() and out.Gas_only_MMscf.gt(0).all()


def test_no_production_means_no_remaining_and_allocations_do_not_double_count():
    r,f,a=fake()
    out=eng.enrich_candidate_volumes(r,f,a,reservoir='I-25 → I-35',phase='Gas scenario')
    absent=eng.remaining_with_production(out,{},phase='Gas scenario')
    assert absent.Remaining_recoverable_MMscf.isna().all()
    produced=eng.remaining_with_production(out,{'I-25 → I-35':7.},phase='Gas scenario')
    assert np.isclose(produced.Cumulative_produced_input.sum(),7.)
    assert produced.Remaining_recoverable_MMscf.notna().all()
    summary=eng.volumes_by_reservoir(produced)
    assert len(summary)==1 and summary.Candidate_count.iloc[0]==2


def test_missing_area_is_excluded_without_fabricated_value():
    r,f,a=fake()
    a.loc[1,'Area_km2']=np.nan
    out=eng.enrich_candidate_volumes(r,f,a,reservoir='TEST',phase='Oil scenario')
    assert np.isfinite(out.loc[out.ID.eq('A'),'STOIIP_MMstb']).all()
    assert out.loc[out.ID.eq('B'),'Status'].iloc[0]=='INVALID SURVEY XY AREA'
    assert eng.volume_ranking(out,'Oil scenario').ID.tolist()==['A']


def test_curated_offline_copilot_has_real_science_and_links_without_endpoint():
    greeting=brain.answer('Hi')
    science=brain.answer('How does 4D seismic NRMS help reservoir monitoring?')
    volumes=brain.answer('Why STOIIP is not remaining?')
    assert 'Hi!' in greeting
    assert '4D' in science and 'http' in science and 'live model' not in science
    assert 'remaining' in volumes and 'http' in volumes
    assert brain.TOPIC_CHOICES.get('Volumetrics')


def test_uniform_reservoir_log_medians_same_base_input_all_candidates():
    r,f,a=fake()
    logs=pd.DataFrame({'Well':['Well-1','Well-2','Well-3'],
           'QC':['LOG_DERIVED','LOG_DERIVED','EXCLUDED'],
           'Gross_m':[18.,22.,80.],'NTG':[.50,.70,.05],
           'PHIE':[.18,.22,.05],'Sw':[.25,.35,.80]})
    med,n=eng.uniform_well_parameters(logs,{'Thickness_m':20.,'NTG':.45,'PHIE':.18,'Sw':.4})
    assert n==2 and med['Thickness_m']==20. and np.isclose(med['PHIE'],.20)
    out=eng.enrich_candidate_volumes(r,f,a,reservoir='I-25 → I-35',phase='Gas scenario',
             uniform_log_params=med,uniform_log_count=n)
    assert out.Input_NTG.nunique()==1
    assert out.Input_PHIE.nunique()==1
    assert out.Input_Sw.nunique()==1
    assert out.GIIP_MMscf.iloc[0] != out.GIIP_MMscf.iloc[1]


def test_portfolio_requires_nonoverlapping_measured_marker_pairs():
    tops=pd.DataFrame({'Marker':['I-25','I-35','I-68'],
                       'TWT_ms':[1500.,1600.,1700.]})
    runs={'A':{'spec':dict(Mode='Interpreted horizon pair',Top_marker='I-25',Base_marker='I-35')},
          'B':{'spec':dict(Mode='Interpreted horizon pair',Top_marker='I-35',Base_marker='I-68')}}
    assert eng.check_stratigraphic_disjoint_specs(runs,tops)[0]
    runs['B']['spec']['Top_marker']='I-25'
    assert not eng.check_stratigraphic_disjoint_specs(runs,tops)[0]


def test_offline_copilot_summarizes_existing_local_runs_no_llm():
    _,_,auto=fake()
    runs={'I25':{'phase':'Gas scenario','volumes':pd.DataFrame({'GIIP_MMscf':[2.0,3.0]})}}
    evidence=brain.local_volume_evidence(runs)
    assert len(evidence)==1 and evidence[0]['GIIP_MMscf']==5.0
    reply=brain.answer('How much GIIP?',evidence,'Local dashboard candidates')
    assert '5.0' in reply and 'NO' in reply and 'http' in reply
