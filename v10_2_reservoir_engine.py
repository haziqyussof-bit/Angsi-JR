"""ANGSI JR V10.2 — evidence-labelled, facies-sensitive volume scenarios.

NEVER equate 4D anomalies, bright amplitudes, or mapped footprints with proven
hydrocarbon closure. No production history exists in the supplied project; remaining
recoverable volumes require cumulative production supplied by the user.
All results are non-bookable exploration SCREENING SCENARIOS.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from v10_volumetrics import in_place

DEFAULT_MULTIPLIERS = {
    "BRIGHT_SAND_PRONE": (1.10, 1.15, -0.08),  # NTG multiplier, phi multiplier, Sw delta
    "INTERMEDIATE_UNCERTAIN": (0.70, 0.90, 0.04),
    "DIM_SHALE_PRONE": (0.15, 0.65, 0.14),
    "NO_VALID_FACIES": (1.0, 1.0, 0.0),
}


def candidate_facies_fractions(region, frame):
    """Read exact sampled cell keys, not nearest well or circular proximity masks."""
    keys=region.get('Keys',[])
    if frame is None or frame.empty or not keys or 'AmplitudeFaciesClass' not in frame:
        return dict(NO_VALID_FACIES=1.0), 0.0
    values=(frame.drop_duplicates(['IL','XL']).set_index(['IL','XL'])['AmplitudeFaciesClass'])
    labels=[]
    for pair in keys:
        try:
            label=str(values.get((int(pair[0]), int(pair[1])), 'NO_VALID_FACIES'))
        except (TypeError,ValueError):
            label='NO_VALID_FACIES'
        labels.append(label if label in DEFAULT_MULTIPLIERS else 'NO_VALID_FACIES')
    if not labels:
        return dict(NO_VALID_FACIES=1.0), 0.0
    counts=pd.Series(labels).value_counts(normalize=True).to_dict()
    return counts,float(1-counts.get('NO_VALID_FACIES',0))


def calculate_facies_volume(area_km2, gross_m, base_ntg, base_phi, base_sw, oil_fraction,
                             facies_shares, *, bo=1.25, bg=.005, scf_per_boe=6000,
                             multipliers=None, oil_rf=.25, gas_rf=.50):
    """Partition one opportunity footprint into mutually exclusive facies areas.

    If phase unknown, report OIL-ONLY and GAS-ONLY alternative end members;
    never pretend a single oil/gas split or sum both end members.
    """
    multipliers = multipliers or DEFAULT_MULTIPLIERS
    if not (np.isfinite([area_km2,gross_m,base_ntg,base_phi,base_sw,bo,bg]).all()
            and area_km2>0 and gross_m>0 and bo>0 and bg>0):
        raise ValueError('Valid area, gross thickness, log analog properties and FVF required')
    if not all(0<=x<=1 for x in (base_ntg,base_phi,base_sw,oil_rf,gas_rf)):
        raise ValueError('NTG/PHIE/Sw/recovery factors must be fractions 0–1')
    valid_shares={str(k):float(v) for k,v in facies_shares.items() if float(v)>0}
    if not valid_shares:
        valid_shares={'NO_VALID_FACIES':1.}
    total=sum(valid_shares.values())
    if not np.isfinite(total) or total<=0:
        raise ValueError('Invalid facies proportions')
    output=dict(GRV_Mm3=0.,HCPV_Mm3=0.,Oil_only_MMstb=0.,Gas_only_MMscf=0.,
                STOIIP_MMstb=np.nan,GIIP_MMscf=np.nan,Inplace_MMboe=np.nan,
                Recoverable_oil_MMstb=np.nan,Recoverable_gas_MMscf=np.nan,
                Facies_effective_NTG=0.,Facies_effective_PHIE=0.,Facies_effective_Sw=0.)
    seen=[]
    for cls, fraction in valid_shares.items():
        frac=fraction/total
        nfac,pfac,dsw=multipliers.get(cls,multipliers.get('NO_VALID_FACIES',(1.,1.,0.)))
        ntg=float(np.clip(base_ntg*nfac,0,1))
        phi=float(np.clip(base_phi*pfac,0,0.45))
        sw=float(np.clip(base_sw+dsw,0,1))
        oil=in_place(area_km2*frac,gross_m,ntg,phi,sw,bo,bg,1.,scf_per_boe)
        gas=in_place(area_km2*frac,gross_m,ntg,phi,sw,bo,bg,0.,scf_per_boe)
        output['GRV_Mm3']+=oil['GRV_m3']/1e6
        output['HCPV_Mm3']+=oil['HCPV_rm3']/1e6
        output['Oil_only_MMstb']+=oil['STOIIP_MMstb']
        output['Gas_only_MMscf']+=gas['GIIP_Bscf']*1000
        output['Facies_effective_NTG']+=frac*ntg
        output['Facies_effective_PHIE']+=frac*phi
        output['Facies_effective_Sw']+=frac*sw
        seen.append((cls,frac))
    if np.isfinite(oil_fraction) and 0<=oil_fraction<=1:
        output['STOIIP_MMstb']=output['Oil_only_MMstb']*oil_fraction
        output['GIIP_MMscf']=output['Gas_only_MMscf']*(1-oil_fraction)
        output['Inplace_MMboe']=output['STOIIP_MMstb'] + output['GIIP_MMscf']/scf_per_boe
        output['Recoverable_oil_MMstb']=output['STOIIP_MMstb']*oil_rf
        output['Recoverable_gas_MMscf']=output['GIIP_MMscf']*gas_rf
    output['Facies_breakdown']=' | '.join(f'{label}:{100*fraction:.0f}%' for label,fraction in seen)
    return output


def uniform_well_parameters(well_qc, fallbacks):
    """Same median measured / interpreted log properties for ALL polygons in a zone."""
    names=('Thickness_m','NTG','PHIE','Sw')
    lookup={'Thickness_m':'Gross_m','NTG':'NTG','PHIE':'PHIE','Sw':'Sw'}
    values=dict(fallbacks)
    n=0
    if well_qc is not None and isinstance(well_qc,pd.DataFrame) and not well_qc.empty:
        good=well_qc[well_qc.QC.eq('LOG_DERIVED')] if 'QC' in well_qc else pd.DataFrame()
        n=len(good)
        for key in names:
            col=lookup[key]
            if col in good:
                v=pd.to_numeric(good[col],errors='coerce').dropna()
                if key=='Thickness_m':v=v[v>0]
                else:v=v[v.between(0,1)]
                if not v.empty:values[key]=float(v.median())
    return values, n


def enrich_candidate_volumes(regions, frame, auto_table, *, reservoir, phase='Gas scenario',
                             fallback_thickness=20.,fallback_ntg=.45,fallback_phi=.18,
                             fallback_sw=.4,bo=1.25,bg=.005,oil_rf=.25,gas_rf=.50,
                             gross_cap_m=None,multipliers=None,weight_sensitivity=.25,
                             property_mode='Uniform reservoir log medians',uniform_log_params=None,uniform_log_count=0):
    """One row per candidate, facies-weighted. Analog and fallback provenance shown.

    *Only use physically calculated area* from V10A XY QC. Missing XY area never
    yields a volume. On absent logs, fallback petrophysics is labelled ASSUMED.
    """
    if auto_table is None or auto_table.empty or not regions:
        return pd.DataFrame()
    table=auto_table.set_index('ID',drop=False)
    rows=[]
    for r in regions:
        cid=r['ID']
        if cid not in table.index:
            continue
        src=table.loc[cid]
        if isinstance(src,pd.DataFrame):src=src.iloc[0]
        area=float(src.get('Area_km2',np.nan))
        if not np.isfinite(area) or area<=0:
            rows.append(dict(Reservoir=reservoir,ID=cid,Status='INVALID SURVEY XY AREA',Area_km2=np.nan))
            continue
        prop={}
        fallback=[]
        uniform=uniform_log_params or {}
        for target,source,default in (('Thickness_m','Thickness_m',fallback_thickness),('NTG','NTG',fallback_ntg),
                                      ('PHIE','PHIE',fallback_phi),('Sw','Sw',fallback_sw)):
            if property_mode=='Uniform reservoir log medians':
                v=float(uniform.get(target,default))
                if target not in uniform:fallback.append(target)
            else:
                v=pd.to_numeric(pd.Series([src.get(source,np.nan)]),errors='coerce').iloc[0]
                if not np.isfinite(v) or v<=0 and target=='Thickness_m':
                    v=default;fallback.append(target)
            prop[target]=float(v)
        if gross_cap_m is not None:
            # Optional scenario ceiling, never secretly tuned to a target volume.
            prop['Thickness_m']=min(prop['Thickness_m'],float(gross_cap_m))
            fallback.append('USER GROSS THICKNESS CAP')
        shares,facies_coverage=candidate_facies_fractions(r,frame)
        fluid=float(src.get('Oil_fraction',np.nan))
        if phase=='Gas scenario': fluid=0.
        elif phase=='Oil scenario':fluid=1.
        elif phase=='Model phase when valid':
            fluid=fluid if np.isfinite(fluid) else np.nan
        elif phase=='Unknown phase · alternatives':fluid=np.nan
        else:raise ValueError('Unknown fluid scenario')
        try:
            calc=calculate_facies_volume(area,prop['Thickness_m'],prop['NTG'],prop['PHIE'],prop['Sw'],fluid,shares,
                         bo=bo,bg=bg,oil_rf=oil_rf,gas_rf=gas_rf,multipliers=multipliers)
        except ValueError as exc:
            rows.append(dict(Reservoir=reservoir,ID=cid,Status=f'NOT QUANTIFIABLE: {exc}',Area_km2=area))
            continue
        row=dict(Reservoir=reservoir,ID=cid,Category=r['Category'],Area_km2=area,
                 Thickness_m=prop['Thickness_m'],Input_NTG=prop['NTG'],Input_PHIE=prop['PHIE'],Input_Sw=prop['Sw'],
                 Facies_QC_Coverage_pct=100*facies_coverage,
                 Phase_scenario=phase,Oil_fraction=fluid,Analog_wells=str(src.get('Analog_wells','')),
                 Input_source=('UNIFORM RESERVOIR MEDIANS' if property_mode=='Uniform reservoir log medians'
                               else 'SPATIAL WELL ANALOG') +
                              (' / assumed: '+','.join(fallback) if fallback else ' / available controls; verify') +
                              (f' / median of {uniform_log_count} QC-passing well(s)' if property_mode=='Uniform reservoir log medians' and uniform_log_count else
                               ' / NO VALID WELL CONTROL — assumption only' if property_mode=='Uniform reservoir log medians' else ''),
                 Structural_closure='NOT VALIDATED — area is a 4D anomaly footprint',
                 Status='EXPLORATORY FACIES SCREENING — NOT RESERVES')
        row.update(calc)
        for name in ('Oil_only_MMstb','Gas_only_MMscf','STOIIP_MMstb','GIIP_MMscf','Inplace_MMboe',
                     'Recoverable_oil_MMstb','Recoverable_gas_MMscf'):
            v=float(row[name]);lo=max(0,v*(1-weight_sensitivity)) if np.isfinite(v) else np.nan
            hi=v*(1+weight_sensitivity) if np.isfinite(v) else np.nan
            row[name.replace('_MM','_LOW_MM')]=lo
            row[name.replace('_MM','_HIGH_MM')]=hi
        rows.append(row)
    return pd.DataFrame(rows)


def remaining_with_production(volume_table, history_by_reservoir, *, phase='Gas scenario'):
    """Subtract *user-entered cumulative production* ONLY from recoverable scenarios.

    In-place minus cumulative production is NOT remaining recoverable reserves.
    If history unavailable, 'Remaining_*' stays NaN, not a false value.
    """
    df=volume_table.copy()
    if df.empty:return df
    df['Cumulative_produced_input']=np.nan
    df['Remaining_recoverable_MMstb']=np.nan
    df['Remaining_recoverable_MMscf']=np.nan
    df['Production_history_status']='NOT SUPPLIED'
    for res,rows in df.groupby('Reservoir',sort=False):
        value=history_by_reservoir.get(str(res),None)
        if value is None or not np.isfinite(float(value)) or float(value)<0:
            continue
        # Production is a reservoir-level figure: allocate by projected recovery,
        # not subtract the *same* produced volume from every polygon.
        column='Recoverable_gas_MMscf' if phase=='Gas scenario' else 'Recoverable_oil_MMstb'
        total=float(pd.to_numeric(rows[column],errors='coerce').fillna(0).sum())
        ratio=(pd.to_numeric(rows[column],errors='coerce').fillna(0)/total if total>0
               else pd.Series(0.,index=rows.index))
        allocated=float(value)*ratio
        df.loc[rows.index,'Cumulative_produced_input']=allocated
        df.loc[rows.index,'Production_history_status']=(
            'USER INPUT ALLOCATED BY RECOVERY SHARE — ILLUSTRATIVE, NOT ACTUAL WELL PRODUCTION')
        if float(value) > total:
            df.loc[rows.index,'Production_history_status']=(
                'INCONSISTENT: entered cumulative production exceeds candidate recoverable scenario')
        dest='Remaining_recoverable_MMscf' if phase=='Gas scenario' else 'Remaining_recoverable_MMstb'
        df.loc[rows.index,dest]=(pd.to_numeric(rows[column],errors='coerce')-allocated).clip(lower=0)
    return df


def volumes_by_reservoir(table):
    if table is None or table.empty:return pd.DataFrame()
    sums=['Area_km2','STOIIP_MMstb','GIIP_MMscf','Inplace_MMboe','Oil_only_MMstb',
          'Gas_only_MMscf','Recoverable_oil_MMstb','Recoverable_gas_MMscf',
          'Remaining_recoverable_MMstb','Remaining_recoverable_MMscf']
    existing=[c for c in sums if c in table]
    df=table.groupby('Reservoir',dropna=False)[existing].sum(min_count=1).reset_index()
    df.insert(1,'Candidate_count',table.groupby('Reservoir',dropna=False)['ID'].nunique().to_numpy())
    return df


def volume_ranking(table, phase='Gas scenario'):
    if table is None or table.empty:return pd.DataFrame()
    column={'Gas scenario':'GIIP_MMscf','Oil scenario':'STOIIP_MMstb',
            'Model phase when valid':'Inplace_MMboe','Unknown phase · alternatives':'Oil_only_MMstb'}[phase]
    if column not in table: return pd.DataFrame()
    good=table[pd.to_numeric(table[column],errors='coerce').notna()].copy()
    good=good.sort_values(column,ascending=False).reset_index(drop=True)
    good.insert(0,'Volume_rank',np.arange(1,len(good)+1))
    return good


def check_stratigraphic_disjoint_specs(runs, measured_tops=None):
    """Conservative NON-OVERLAP check for summing scenario totals across reservoirs.

    Requires all run modes to be two-marker intervals with ordered, measured TWT.
    Intervals sharing boundary are allowed. Horizontal polygon overlap is okay
    across distinct vertical stratigraphy but does not prove fluid independence.
    """
    if not runs:return False,'No stored reservoir intervals'
    if measured_tops is None or measured_tops.empty:return False,'No independently measured marker times'
    if not {'Marker','TWT_ms'}.issubset(measured_tops):return False,'Missing marker QC fields'
    times=(measured_tops.dropna(subset=['TWT_ms']).groupby('Marker').TWT_ms.median().to_dict())
    intervals=[]
    for name,run in runs.items():
        spec=run.get('spec',{})
        if spec.get('Mode')!='Interpreted horizon pair':
            return False,f'{name} is time/depth window, not a unique stratigraphic interval'
        a,b=spec.get('Top_marker'),spec.get('Base_marker')
        if a not in times or b not in times:return False,f'{name} has incomplete marker timing'
        start,end=float(times[a]),float(times[b])
        if not start<end:return False,f'{name}: top/base reversed or identical'
        intervals.append((start,end,name))
    intervals.sort()
    for first,second in zip(intervals[:-1],intervals[1:]):
        if second[0] < first[1]-1e-6:
            return False,f'Intervals {first[2]} and {second[2]} overlap in marker TWT'
    return True,'Candidate intervals are non-overlapping according to median measured marker TWT; geology and contacts still unvalidated'
