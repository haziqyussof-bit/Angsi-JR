"""ANGSI JR V10.1: measured-well-derived, provisional candidate-footprint volumes.

Read only. All outputs are SCREENING IN-PLACE scenarios, NOT reserves, not
remaining volumes, not independently proved sand/gas/oil. Candidate polygons
are *4D anomaly footprints* and MUST NOT be sold as reservoir closures.

A near-by well supplies a *local analog*, not evidence of reservoir connectivity.
FVF are explicitly assumed because no lab PVT inputs exist in this release.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from v10_volumetrics import in_place, SCF_PER_SM3, STB_PER_M3
from v10_features import affine_survey_xy, candidate_areas_km2

ANGSI_REFERENCE_WGS84 = (5.1957, 104.7093)  # public approximate field locality, NOT survey datum


def survey_crs_qc(base_grid):
    """Check affine projected metric geometry and geographic plausibility.

    EPSG:32648 is an ASSUMED CRS from geographic location, never established
    solely by an affine fit; map XY remains survey authoritative. Never silently
    reproject the data for volumetrics.
    """
    fit = affine_survey_xy(base_grid)
    result = dict(metric_geometry_valid=bool(fit.get('valid')), assumed_crs='WGS84 / UTM 48N (EPSG:32648) — UNVERIFIED',
                  area_bin_m2=float(fit.get('cell_area_m2',np.nan)), affine_rmse_m=float(fit.get('rmse_m',np.nan)),
                  lat=np.nan, lon=np.nan, geographic_plausibility='CRS not checked')
    if not fit.get('valid'):
        result['geographic_plausibility']=fit.get('reason','Invalid survey XY')
        return result
    try:
        from pyproj import Transformer
        x = pd.to_numeric(base_grid.X,errors='coerce').to_numpy(float)
        y = pd.to_numeric(base_grid.Y,errors='coerce').to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y) & (np.abs(x)>10000) & (np.abs(y)>10000)
        lon,lat=Transformer.from_crs(32648,4326,always_xy=True).transform(float(np.median(x[good])),float(np.median(y[good])))
        result['lat'],result['lon']=float(lat),float(lon)
        # a broad plausibility test, not formal CRS validation
        sep_km=111.2*np.hypot(lat-ANGSI_REFERENCE_WGS84[0],(lon-ANGSI_REFERENCE_WGS84[1])*np.cos(np.deg2rad(lat)))
        result['geographic_plausibility']=f'{sep_km:.1f} km from public Angsi location; survey CRS still requires metadata confirmation'
        result['epsg32648_plausible']=bool(sep_km<=80.0)
    except (ImportError,ValueError,TypeError) as e:
        result['geographic_plausibility']=f'Unable to transform: {e}'
    return result


def _extract_zone(w, specs, tops, checkshot):
    """Select well-local TVDSS interval; no checkshot extrapolation."""
    mode=str(specs.get('Mode',''))
    if mode=='Interpreted horizon pair':
        t,b=specs.get('Top_marker'),specs.get('Base_marker')
        marker=tops[tops.Well.eq(w)] if tops is not None and not tops.empty else pd.DataFrame()
        a=marker[marker.Marker.eq(t)] if not marker.empty else pd.DataFrame()
        z=marker[marker.Marker.eq(b)] if not marker.empty else pd.DataFrame()
        if len(a)==1 and len(z)==1:
            bounds=(float(a.iloc[0].TVDSS_m),float(z.iloc[0].TVDSS_m))
            if np.isfinite(bounds).all() and bounds[0]<bounds[1]:
                src=str(a.iloc[0].Depth_Source)+' / '+str(z.iloc[0].Depth_Source)
                return bounds,'well markers: '+str(t)+'→'+str(b)+' ('+src+')'
        return None,'missing or reversed measured marker pair'
    if mode=='Depth interval (TVDSS)':
        interval=specs.get('Depth_bounds_m')
        if interval is not None and len(interval)==2 and np.isfinite(interval).all() and interval[0]<interval[1]:
            return tuple(map(float,interval)),'display depth window (single reference velocity assumption)'
        return None,'invalid requested depth interval'
    if checkshot is None or len(checkshot)<2:
        return None,'missing own checkshot'
    t0,t1=specs.get('Top_ms',np.nan),specs.get('Base_ms',np.nan)
    t=checkshot.TWT_ms.to_numpy(float)
    z=checkshot.TVDSS_m.to_numpy(float)
    if not np.isfinite([t0,t1]).all() or t0>=t1 or min(t)<0 or t0<min(t) or t1>max(t):
        return None,'outside own checkshot range'
    return tuple(map(float,np.interp([t0,t1],t,z))),'own well checkshot'


def well_analog_records(wells, well_dir, tops, spec, *, minimum_samples=12,
                        phie_cutoff=.10, shale_cutoff=.40, sw_cutoff=.70):
    """Derive net fraction/PHIE/Sw from the actual LAS; exclude invalid zones.

    VOIL/VGAS, where present, only support a model-derived phase split.
    """
    import angsi_core as C
    rows=[]
    for w in wells.itertuples(index=False):
        rec=dict(Well=w.Well,X=float(w.X),Y=float(w.Y),Gross_m=np.nan,NTG=np.nan,PHIE=np.nan,Sw=np.nan,
                 Oil_fraction=np.nan,Phase_source='Unavailable',Evidence='No usable interval',NPay_samples=0,
                 QC='EXCLUDED',Shale_source='not available')
        try:
            cs=C.read_checkshot(well_dir,w.Well)
            bounds,origin=_extract_zone(w.Well,spec,tops,cs)
            if bounds is None:
                rec['Evidence']=origin;rows.append(rec);continue
            log=C.las_log(well_dir,w.Well)
            z=pd.to_numeric(log.iloc[:,0],errors='coerce').to_numpy(float)-float(w.RT)
            zone=np.isfinite(z)&(z>=bounds[0])&(z<=bounds[1])
            if zone.sum()<minimum_samples:
                rec['Evidence']='too few LAS samples in interval';rows.append(rec);continue
            gross=bounds[1]-bounds[0]
            subset=z[zone]
            coverage=(float(np.nanmax(subset))-float(np.nanmin(subset)))/gross if gross>0 else 0.
            if coverage<.55:
                rec['Evidence']=f'partial LAS coverage {coverage:.0%}';rows.append(rec);continue
            def curve(name):
                return pd.to_numeric(log[name],errors='coerce').to_numpy(float)[zone] if name in log else None
            phi=curve('PHIE')
            sw=curve('SWE')
            if phi is None or sw is None:
                rec['Evidence']='missing PHIE or SWE';rows.append(rec);continue
            sh=curve('VSHALE')
            shale_source='measured/interpreted VSHALE'
            if sh is None:
                gr=curve('GR_P')
                if gr is None:
                    rec['Evidence']='missing VSHALE and GR';rows.append(rec);continue
                grall=pd.to_numeric(log['GR_P'],errors='coerce').to_numpy(float)
                ref=grall[np.isfinite(grall)&(grall>=0)&(grall<=250)]
                if len(ref)<50:
                    rec['Evidence']='insufficient GR for shale proxy';rows.append(rec);continue
                a,b=np.percentile(ref,[10,90])
                if b-a<12:
                    rec['Evidence']='low GR dynamic range';rows.append(rec);continue
                sh=np.clip((gr-a)/(b-a),0,1)
                shale_source='UNVALIDATED GR index'
            valid=np.isfinite(phi)&np.isfinite(sw)&np.isfinite(sh)&(phi>=0)&(phi<=.6)&(sw>=0)&(sw<=1)&(sh>=0)&(sh<=1)
            if valid.sum()<minimum_samples or valid.mean()<.5:
                rec['Evidence']='insufficient valid PHIE/SWE/Vsh samples';rows.append(rec);continue
            pay=valid&(phi>=phie_cutoff)&(sh<=shale_cutoff)&(sw<=sw_cutoff)
            # Depth-weight by real sample spacing, capped to avoid one gap controlling statistics.
            dz=np.asarray(z[zone],float)
            order=np.argsort(dz)
            zz=dz[order];vv=valid[order];pp=pay[order]
            steps=np.diff(zz)
            dmed=float(np.median(steps[steps>0])) if (steps>0).any() else 1.
            wt=np.r_[np.minimum(np.maximum(steps,0),2*dmed),dmed]
            frac=float(wt[pp].sum()/max(wt[vv].sum(),1e-12))
            rec.update(Gross_m=float(gross),NTG=frac,NPay_samples=int(pay.sum()),Shale_source=shale_source,
                       Evidence=origin,QC='LOG_DERIVED')
            if pay.sum()<minimum_samples:
                # Valid zero pay is an informative geological result, not a failed computation.
                rec.update(PHIE=0.,Sw=1.,Oil_fraction=np.nan,Phase_source='No log-defined net pay')
            else:
                rec['PHIE']=float(np.nanmean(phi[pay]))
                rec['Sw']=float(np.nanmean(sw[pay]))
                oil=curve('VOIL'); gas=curve('VGAS')
                if oil is not None and gas is not None:
                    po=np.nanmedian(np.where(pay & np.isfinite(oil),oil,np.nan))
                    pg=np.nanmedian(np.where(pay & np.isfinite(gas),gas,np.nan))
                    if np.isfinite([po,pg]).all() and po>=0 and pg>=0 and po+pg>1e-5:
                        rec['Oil_fraction']=float(po/(po+pg))
                        rec['Phase_source']='VOIL/VGAS log-model ratio; UNCONFIRMED'
                if rec['Phase_source']=='Unavailable':
                    rec['Phase_source']='No phase discrimination; separate oil-only/gas-only alternatives'
        except Exception as e:
            rec['Evidence']=type(e).__name__+': '+str(e)[:110]
        rows.append(rec)
    return pd.DataFrame(rows)


def _nearest_analogs(cx,cy,analogs,k=3,preferred_radius_m=6000.):
    good=analogs[analogs.QC.eq('LOG_DERIVED')].dropna(subset=['X','Y','Gross_m','NTG','PHIE','Sw'])
    if good.empty:return None, 'NO_LOG_CONTROL',np.nan,[]
    xy=good[['X','Y']].to_numpy(float)
    d=np.hypot(xy[:,0]-cx,xy[:,1]-cy)
    order=np.argsort(d)[:min(k,len(good))]
    nearest=float(d[order[0]])
    selected=good.iloc[order].copy()
    distances=d[order]
    weights=1/np.maximum(distances,500.)**2
    weights=weights/weights.sum()
    # Do not treat a distant analog as measured candidate reservoir data
    label='NEARBY_LOG_ANALOG' if nearest<=preferred_radius_m else 'DISTANT_FIELD_ANALOG_EXTRAPOLATION'
    prop={col:float(np.sum(weights*selected[col].to_numpy(float))) for col in ('Gross_m','NTG','PHIE','Sw')}
    goodphase=selected.Oil_fraction.notna().to_numpy()
    prop['Oil_fraction']=float(np.average(selected.Oil_fraction.to_numpy(float)[goodphase],weights=weights[goodphase])) if goodphase.any() else np.nan
    return prop,label,nearest,selected.Well.tolist()


def _candidate_xy(region, affine_coeff):
    p=region['Geometry'].representative_point()
    # polygon geometry: x=IL, y=XL
    il,xl=float(p.x),float(p.y)
    return (np.array([il,xl,1.0])@np.asarray(affine_coeff,float)).tolist()


def candidate_horizon_thickness(region, frame, wells, well_dir, x, y):
    """Convert local horizon TWT pair to TVDSS thickness using nearest OWN checkshot.

    QC-labelled estimate only: spatial velocity variation and horizon picking remain
    uncalibrated. Never extrapolate outside the checkshot range.
    """
    if frame is None or frame.empty or not {'IL','XL','Top_ms','Base_ms'}.issubset(frame):
        return np.nan, 'No trace-level horizon pair'
    import angsi_core as C
    table=frame.set_index(['IL','XL'])
    keys=[tuple(map(int,k)) for k in region.get('Keys',[]) if tuple(map(int,k)) in table.index]
    if not keys:return np.nan, 'No matching candidate seismic cells'
    sub=table.loc[keys]
    if isinstance(sub,pd.Series):sub=sub.to_frame().T
    top=pd.to_numeric(sub['Top_ms'],errors='coerce').to_numpy(float)
    base=pd.to_numeric(sub['Base_ms'],errors='coerce').to_numpy(float)
    good=np.isfinite(top)&np.isfinite(base)&(base>top)
    if good.sum()<1:return np.nan,'Horizon picks unavailable or crossed'
    t0,t1=float(np.median(top[good])),float(np.median(base[good]))
    order=np.argsort(np.hypot(wells.X.to_numpy(float)-x,wells.Y.to_numpy(float)-y))
    for ix in order:
        name=str(wells.iloc[ix].Well)
        cs=C.read_checkshot(well_dir,name)
        if len(cs)<2:continue
        t=cs.TWT_ms.to_numpy(float);z=cs.TVDSS_m.to_numpy(float)
        if t0<min(t) or t1>max(t):continue
        z0,z1=np.interp([t0,t1],t,z)
        if np.isfinite([z0,z1]).all() and z1>z0:
            return float(z1-z0),f'seismic local Top/Base TWT + {name} OWN checkshot (estimated; verify velocity)'
    return np.nan,'No own checkshot covering local horizon interval'


def _quantiles(base,oil_fraction,bo,bg,uncertainty=.25,draws=500,seed=2026,scf_per_boe=6000.):
    """Sensitivity-only quantiles; independent triangular inputs, not geological P-values."""
    if not np.isfinite(oil_fraction):return dict(P90_MMboe=np.nan,P50_MMboe=np.nan,P10_MMboe=np.nan)
    rng=np.random.default_rng(seed)
    v=[]
    for key in ('Area_km2','Thickness_m','NTG','PHIE','Sw'):
        x=float(base[key]); lo=max(0.,x*(1-uncertainty)); hi=x*(1+uncertainty)
        seq=rng.triangular(lo,x,hi,draws) if hi>lo else np.repeat(x,draws)
        if key in ('NTG','PHIE','Sw'):seq=np.clip(seq,0,1)
        v.append(seq)
    area,thick,ntg,phi,sw=v
    hcpv=area*1e6*thick*ntg*phi*(1-sw)
    boe=(hcpv*oil_fraction/bo*STB_PER_M3 +
         hcpv*(1-oil_fraction)/bg*SCF_PER_SM3/scf_per_boe)/1e6
    p=np.quantile(boe,[.1,.5,.9]);return dict(zip(['P90_MMboe','P50_MMboe','P10_MMboe'],map(float,p)))


def auto_candidate_volumes(regions,base_grid,wells,well_dir,tops,spec,*,bo=1.25,bg=.005,
                           scf_per_boe=6000., uncertainty=.25, minimum_samples=12,
                           attribute_frame=None):
    """Return table (one row per 4D candidate), source well-QC table and spatial QC."""
    areas,fit=candidate_areas_km2(regions,base_grid,int(spec['Stride']))
    crs=survey_crs_qc(base_grid)
    logs=well_analog_records(wells,well_dir,tops,spec,minimum_samples=minimum_samples)
    data=[]
    for i,r in enumerate(regions):
        cid=r['ID'];area=areas.get(cid,np.nan)
        row=dict(ID=cid,Area_km2=area,GRV_Mm3=np.nan,HCPV_Mm3=np.nan,STOIIP_MMstb=np.nan,
                 GIIP_Bscf=np.nan,Inplace_MMboe=np.nan,P90_MMboe=np.nan,P50_MMboe=np.nan,P10_MMboe=np.nan,
                 Oil_only_STOIIP_MMstb=np.nan,Gas_only_GIIP_Bscf=np.nan,Thickness_m=np.nan,
                 NTG=np.nan,PHIE=np.nan,Sw=np.nan,Oil_fraction=np.nan,Analog_wells='',Nearest_analog_m=np.nan,
                 Phase_evidence='',Status='NOT QUANTIFIABLE',Centroid_E_m=np.nan,Centroid_N_m=np.nan,
                 Centroid_Lat_WGS84=np.nan,Centroid_Lon_WGS84=np.nan,
                 Area_source='4D anomaly footprint, NOT structural closure',
                 PVT_source=f'ASSUMED Bo={bo:g} res m3/ST m3, Bg={bg:g} res m3/std m3',
                 Thickness_source='')
        if not fit.get('valid') or not np.isfinite(area) or area<=0:
            row['Status']='Invalid XY geometry / physical area';data.append(row);continue
        x,y=_candidate_xy(r,fit['xy_per_grid'])
        row['Centroid_E_m'],row['Centroid_N_m']=x,y
        if crs.get('epsg32648_plausible',False):
            try:
                from pyproj import Transformer
                row['Centroid_Lon_WGS84'],row['Centroid_Lat_WGS84']=Transformer.from_crs(
                    32648,4326,always_xy=True).transform(x,y)
            except (ImportError,ValueError,TypeError):pass
        prop,flag,near,names=_nearest_analogs(x,y,logs)
        if prop is None:
            row['Status']='NO LOG PHIE/SWE/SHALE INTERVAL';data.append(row);continue
        local_thick,thick_source=candidate_horizon_thickness(r,attribute_frame,wells,well_dir,x,y)
        # Correctly label spatially predicted horizon thickness; do not claim that
        # median same-well thickness is a mapped closure.
        gross=local_thick if np.isfinite(local_thick) and local_thick>0 else prop['Gross_m']
        thickness_source=thick_source if gross==local_thick else 'well interval analog, unverified lateral extension'
        row.update(Thickness_m=gross,Thickness_source=thickness_source,
                   NTG=prop['NTG'],PHIE=prop['PHIE'],Sw=prop['Sw'],
                   Oil_fraction=prop['Oil_fraction'],Analog_wells=', '.join(names),Nearest_analog_m=near,
                   Phase_evidence='VOIL/VGAS model-derived' if np.isfinite(prop['Oil_fraction']) else 'unknown phase',
                   Status=flag+' / PRELIMINARY')
        # Gross from analog interval, NTG from in-interval cutoffs. NO independent mapped closure.
        if gross<=0 or prop['NTG']<0 or prop['PHIE']<0:
            row['Status']='Nonphysical analog properties';data.append(row);continue
        oc=in_place(area,gross,prop['NTG'],prop['PHIE'],prop['Sw'],bo,bg,1.,scf_per_boe)
        gc=in_place(area,gross,prop['NTG'],prop['PHIE'],prop['Sw'],bo,bg,0.,scf_per_boe)
        row.update(Oil_only_STOIIP_MMstb=oc['STOIIP_MMstb'],Gas_only_GIIP_Bscf=gc['GIIP_Bscf'],
                   GRV_Mm3=oc['GRV_m3']/1e6,HCPV_Mm3=oc['HCPV_rm3']/1e6)
        if np.isfinite(prop['Oil_fraction']):
            c=in_place(area,gross,prop['NTG'],prop['PHIE'],prop['Sw'],bo,bg,prop['Oil_fraction'],scf_per_boe)
            row.update(STOIIP_MMstb=c['STOIIP_MMstb'],GIIP_Bscf=c['GIIP_Bscf'],Inplace_MMboe=c['Inplace_MMboe'])
            q=_quantiles(row,prop['Oil_fraction'],bo,bg,uncertainty=uncertainty,seed=2026+i,scf_per_boe=scf_per_boe)
            row.update(q)
        else:
            row['Status']+=' / PHASE UNKNOWN — OIL/GAS END MEMBERS ONLY'
        data.append(row)
    return pd.DataFrame(data),logs,dict(affine=fit,crs=crs)
