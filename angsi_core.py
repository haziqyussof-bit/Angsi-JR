"""Angsi JR seismic/well data and conservative interpretation engines.

Read-only against the original GeoHackathon dataset. All interpreted horizons
are previews, NOT audited seismic interpretations, reserves estimates or fluid picks.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import segyio
import lasio

from interpretation import convert, reflector
from seismic_qc import sample_mask

DATA_ROOT = Path(os.getenv("ANGSI_DATA_ROOT", "/data/epic20/APGCE/input_data/challenge3"))


def find_project(root: Path = DATA_ROOT):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    folder = root / "volumes"
    seismic_files = [p for p in folder.iterdir() if p.suffix.lower() in (".sgy", ".segy")]
    base = [p for p in seismic_files if "baseline" in p.name.lower()]
    mon = [p for p in seismic_files if "monitor" in p.name.lower()]
    if len(base) != 1 or len(mon) != 1:
        raise FileNotFoundError("Need exactly one Baseline and one Monitor SEG-Y in volumes/")
    well_dir = root / "well_data"
    excels = list(well_dir.glob("*.xlsx"))
    if not excels:
        raise FileNotFoundError("Missing well list .xlsx file")
    preferred = [x for x in excels if "welllist" in x.name.lower()]
    return base[0], mon[0], well_dir, (preferred or excels)[0]


def well_table(excel_path):
    df = pd.read_excel(excel_path, header=2)
    df = df.rename(columns={"Well Name": "Well"})
    if not {"Well", "X", "Y", "RT"}.issubset(df.columns):
        raise ValueError("Expected Well Name, X, Y, RT at Excel header row 3")
    df["Well"] = df["Well"].astype(str).str.strip()
    df = df[df.Well.str.fullmatch(r"Well-\d+")].copy()
    for key in ("X", "Y", "RT", "WD", "WB", "TD"):
        if key in df:
            df[key] = pd.to_numeric(df[key], errors="coerce")
    df = df.dropna(subset=["X", "Y", "RT"]).drop_duplicates("Well")
    df["Well_No"] = df.Well.str.extract(r"(\d+)").astype(int)
    return df.sort_values("Well_No").reset_index(drop=True)


def las_log(well_dir, well):
    path = Path(well_dir) / "las" / f"{well}.las"
    log = lasio.read(str(path), engine="normal")
    return log.df().reset_index()


def read_checkshot(well_dir, well):
    path = Path(well_dir) / "checkshots" / f"{well}_Checkshot.txt"
    if not path.exists():
        return pd.DataFrame(columns=["TVDSS_m", "TWT_ms"])
    cs = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                     names=["TVDSS_m", "TWT_ms"], usecols=[0, 1])
    for col in ("TVDSS_m", "TWT_ms"):
        cs[col] = pd.to_numeric(cs[col], errors="coerce")
    cs = cs.dropna().drop_duplicates().sort_values("TVDSS_m").reset_index(drop=True)
    if len(cs) < 2 or not np.isfinite(convert(cs.TVDSS_m, cs.TVDSS_m, cs.TWT_ms)).all():
        return cs.iloc[:0]
    return cs


def nearest_donor(well, wells, well_dir):
    own = read_checkshot(well_dir, well)
    if len(own) >= 2:
        return well, 0.0, "OWN"
    available = [w for w in wells.Well if len(read_checkshot(well_dir, w)) >= 2]
    if not available:
        return None, np.nan, "MISSING"
    table = wells.set_index("Well")
    target = table.loc[well, ["X", "Y"]].to_numpy(float)
    arr = table.loc[available, ["X", "Y"]].to_numpy(float)
    distances = np.linalg.norm(arr-target, axis=1)
    ix = int(np.argmin(distances))
    return available[ix], float(distances[ix]), "BORROWED"


def marker_file(well_dir, well, rt):
    path = Path(well_dir) / "markers" / f"{well}_Markers.txt"
    cols = ["Marker", "MDKB_m", "TVDSS_m", "Depth_Source", "Datum_Diff_m"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    headers, rows = [], []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            headers.append(line.upper())
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        h = " ".join(headers)
        md, tvd = np.nan, np.nan
        src = "none"
        if "MDKB" in h and "TVDSS" in h and len(vals) >= 2:
            md, tvd = vals[:2]
            src = "provided_both"
        elif "MDKB" in h:
            md = vals[0]
            tvd = md - float(rt)
            src = "assumed_vertical_from_MDKB"
        elif "TVDSS" in h:
            tvd = vals[0]
            src = "provided_TVDSS"
        if not np.isfinite(tvd):
            continue
        err = md - float(rt) - tvd if np.isfinite(md) else np.nan
        rows.append([parts[0], md, tvd, src, err])
    return pd.DataFrame(rows, columns=cols)


def marker_inventory(wells, well_dir):
    all_rows = []
    for w in wells.itertuples():
        donor, distance, status = nearest_donor(w.Well, wells, well_dir)
        mk = marker_file(well_dir, w.Well, w.RT)
        cs = read_checkshot(well_dir, donor) if donor else pd.DataFrame()
        for _, row in mk.iterrows():
            z = float(row.TVDSS_m)
            twt = np.nan
            if len(cs) >= 2 and cs.TVDSS_m.min() <= z <= cs.TVDSS_m.max():
                twt = float(depth_to_time([z], cs)[0])
            all_rows.append({
                "Well": w.Well, "Marker": row.Marker,
                "MDKB_m": row.MDKB_m, "TVDSS_m": z, "TWT_ms": twt,
                "Depth_Source": row.Depth_Source,
                "Checkshot_Donor": donor, "Donor_Distance_m": distance,
                "Checkshot_Type": status, "Datum_Diff_m": row.Datum_Diff_m
            })
    return pd.DataFrame(all_rows, columns=[
        "Well", "Marker", "MDKB_m", "TVDSS_m", "TWT_ms", "Depth_Source",
        "Checkshot_Donor", "Donor_Distance_m", "Checkshot_Type", "Datum_Diff_m"
    ])


def depth_to_time(z, cs):
    if cs is None or len(cs) < 2:
        return np.full_like(np.asarray(z, dtype=float), np.nan)
    return convert(z, cs.TVDSS_m, cs.TWT_ms)


def time_to_depth(t, cs):
    if cs is None or len(cs) < 2:
        return np.full_like(np.asarray(t, dtype=float), np.nan)
    return convert(t, cs.TWT_ms, cs.TVDSS_m)


def seismic_grid(path, xy=False):
    # For monitor XY, headers are not valid: use IL/XL only.
    with segyio.open(str(path), "r", strict=False) as f:
        columns = {
            "idx": np.arange(f.tracecount, dtype=np.int32),
            "IL": np.asarray(f.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int32),
            "XL": np.asarray(f.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int32)
        }
        if xy:
            rawx = np.asarray(f.attributes(segyio.TraceField.CDP_X)[:], dtype=float)
            rawy = np.asarray(f.attributes(segyio.TraceField.CDP_Y)[:], dtype=float)
            scalar = np.asarray(f.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=float)
            scale = np.where(scalar < 0, 1.0/np.maximum(abs(scalar), 1),
                             np.where(scalar == 0, 1.0, scalar))
            columns["X"], columns["Y"] = rawx*scale, rawy*scale
    return pd.DataFrame(columns)


def well_seismic_matches(wells, base_grid, mon_grid):
    if base_grid.duplicated(["IL", "XL"]).any() or mon_grid.duplicated(["IL", "XL"]).any():
        raise ValueError("Duplicate IL/XL header pairs: investigate geometry first")
    xy = base_grid[["X", "Y"]].to_numpy(float)
    valid = np.isfinite(xy).all(axis=1) & (np.abs(xy).sum(axis=1) > 0)
    validgrid = base_grid.loc[valid].reset_index(drop=True)
    if validgrid.empty:
        raise ValueError("No valid baseline CDP coordinates")
    d, idx = cKDTree(validgrid[["X", "Y"]].to_numpy()).query(wells[["X", "Y"]].to_numpy(float))
    selected = validgrid.iloc[idx].reset_index(drop=True)
    out = wells[["Well", "X", "Y"]].copy().reset_index(drop=True)
    out["Distance_m"] = d
    out["IL"], out["XL"] = selected.IL.to_numpy(), selected.XL.to_numpy()
    out["Base_Index"] = selected.idx.to_numpy()
    ml = mon_grid.rename(columns={"idx": "Monitor_Index"})
    return out.merge(ml[["IL", "XL", "Monitor_Index"]], on=["IL", "XL"], how="left", validate="many_to_one")


def coverage_pairs(base_grid, mon_grid, orientation="Inline", value=864):
    if orientation not in ("Inline", "Crossline"):
        raise ValueError("orientation must be Inline or Crossline")
    fixed, moving = ("IL", "XL") if orientation == "Inline" else ("XL", "IL")
    b = base_grid.loc[base_grid[fixed] == int(value), ["IL", "XL", "idx"]].rename(columns={"idx": "bi"})
    m = mon_grid.loc[mon_grid[fixed] == int(value), ["IL", "XL", "idx"]].rename(columns={"idx": "mi"})
    pair = b.merge(m, on=["IL", "XL"], validate="one_to_one").sort_values(moving)
    return pair.reset_index(drop=True)


def read_paired_section(bpath, mpath, pairs, max_traces=340):
    if pairs.empty:
        return None
    if len(pairs) > max_traces:
        pairs = pairs.iloc[np.linspace(0, len(pairs)-1, max_traces).astype(int)].copy()
    with segyio.open(str(bpath), "r", strict=False) as b, segyio.open(str(mpath), "r", strict=False) as m:
        tb = np.asarray(b.samples, dtype=float).copy()
        tm = np.asarray(m.samples, dtype=float).copy()
        if len(tb) != len(tm) or not np.allclose(tb, tm, atol=0.05):
            raise ValueError("Seismic time sampling differs; resampling/registration is required")
        bv = np.stack([np.asarray(b.trace[int(i)], dtype=np.float32) for i in pairs.bi], axis=1)
        mv = np.stack([np.asarray(m.trace[int(i)], dtype=np.float32) for i in pairs.mi], axis=1)
    return pairs.reset_index(drop=True), tb, bv, mv


def valid_pair(b, m):
    b = np.asarray(b, dtype=float)
    m = np.asarray(m, dtype=float)
    return np.isfinite(b) & np.isfinite(m)


def nrms(b, m, axis=None):
    valid = valid_pair(b, m)
    x, y = np.where(valid, b, np.nan), np.where(valid, m, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = np.sqrt(np.nanmean(x*x, axis=axis))
        rm = np.sqrt(np.nanmean(y*y, axis=axis))
        rd = np.sqrt(np.nanmean((y-x)**2, axis=axis))
        return np.where(rb+rm > 0, 200*rd/(rb+rm), np.nan)


def nearest_trace_section(bpath, mpath, base_grid, mon_grid, points_xy, max_distance_m=100,
                          extension_m=0):
    """Sample an arbitrary polyline from E to W via existing well XY locations."""
    base_xy = base_grid[["X", "Y"]].to_numpy(float)
    good = np.isfinite(base_xy).all(axis=1) & (base_xy[:, 0] != 0)
    sub = base_grid.loc[good].reset_index(drop=True)
    if sub.empty or len(points_xy) < 2:
        return None
    tree = cKDTree(sub[["X", "Y"]].to_numpy())
    xy = np.asarray(points_xy, float)
    # V10 extends BOTH edges beyond outermost wells where real paired traces exist.
    # Returned station distances refer to the ORIGINAL wells on the longer route.
    if extension_m:
        from v10_features import extend_route_xy
        xy, well_stations = extend_route_xy(xy,extension_m)
    else:
        well_stations = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]/1000
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if cum[-1] <= 0:
        return None
    sample_m = np.unique(np.r_[np.arange(0, cum[-1], 100.0), cum])
    px = np.interp(sample_m, cum, xy[:, 0])
    py = np.interp(sample_m, cum, xy[:, 1])
    dist, ind = tree.query(np.column_stack([px, py]))
    picked = sub.iloc[ind].reset_index(drop=True)
    mlookup = mon_grid.set_index(["IL", "XL"])["idx"]
    keys = pd.MultiIndex.from_arrays([picked.IL, picked.XL])
    monidx = mlookup.reindex(keys).to_numpy(dtype=float)
    mask = (dist <= max_distance_m) & np.isfinite(monidx)
    if mask.sum() < 2:
        return None
    pairs = pd.DataFrame({"bi": picked.idx.to_numpy()[mask], "mi": monidx[mask].astype(int),
                          "distance_km": sample_m[mask]/1000})
    pairs = pairs.loc[~pairs.bi.duplicated(keep="first")].reset_index(drop=True)
    return (pairs, well_stations) if len(pairs)>=2 else None


def read_arbitrary(bpath, mpath, points, max_traces=250):
    if points.empty: return None
    if len(points) > max_traces:
        points = points.iloc[np.linspace(0, len(points)-1, max_traces).astype(int)]
    with segyio.open(str(bpath), "r", strict=False) as b, segyio.open(str(mpath), "r", strict=False) as m:
        t = np.asarray(b.samples, dtype=float).copy()
        tm = np.asarray(m.samples, dtype=float).copy()
        if len(t) != len(tm) or not np.allclose(t, tm, atol=0.05):
            raise ValueError("Time axis mismatch")
        bv = np.stack([np.asarray(b.trace[int(i)], float) for i in points.bi], axis=1)
        mv = np.stack([np.asarray(m.trace[int(i)], float) for i in points.mi], axis=1)
    return points.distance_km.to_numpy(), t, bv, mv


def trend_surface(coords, anchor_xy, anchor_twt):
    """Smooth, lightly regularized structural trend; extrapolations are *predictions*."""
    coords = np.asarray(coords, float)
    sites = np.asarray(anchor_xy, float)
    z = np.asarray(anchor_twt, float)
    if len(sites) == 0: return np.full(len(coords), np.nan), np.full(len(coords), np.inf)
    center = np.nanmean(sites, axis=0)
    spread = max(float(np.max(np.ptp(sites, axis=0))), 20.0)
    a = (sites-center)/spread
    q = (coords-center)/spread
    mat = np.column_stack([np.ones(len(a)), a])
    target = z-np.median(z)
    ridge = np.diag([1e-6, 0.15, 0.15])
    params = np.linalg.solve(mat.T@mat+ridge, mat.T@target)
    t = np.median(z)+np.column_stack([np.ones(len(q)), q])@params
    dist = np.linalg.norm(coords[:, None, :]-sites[None, :, :], axis=2)
    nearest = dist.min(axis=1)
    residual = z-(np.median(z)+mat@params)
    weight = np.exp(-dist**2 / (2*max(spread*.65, 20.0)**2))
    corr = (weight@residual) / np.maximum(weight.sum(axis=1), 1e-12)
    return t+corr, nearest


def track_reflector_section(time, section, seed, win_ms=24, continuity=0.14):
    """Dynamic-programming horizon path. Penalizes vertical jumps and seed drift.

    Does not know faults, polarity or lithology; output is explicitly preliminary.
    """
    time = np.asarray(time, float)
    z = np.asarray(section, float)
    seed = np.asarray(seed, float)
    if z.ndim != 2 or z.shape[1] != len(seed): raise ValueError("Section/seed mismatch")
    if len(time) < 4: return seed.copy()
    dt = float(np.median(np.diff(time)))
    width = min(34, max(3, int(round(win_ms / dt))))
    offsets = np.arange(-width, width+1)
    k = len(offsets)
    centers = np.searchsorted(time, np.nan_to_num(seed, nan=time[0]))
    sample = centers[:, None] + offsets[None, :]
    valid = (sample >= 0)&(sample < len(time))&np.isfinite(seed[:, None])
    sample = np.clip(sample,0,len(time)-1)
    vals = z[sample, np.arange(z.shape[1])[:, None]].astype(float)
    vals[~valid] = np.nan
    # Median-valued noise scale per trace for robust likelihood.
    scale = np.nanpercentile(np.abs(vals), 90, axis=1, keepdims=True)
    scale = np.maximum(np.where(np.isfinite(scale), scale, 1), 1e-8)
    scores = np.nan_to_num(np.abs(vals)/scale, nan=-1e6, posinf=0, neginf=-1e6)
    scores -= 0.50*(offsets[None, :]/max(width,1))**2
    scores[~valid] = -1e6
    dp = np.zeros((z.shape[1], k), float)
    ptr = np.zeros((z.shape[1], k), np.int16)
    dp[0] = scores[0]
    jumps = (offsets[:, None]-offsets[None,:])**2
    for i in range(1,z.shape[1]):
        paths = dp[i-1][:,None]-continuity*jumps
        ptr[i] = np.argmax(paths, axis=0)
        dp[i] = scores[i]+paths[ptr[i], np.arange(k)]
    selected = np.zeros(z.shape[1], int)
    selected[-1] = int(np.argmax(dp[-1]))
    for i in range(len(selected)-2,-1,-1):
        selected[i] = ptr[i+1,selected[i+1]]
    out = time[sample[np.arange(len(selected)),selected]]
    out[~np.isfinite(seed)] = np.nan
    if len(out)>4 and np.isfinite(out).all():
        out = gaussian_filter1d(out,1.25)
    return out


def surface_preview(bpath, mpath, base_grid, mon_grid, matches, markers,
                    marker, stride=22, search_ms=22, max_anchor_distance=350,
                    confidence_limit=250, polarity=0):
    """Coverage-wide *preview*; both observed/predicted flags returned.

    Only evaluates actual matching IL/XL traces. Where no valid peak or no control,
    smooth trend is shown as low confidence, never as validated horizon.
    """
    source = markers[(markers.Marker==marker)&markers.TWT_ms.notna()]
    own = source[(source.Checkshot_Donor==source.Well) & ~source.Depth_Source.str.contains("assumed",na=False)]
    ctrl = source[["Well","TWT_ms"]]
    independent = set(own.Well)
    ctrl = ctrl[~ctrl.duplicated("Well",keep=False)]
    ctrl = ctrl.merge(matches[["Well","IL","XL","Distance_m"]],on="Well")
    ctrl = ctrl.loc[ctrl.Distance_m<=100]
    if len(ctrl)<2: return None
    bmin,bmax = int(base_grid.IL.min()), int(base_grid.IL.max())
    xmin,xmax = int(base_grid.XL.min()), int(base_grid.XL.max())
    ilvals = np.arange(bmin,bmax+1,stride)
    xlvals = np.arange(xmin,xmax+1,stride)
    IL,XL = np.meshgrid(ilvals,xlvals,indexing="ij")
    target = pd.DataFrame({"IL":IL.ravel(),"XL":XL.ravel()})
    sample = target.merge(base_grid[["IL","XL","idx","X","Y"]],on=["IL","XL"],how="left")
    sample = sample.merge(mon_grid[["IL","XL","idx"]].rename(columns={"idx":"mi"}),on=["IL","XL"],how="left")
    coord = sample[["IL","XL"]].to_numpy(float)
    seed, dist = trend_surface(coord, ctrl[["IL","XL"]].to_numpy(float), ctrl.TWT_ms.to_numpy(float))
    raw_b = np.full(len(sample),np.nan,float)
    raw_m = np.full(len(sample),np.nan,float)
    alternate = np.full(len(sample), np.nan)
    margin = np.full(len(sample), np.nan)
    independent_ctrl = ctrl[ctrl.Well.isin(independent)]
    _, independent_distance = trend_surface(coord, independent_ctrl[["IL","XL"]],
                                             independent_ctrl.TWT_ms)
    # Valid actual matched traces; missing survey cells remain NaN and transparent.
    has_data = sample.idx.notna().to_numpy() & sample.mi.notna().to_numpy()
    with segyio.open(str(bpath),"r",strict=False) as b, segyio.open(str(mpath),"r",strict=False) as m:
        tb=np.asarray(b.samples,float)
        tm=np.asarray(m.samples,float)
        if len(tb)!=len(tm) or not np.allclose(tb,tm,atol=.05):
            raise ValueError("Survey sampling mismatch")
        for k in np.flatnonzero(has_data):
            loc = np.flatnonzero(abs(tb-seed[k]) <= search_ms*1.5 + 2*np.median(np.diff(tb)))
            if len(loc)<5:
                continue
            t1 = np.asarray(b.trace[int(sample.idx.iloc[k])],float)[loc]
            t2 = np.asarray(m.trace[int(sample.mi.iloc[k])],float)[loc]
            t1[~sample_mask(t1)] = np.nan
            t2[~sample_mask(t2)] = np.nan
            for values, output in ((t1,raw_b),(t2,raw_m)):
                pick, separation, _, _ = reflector(tb[loc], values[:,None], [seed[k]], search_ms, polarity, 0)
                output[k] = pick[0]
                if output is raw_b:
                    margin[k] = separation[0]
            trial, _, _, _ = reflector(tb[loc],t1[:,None],[seed[k]],search_ms*1.5,polarity,0)
            alternate[k] = trial[0]
    shape=IL.shape
    data_mask=has_data.reshape(shape)
    confident=(has_data & (len(independent_ctrl)>=2) & (independent_distance<=confidence_limit)
               & np.isfinite(raw_b) & np.isfinite(raw_m) & (abs(raw_b-alternate)<=search_ms/2)
               & (~np.isfinite(margin) | (margin>=.05))).reshape(shape)
    trend=seed.reshape(shape)
    def smooth_pick(raw):
        arr=raw.reshape(shape)
        # Smooth dense reflector variations; never sawtooth maps.
        isfinite=np.isfinite(arr)&data_mask
        residual=np.where(isfinite,arr-trend,0.0)
        numerator=gaussian_filter(residual,sigma=(1.7,1.7))
        denominator=gaussian_filter(isfinite.astype(float),sigma=(1.7,1.7))
        fit=trend+np.clip(numerator/np.maximum(denominator,1e-8),-search_ms,search_ms)
        fit[~isfinite]=np.nan
        return fit
    tb=smooth_pick(raw_b)
    tm=smooth_pick(raw_m)
    scenarios = np.stack([tb, raw_b.reshape(shape), alternate.reshape(shape)])
    envelope_low = np.fmin(np.fmin(scenarios[0],scenarios[1]),scenarios[2])
    envelope_high = np.fmax(np.fmax(scenarios[0],scenarios[1]),scenarios[2])
    envelope_low[~np.isfinite(tb)] = np.nan
    envelope_high[~np.isfinite(tb)] = np.nan
    return {"IL":IL,"XL":XL,"base":tb,"monitor":tm,
            "envelope_low":envelope_low,"envelope_high":envelope_high,
            "sensitivity":envelope_high-envelope_low,"peak_margin":margin.reshape(shape),
            "trend":trend,"has_data":data_mask,"confident":confident,
            "distance_anchor":independent_distance.reshape(shape),"anchors":ctrl}


def window_stats(b,m,time,start,end):
    t=np.asarray(time)
    mask=(t>=start)&(t<=end)
    if mask.sum()<3:return None
    a=np.asarray(b)[mask].astype(float)
    z=np.asarray(m)[mask].astype(float)
    good=np.isfinite(a)&np.isfinite(z)
    if good.sum()<4:return None
    a,z=a[good],z[good]
    rb=np.sqrt(np.mean(a*a));rm=np.sqrt(np.mean(z*z))
    d=z-a
    return {
        "RMS_1995":float(rb),"RMS_2006":float(rm),
        "RMS_Diff":float(np.sqrt(np.mean(d*d))),
        "Signed_Mean_Diff":float(np.mean(d)),
        "NRMS_pct":float(200*np.sqrt(np.mean(d*d))/max(rb+rm,1e-8)),
        "Correlation":float(np.corrcoef(a,z)[0,1]) if np.std(a)>0 and np.std(z)>0 else np.nan,
        "Valid_Samples":int(len(a))
    }


def seismic_attribute_map(bpath,mpath,base_grid,mon_grid,matched,
                         window=(1500,1700),reference=(700,950),stride=20):
    """Map real 4D attributes on matched IL/XL grid. Missing cells = NaN."""
    ilmin,ilmax=int(base_grid.IL.min()),int(base_grid.IL.max())
    xlmin,xlmax=int(base_grid.XL.min()),int(base_grid.XL.max())
    il=np.arange(ilmin,ilmax+1,stride)
    xl=np.arange(xlmin,xlmax+1,stride)
    IL,XL=np.meshgrid(il,xl,indexing="ij")
    targets=pd.DataFrame({"IL":IL.ravel(),"XL":XL.ravel()})
    targets=targets.merge(base_grid[["IL","XL","idx"]].rename(columns={"idx":"bi"}),on=["IL","XL"],how="left")
    targets=targets.merge(mon_grid[["IL","XL","idx"]].rename(columns={"idx":"mi"}),on=["IL","XL"],how="left")
    good=targets.bi.notna()&targets.mi.notna()
    results={name:np.full(len(targets),np.nan) for name in
             ("NRMS_pct","RMS_Diff","Signed_Mean_Diff","RMS_1995","RMS_2006","Overburden_NRMS")}
    with segyio.open(str(bpath),"r",strict=False) as b, segyio.open(str(mpath),"r",strict=False) as m:
        t=np.asarray(b.samples,float)
        tm=np.asarray(m.samples,float)
        if len(t)!=len(tm) or not np.allclose(t,tm,atol=.05):raise ValueError("Sampling mismatch")
        for k in np.flatnonzero(good.to_numpy()):
            a=np.asarray(b.trace[int(targets.bi.iloc[k])],float)
            z=np.asarray(m.trace[int(targets.mi.iloc[k])],float)
            rs=window_stats(a,z,t,*window)
            bg=window_stats(a,z,t,*reference)
            if rs is None:continue
            for name in ("NRMS_pct","RMS_Diff","Signed_Mean_Diff","RMS_1995","RMS_2006"):
                results[name][k]=rs[name]
            if bg is not None:
                results["Overburden_NRMS"][k]=bg["NRMS_pct"]
    result={"IL":IL,"XL":XL}
    for name,val in results.items():result[name]=val.reshape(IL.shape)
    # Local 4D-change contrast relative to local smoothed overburden repeatability.
    over=result["Overburden_NRMS"]
    mask=np.isfinite(over)
    smooth=gaussian_filter(np.where(mask,over,0),sigma=1.5)/np.maximum(
        gaussian_filter(mask.astype(float),sigma=1.5),1e-6)
    with np.errstate(divide="ignore",invalid="ignore"):
        contrast=np.log(np.maximum(result["NRMS_pct"],1)/np.maximum(smooth,1))
    good=np.isfinite(contrast)
    if good.sum()>10:
        med=np.nanmedian(contrast[good]);mad=1.4826*np.nanmedian(np.abs(contrast[good]-med))
        result["z_local"]=(contrast-med)/max(mad,.08)
    else:result["z_local"]=np.full_like(contrast,np.nan)
    result["data_mask"]=np.isfinite(result["NRMS_pct"])
    return result


def fluid_screen(log):
    """Classify *interpreted* log model candidates; not observed fluid types."""
    n=len(log)
    labels=np.full(n,"Uncertain",dtype=object)
    def col(key):
        return pd.to_numeric(log[key],errors="coerce").to_numpy(float) if key in log else None
    ph, sw = col("PHIE"), col("SWE")
    if ph is None or sw is None:return labels
    clay=col("VSHALE")
    valid=np.isfinite(ph)&np.isfinite(sw)
    reservoir=valid&(ph>=.10)&(sw<=.60)
    if clay is not None:
        reservoir &= np.isfinite(clay)&(clay<=.40)
    labels[reservoir]="HC candidate"
    labels[valid&(ph>=.10)&(sw>=.80)]="Water model"
    oil=col("VOIL");gas=col("VGAS")
    # Some petrophysical exports use fractions, others percent; determine from values.
    def fractional(vals):
        if vals is None:return None
        positive=vals[np.isfinite(vals)]
        if len(positive)==0:return vals
        q=float(np.nanpercentile(positive,95))
        return vals/100.0 if 1.0<q<=100 else vals
    oil= fractional(oil);gas=fractional(gas)
    if oil is not None:
        labels[reservoir&np.isfinite(oil)&(oil>=.5)]="Oil-model"
    if gas is not None:
        labels[reservoir&np.isfinite(gas)&(gas>=.5)]="Gas-model"
    return labels
