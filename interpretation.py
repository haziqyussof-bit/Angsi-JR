"""Checkshot validation, seeded reflector QC and synthetic tie primitives."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def convert(values, source, target):
    values = np.asarray(values, float)
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (len(source) < 2 or source.shape != target.shape or
            not np.isfinite(source + target).all() or
            np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0)):
        return np.full_like(values, np.nan)
    return np.interp(values, source, target, left=np.nan, right=np.nan)


def reflector(time, section, seed, radius, polarity=0, smooth=1, jump_ms=12):
    """Select a local extremum, then smooth within continuous supported segments."""
    time, section, seed = map(lambda x: np.asarray(x, float), (time, section, seed))
    pick = np.full(len(seed), np.nan)
    separation = np.full(len(seed), np.nan)
    for i, center in enumerate(seed):
        signal = section[:, i]
        strength = abs(signal) if polarity == 0 else polarity*signal
        peak = np.flatnonzero((strength[1:-1] >= strength[:-2]) &
                             (strength[1:-1] >= strength[2:]) &
                             ((strength[1:-1] > strength[:-2]) |
                              (strength[1:-1] > strength[2:])))+1
        peak = peak[(abs(time[peak]-center) <= radius) & (strength[peak] > 0)]
        if not len(peak):
            continue
        score = strength[peak]/max(strength[peak]) - .35*(abs(time[peak]-center)/radius)**2
        order = np.argsort(score)
        pick[i] = time[peak[order[-1]]]
        if len(order) > 1:
            separation[i] = score[order[-1]]-score[order[-2]]
    breaks = np.r_[True, (~np.isfinite(pick[1:]+pick[:-1])) |
                   (abs(np.diff(pick)) > jump_ms)]
    cuts = np.r_[np.flatnonzero(breaks), len(pick)]
    smoothed = pick.copy()
    if smooth > 0:
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi-lo > 2:
                smoothed[lo:hi] = gaussian_filter1d(pick[lo:hi], smooth)
    alternate = abs(pick-smoothed)
    flags = np.r_[False, np.isfinite(pick[1:]+pick[:-1]) & (abs(np.diff(pick)) > jump_ms)]
    return smoothed, separation, flags, alternate


def fence_seed(distance, stations, times):
    """Linear trend between independent measured anchors; no edge extrapolation."""
    valid = np.isfinite(stations) & np.isfinite(times)
    anchors = pd.DataFrame({"station": np.asarray(stations)[valid], "time": np.asarray(times)[valid]})
    anchors = anchors.groupby("station").time.median()
    if len(anchors) < 2:
        return np.full(len(distance), np.nan)
    return np.interp(distance, anchors.index, anchors, left=np.nan, right=np.nan)


def marker_qc(tops, checkshot_for):
    records = []
    for row in tops.itertuples():
        cs = checkshot_for(row.Well)
        z = convert([row.TWT_ms], cs.get("TWT_ms", []), cs.get("TVDSS_m", []))[0]
        residual = z-row.TVDSS_m
        vertical_difference = getattr(row, "Datum_Diff_m", np.nan)
        records.append({"Well": row.Well, "Marker": row.Marker,
                        "TVDSS_m": row.TVDSS_m, "TWT_ms": row.TWT_ms,
                        "Round_trip_residual_m": residual, "MDKB_minus_RT_minus_TVDSS_m": vertical_difference,
                        "Depth_source": row.Depth_Source, "Checkshot": row.Checkshot_Type,
                        "Status": "REVIEW" if not np.isfinite(residual) or abs(residual) > .1
                        or row.Checkshot_Type != "OWN" or "assumed" in row.Depth_Source
                        or (np.isfinite(vertical_difference) and abs(vertical_difference)>1) else "PASS"})
    return pd.DataFrame(records)


def synthetic(time, depth, slowness, density, cs_depth, cs_time, frequency_hz):
    """Normal-incidence convolution with an explicit zero-phase Ricker wavelet."""
    depth, slowness, density = map(lambda x: np.asarray(x, float), (depth, slowness, density))
    time = np.asarray(time, float)
    if (len(time) < 2 or len(depth) < 4 or frequency_hz <= 0 or
            not np.isfinite(depth+slowness+density).all() or
            np.any(np.diff(depth) <= 0) or np.any(slowness <= 0) or np.any(density <= 0)):
        raise ValueError("Synthetic requires continuous positive sonic/density and increasing depth")
    dt = (time[1]-time[0])/1000
    if dt <= 0 or not np.allclose(np.diff(time), dt*1000) or frequency_hz >= .5/dt:
        raise ValueError("Invalid sample interval or wavelet frequency")
    impedance = density/slowness
    twt = convert(depth, cs_depth, cs_time)
    supported = np.isfinite(twt)
    if supported.sum() < 4:
        raise ValueError("Insufficient own checkshot overlap")
    ai = np.interp(time, twt[supported], impedance[supported], left=np.nan, right=np.nan)
    coeff = np.diff(ai)/(ai[1:]+ai[:-1])
    radius = min(int(np.ceil(2/(frequency_hz*dt))), (len(time)-1)//2)
    t = np.arange(-radius, radius+1)*dt
    x = (np.pi*frequency_hz*t)**2
    wavelet = (1-2*x)*np.exp(-x)
    trace = np.convolve(np.r_[np.nan, coeff], wavelet, mode="same")
    trace[:radius] = np.nan
    if radius:
        trace[-radius:] = np.nan
    return trace

