"""Interval-local support from explicitly fractional source log curves."""
import numpy as np
import pandas as pd


def interval_support(log, rt, top, base, phie_min=.10, shale_max=.40,allow_gr_proxy=False):
    missing = {"quality": np.nan, "samples": 0, "cutoff_span": np.nan}
    if not np.isfinite([rt, top, base]).all() or top >= base:
        return missing
    if "PHIE" not in log:
        return missing
    depth = pd.to_numeric(log.iloc[:, 0], errors="coerce").to_numpy()-rt
    phie = pd.to_numeric(log.PHIE, errors="coerce").to_numpy()
    shale_source="provided_VSHALE"
    if "VSHALE" in log:
        shale = pd.to_numeric(log.VSHALE, errors="coerce").to_numpy()
    elif allow_gr_proxy and "GR_P" in log:
        gr = pd.to_numeric(log.GR_P,errors="coerce").to_numpy(float)
        clean = gr[np.isfinite(gr) & (gr>=0) & (gr<=250)]
        if len(clean)<50: return missing
        lo,hi=np.percentile(clean,[10,90])
        if hi-lo < 12: return missing
        shale=np.clip((gr-lo)/(hi-lo),0,1)
        shale_source="GR_quantile_index_UNCALIBRATED"
    else:
        return missing
    valid = ((depth >= top) & (depth <= base) & (phie >= 0) & (phie <= 1)
             & (shale >= 0) & (shale <= 1))
    if valid.sum() < 5:
        return missing
    order = np.argsort(depth[valid])
    z, p, s = depth[valid][order], phie[valid][order], shale[valid][order]
    step = np.diff(z)
    typical = np.median(step)
    if typical <= 0:
        return missing
    weight = np.r_[np.minimum(step, typical*2), typical]
    if weight.sum() < .6*(base-top):
        return missing
    scenarios = [np.average((p >= phie_min+dp) & (s <= shale_max+ds), weights=weight)
                 for dp, ds in [(0, 0), (-.02, .05), (.02, -.05)]]
    return {"quality": scenarios[0], "samples": int(valid.sum()),
            "cutoff_span": scenarios[1]-scenarios[2],"shale_source":shale_source}

