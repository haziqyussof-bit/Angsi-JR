"""SEG-Y-backed survey sampling and bounded-memory attribute extraction."""
from pathlib import Path
import numpy as np
import pandas as pd
import segyio

from seismic_qc import analyze_trace, register, metrics, window

VERSION = "8.2.0"


def file_stamp(path):
    path = Path(path)
    info = path.stat()
    return str(path.resolve()), info.st_size, info.st_mtime_ns


def sample_pairs(baseline, monitor, stride):
    anchor_il, anchor_xl = baseline.IL.min(), baseline.XL.min()
    pairs = register(baseline, monitor)
    sampled = ((pairs.IL-anchor_il) % stride == 0) & ((pairs.XL-anchor_xl) % stride == 0)
    return pairs[sampled & (pairs["_merge"] == "both")].sort_values(["IL", "XL"])


def evaluate(bp, mp, pairs, params, top=None, base=None, return_qc=False):
    """Read each trace pair once. Optional horizons keyed by (IL, XL)."""
    records = []
    qc_records = []
    with segyio.open(str(bp), "r", strict=False) as b, segyio.open(str(mp), "r", strict=False) as m:
        time, mt = np.array(b.samples), np.array(m.samples)
        if time.shape != mt.shape or not np.allclose(time, mt, rtol=0, atol=.05):
            raise ValueError("Baseline and monitor sampling mismatch")
        for pair in pairs.itertuples(index=False):
            key = (pair.IL, pair.XL)
            options = params.copy()
            if top is not None:
                bounds = (top.get(key, np.nan), base.get(key, np.nan))
                if not np.isfinite(bounds).all():
                    qc_records.append(dict(IL=pair.IL, XL=pair.XL, reason="missing_or_unreliable_horizon", included=False))
                    continue
                if bounds[0] >= bounds[1]:
                    qc_records.append(dict(IL=pair.IL, XL=pair.XL, reason="crossing_or_zero_horizon_interval", included=False))
                    continue
                # A crossing with reference data invalidates this conformable cell.
                guard = (params["max_lag"] + (4 if params["spectral"] else 0))*float(time[1]-time[0])
                ref = params["reference"]
                if bounds[0] <= ref[1]+guard and bounds[1] >= ref[0]-guard:
                    qc_records.append(dict(IL=pair.IL, XL=pair.XL, reason="reference_overlap_or_guard", included=False))
                    continue
                options["reservoir"] = bounds
            row, a, z, corrected = analyze_trace(time, b.trace[int(pair.bi)], m.trace[int(pair.mi)], **options)
            lo, hi = options["reservoir"]
            dt = float(time[1]-time[0])
            perturbations = []
            amplitudes = []
            for offset in (-dt, dt):
                mask = window(time, (lo+offset, hi+offset))
                stat = metrics(a[mask], z[mask])
                perturbations.append(stat["nrms"])
                amplitudes.append(stat["rms_difference"])
            row.update(IL=pair.IL, XL=pair.XL, X=pair.X, Y=pair.Y,
                       bi=int(pair.bi), mi=int(pair.mi),
                       top_ms=lo, base_ms=hi,
                       nrms_window_span=np.ptp(perturbations),
                       rms_window_span=np.ptp(amplitudes),
                       version=VERSION)
            records.append(row)
            target_ok = row.get("raw_valid_samples", 0) >= 8 and row.get("raw_valid_fraction", 0) >= .80
            ref_fraction = row.get("reference_valid_fraction", 0)
            reference_ok = (ref_fraction >= .80 and np.isfinite(row.get("reference_nrms", np.nan))
                            and 0 < row["reference_nrms"] <= params["reference_limit"])
            contrast_ok = np.isfinite(row.get("local_repeatability_contrast", np.nan))
            if not target_ok:
                reason = "insufficient_valid_target_samples"
            elif not reference_ok:
                reason = "unreliable_or_missing_reference"
            elif not contrast_ok:
                reason = "undefined_local_repeatability_contrast"
            else:
                reason = "valid_for_screening"
            qc_records.append(dict(IL=pair.IL, XL=pair.XL, reason=reason,
                                   included=reason == "valid_for_screening",
                                   valid_target=bool(target_ok), valid_reference=bool(reference_ok)))
    frame = pd.DataFrame(records)
    if return_qc:
        return frame, pd.DataFrame(qc_records, columns=["IL", "XL", "reason", "included", "valid_target", "valid_reference"])
    return frame

