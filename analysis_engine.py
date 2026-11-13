"""V9 reservoir analysis: auditable trace attributes, sparse well-conditioned facies,
connected candidate polygons and reproducible ranking sensitivities.

All amplitude-based outputs are indicators; no rock-fluid classification or
production attribution is inferred by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import pandas as pd
from scipy.ndimage import label as label_components
from scipy.signal import hilbert
from scipy.spatial import cKDTree
from scipy.special import expit
from shapely.geometry import box
from shapely.ops import unary_union

ATTRIBUTE_CATALOG = {
    "RMS": ("RMS amplitude", "amplitude", "sqrt(mean(amplitude²))"),
    "Mean": ("Mean signed amplitude", "amplitude", "mean(amplitude)"),
    "MeanAbs": ("Mean absolute amplitude", "amplitude", "mean(abs(amplitude))"),
    "MaxPositive": ("Maximum positive amplitude", "amplitude", "maximum positive value"),
    "MostNegative": ("Most negative amplitude", "amplitude", "minimum amplitude value"),
    "AbsolutePeak": ("Absolute peak amplitude", "amplitude", "max(abs(amplitude))"),
    "Energy": ("Window energy", "amplitude²", "sum(amplitude²); depends on window sample count"),
    "Variance": ("Amplitude variance", "amplitude²", "population variance"),
    "EnvelopeMean": ("Mean instantaneous amplitude", "amplitude", "mean(abs(analytic trace))"),
    "EnvelopeMax": ("Maximum instantaneous amplitude", "amplitude", "max(abs(analytic trace))"),
    "PhaseMean": ("Circular mean instantaneous phase", "degrees", "circular mean phase (not scalar amplitude)"),
    "DominantHz": ("Dominant frequency", "Hz", "peak amplitude spectral bin in the window"),
    "Spectral10": ("Spectral amplitude ~10 Hz", "amplitude", "nearest FFT bin, tapered and normalized"),
    "Spectral25": ("Spectral amplitude ~25 Hz", "amplitude", "nearest FFT bin, tapered and normalized"),
    "Spectral50": ("Spectral amplitude ~50 Hz", "amplitude", "nearest FFT bin, tapered and normalized"),
    "Skewness": ("Amplitude skewness", "unitless", "standardized third moment"),
    "Kurtosis": ("Excess kurtosis", "unitless", "standardized fourth moment minus three"),
}
ATTR_KEYS = tuple(ATTRIBUTE_CATALOG)
STATIC_FEATURES = ("B_RMS", "B_EnvelopeMean", "B_MostNegative")


def _longest_finite_segment(values):
    good = np.isfinite(values)
    if not good.any():
        return np.empty(0, dtype=float)
    boundaries = np.diff(np.r_[False, good, False].astype(int))
    starts = np.where(boundaries == 1)[0]
    ends = np.where(boundaries == -1)[0]
    j = int(np.argmax(ends - starts))
    return np.asarray(values[starts[j]:ends[j]], float)


def trace_attributes(values, dt_ms: float, min_samples=8):
    """Return raw trace attributes without filling missing waveform samples.

    Hilbert/FFT attributes are only computed from the longest contiguous
    finite segment (>=16 samples); no bridging through acquisition mutes.
    """
    out = {name: np.nan for name in ATTR_KEYS}
    x = np.asarray(values, float).ravel()
    v = x[np.isfinite(x)]
    if len(v) < min_samples or not np.isfinite(dt_ms) or dt_ms <= 0:
        return out
    mean = float(v.mean())
    rms = float(np.sqrt(np.mean(v*v)))
    sd = float(v.std())
    out.update(RMS=rms, Mean=mean, MeanAbs=float(np.mean(abs(v))),
               MaxPositive=float(np.max(v)), MostNegative=float(np.min(v)),
               AbsolutePeak=float(np.max(abs(v))), Energy=float(np.sum(v*v)),
               Variance=float(sd*sd))
    if sd > 1e-12:
        normalized = (v-mean)/sd
        out["Skewness"] = float(np.mean(normalized**3))
        out["Kurtosis"] = float(np.mean(normalized**4)-3)
    run = _longest_finite_segment(x)
    if len(run) < 16:
        return out
    analytic = hilbert(run)
    envelope = abs(analytic)
    out["EnvelopeMean"] = float(np.mean(envelope))
    out["EnvelopeMax"] = float(np.max(envelope))
    phase = np.angle(analytic)
    phasor = np.mean(np.exp(1j*phase))
    if abs(phasor) > 1e-8:
        out["PhaseMean"] = float(np.degrees(np.angle(phasor)))
    hann = np.hanning(len(run))
    spec = np.abs(np.fft.rfft((run-run.mean())*hann))
    hz = np.fft.rfftfreq(len(run), d=dt_ms/1000.0)
    scale = max(hann.sum()/2.0, 1e-12)
    if len(spec) > 1:
        peak_idx = int(np.argmax(spec[1:])+1)
        out["DominantHz"] = float(hz[peak_idx])
        for freq in (10, 25, 50):
            ix = int(np.argmin(abs(hz-freq)))
            out[f"Spectral{freq}"] = float(spec[ix]/scale)
    return out


def _valid_fraction(a, b, mask):
    return float(np.count_nonzero(mask & np.isfinite(a) & np.isfinite(b)) / max(1, int(mask.sum())))


def extraction_qc(a, b, t, bounds, reference, zero_run=8):
    """Conservative paired-data QC, independent of any cross-equalization model."""
    from seismic_qc import sample_mask
    x = np.where(sample_mask(a, zero_run), a, np.nan)
    y = np.where(sample_mask(b, zero_run), b, np.nan)
    target = (t >= bounds[0]) & (t <= bounds[1])
    ref = (t >= reference[0]) & (t <= reference[1])
    if not target.any():
        return x, y, target, ref, "outside_sample_range", 0.0, 0.0
    frac = _valid_fraction(x, y, target)
    ref_frac = _valid_fraction(x, y, ref)
    reason = "valid"
    if target.sum() < 8 or frac < .80:
        reason = "insufficient_valid_target"
    elif ref.sum() < 8 or ref_frac < .80:
        reason = "insufficient_valid_reference"
    elif max(bounds[0], reference[0]) <= min(bounds[1], reference[1]):
        reason = "reference_target_overlap"
    return x, y, target, ref, reason, frac, ref_frac


def compute_attribute_volume(bp, mp, pairs, *, bounds=(1500., 1700.),
                             top_map: Mapping | None = None,
                             base_map: Mapping | None = None,
                             reference=(700., 950.), zero_run=8,
                             max_lag=4, reference_limit=50., spectral=False,
                             mute_before_ms=None, use_correction=True):
    """Read sampled paired SEG-Y traces once and return a long table + QC.

    top/base keyed by (IL, XL) for horizon-conformable extraction.
    Amplitude attributes use same common live samples in BOTH vintages.
    """
    import segyio
    from seismic_qc import metrics, analyze_trace
    if bounds[0] >= bounds[1] or reference[0] >= reference[1]:
        raise ValueError("Both target and reference require ordered intervals")
    if top_map is None and base_map is not None or top_map is not None and base_map is None:
        raise ValueError("Horizon extraction requires BOTH top and base maps")
    rows, qc = [], []
    if pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    with segyio.open(str(bp), "r", strict=False) as bf, segyio.open(str(mp), "r", strict=False) as mf:
        time = np.asarray(bf.samples, float)
        other = np.asarray(mf.samples, float)
        if len(time) < 2 or time.shape != other.shape or not np.allclose(time, other, atol=.05):
            raise ValueError("SEG-Y vintages have incompatible time sampling")
        dt = float(time[1]-time[0])
        if not np.allclose(np.diff(time), dt):
            raise ValueError("SEG-Y TWT samples are nonuniform")
        for r in pairs.itertuples(index=False):
            il, xl = int(r.IL), int(r.XL)
            key = (il, xl)
            interval = (float(top_map.get(key, np.nan)), float(base_map.get(key, np.nan))) if top_map is not None else bounds
            qrow = dict(IL=il, XL=xl, included=False, reason="", top_ms=interval[0], base_ms=interval[1])
            if not np.isfinite(interval).all():
                qrow["reason"] = "missing_or_unreliable_horizons"
                qc.append(qrow)
                continue
            if interval[0] >= interval[1]:
                qrow["reason"] = "crossing_or_zero_thickness_horizons"
                qc.append(qrow)
                continue
            if max(interval[0], reference[0]) <= min(interval[1], reference[1]):
                qrow["reason"] = "reference_target_overlap"
                qc.append(qrow)
                continue
            aa = np.asarray(bf.trace[int(r.bi)], dtype=float)
            mm = np.asarray(mf.trace[int(r.mi)], dtype=float)
            if mute_before_ms is not None:
                live = time >= float(mute_before_ms)
                aa, mm = np.where(live, aa, np.nan), np.where(live, mm, np.nan)
            aa, mm, target, ref, reason, tf, rf = extraction_qc(
                aa, mm, time, interval, reference, zero_run=zero_run)
            qrow.update(reason=reason, valid_target_fraction=tf, valid_reference_fraction=rf)
            # Raw amplitude attributes need a valid PAIRED TARGET window, not
            # a successful non-reservoir reference. Preserve those raw cells
            # across the complete common seismic footprint. Reference failures
            # are separately flagged and forbidden as validated 4D evidence.
            target_ok = bool(target.sum() >= 8 and tf >= .80)
            reference_ok = bool(reason == "valid")
            if not target_ok or reason == "outside_sample_range":
                qc.append(qrow)
                continue
            qrow["raw_attribute_valid"] = True
            qrow["screening_eligible"] = reference_ok
            if not reference_ok:
                qrow["reason"] = "raw_only__" + reason
            common = target & np.isfinite(aa) & np.isfinite(mm)
            xs, ys = np.where(common, aa, np.nan), np.where(common, mm, np.nan)
            data = dict(IL=il, XL=xl, X=float(r.X), Y=float(r.Y),
                        Top_ms=interval[0], Base_ms=interval[1],
                        Target_valid_pct=100*tf, Reference_valid_pct=100*rf)
            for tag, values in (("B", xs[target]), ("M", ys[target]), ("D", (ys-xs)[target])):
                data.update({f"{tag}_{k}": v for k, v in trace_attributes(values, dt).items()})
            rawstats = metrics(aa[target], mm[target])
            refstats = metrics(aa[ref], mm[ref])
            data["Raw_NRMS"] = rawstats["nrms"]
            data["Raw_Predictability"] = rawstats["predictability"]
            data["Reference_NRMS"] = refstats["nrms"]
            data["Local_contrast"] = (rawstats["nrms"]/refstats["nrms"]
                                      if reference_ok and np.isfinite(refstats["nrms"])
                                      and refstats["nrms"] > 1e-8 else np.nan)
            data["QC_screening_pass"] = bool(reference_ok and np.isfinite(refstats["nrms"])
                                                  and 0 < refstats["nrms"] <= reference_limit)
            data["QC_reference_status"] = "PASS" if reference_ok else "RAW_ONLY"
            data["Correction_status"] = "NOT_REQUESTED"
            data["Corrected_NRMS"] = np.nan
            if use_correction and reference_ok:
                try:
                    rec, masked_b, masked_m, corrected = analyze_trace(
                        time, np.where(np.isfinite(aa), aa, np.nan),
                        np.where(np.isfinite(mm), mm, np.nan),
                        reference=reference, reservoir=interval,
                        max_lag=max_lag, zero_run=0, reference_limit=reference_limit,
                        spectral=spectral, mute_before_ms=mute_before_ms)
                    data["Correction_status"] = str(rec.get("status", "REVIEW"))
                    data["Corrected_NRMS"] = float(rec.get("corrected_reservoir_nrms", np.nan))
                    if data["Correction_status"] == "PASS":
                        valid = target & np.isfinite(masked_b) & np.isfinite(corrected)
                        cvalues = np.where(valid, corrected-masked_b, np.nan)[target]
                        data.update({f"C_{k}": v for k, v in trace_attributes(cvalues, dt).items()})
                except ValueError as exc:
                    data["Correction_status"] = "REVIEW"
                    data["Correction_error"] = str(exc)
            for side in (-dt, dt):
                mask = (time >= interval[0]+side) & (time <= interval[1]+side)
                data[f"Perturbed_RMS_{'minus' if side<0 else 'plus'}"] = metrics(aa[mask], mm[mask])["rms_difference"]
            perturbed = [data.get("Perturbed_RMS_minus"), data.get("Perturbed_RMS_plus")]
            data["Window_RMS_span"] = float(np.ptp(perturbed)) if np.isfinite(perturbed).all() else np.nan
            qrow.update(included=True, reason="valid_for_attributes" if reference_ok else qrow["reason"],
                        screening_eligible=bool(data["QC_screening_pass"]),
                        correction=data["Correction_status"])
            rows.append(data)
            qc.append(qrow)
    return pd.DataFrame(rows), pd.DataFrame(qc)


def choose_well_interval(well, mode, *, tops, time_window, depth_window, cs_loader):
    """Return TVDSS bounds and source; absent measured control -> no training label."""
    if mode == "Horizon pair":
        a, b = time_window  # marker name pair in this mode
        subset = tops[tops.Well == well]
        va = subset[subset.Marker == a]
        vb = subset[subset.Marker == b]
        if len(va) != 1 or len(vb) != 1:
            return None, "missing_well_marker"
        if any("assumed" in str(s) for s in [va.Depth_Source.iloc[0], vb.Depth_Source.iloc[0]]):
            return None, "assumed_vertical_marker"
        z = (float(va.TVDSS_m.iloc[0]), float(vb.TVDSS_m.iloc[0]))
    elif mode == "Depth (TVDSS)":
        z = depth_window
    else:
        cs = cs_loader(well)
        if len(cs) < 2:
            return None, "no_own_checkshot"
        import angsi_core as C
        z = tuple(C.time_to_depth(np.asarray(time_window, float), cs))
    if not np.isfinite(z).all() or z[0] >= z[1]:
        return None, "unvalidated_or_reversed_zone"
    return z, "measured_control" if mode == "Horizon pair" else "conditional_window"


def well_labels(frame, wells, matched, tops, *, wd, mode, time_window, depth_window, cs_loader, log_loader,
                phie_min=.10, vshale_max=.40, allow_gr_proxy=True):
    """One independent training label per well, never per LAS depth sample."""
    from petrophysics import interval_support
    match = matched.set_index("Well")
    results = []
    for w in wells.itertuples(index=False):
        if w.Well not in match.index:
            continue
        location = match.loc[w.Well]
        d = float(location.Distance_m)
        if not np.isfinite(d) or d > 100:
            results.append(dict(Well=w.Well, Label="EXCLUDED", Reason="outside_seismic_coverage", Distance_m=d))
            continue
        depth, reason = choose_well_interval(w.Well, mode, tops=tops,
                                             time_window=time_window, depth_window=depth_window,
                                             cs_loader=cs_loader)
        if depth is None:
            results.append(dict(Well=w.Well, Label="EXCLUDED", Reason=reason, Distance_m=d))
            continue
        try:
            supp = interval_support(log_loader(w.Well), w.RT, *depth, phie_min, vshale_max,allow_gr_proxy=allow_gr_proxy)
        except (ValueError, OSError, KeyError, TypeError):
            supp = {"quality":np.nan, "samples":0}
        quality = float(supp.get("quality", np.nan))
        if not np.isfinite(quality):
            results.append(dict(Well=w.Well, Label="EXCLUDED", Reason="insufficient_PHIE_VSHALE", Distance_m=d))
            continue
        distances = np.hypot(frame.IL.to_numpy()-float(location.IL), frame.XL.to_numpy()-float(location.XL))
        if not len(distances):
            continue
        idx = int(np.argmin(distances))
        row = frame.iloc[idx]
        if distances[idx] > 22:
            results.append(dict(Well=w.Well, Label="EXCLUDED", Reason="no_nearby_sampled_cell", Distance_m=d))
            continue
        status = "RESERVOIR" if quality >= .55 else "NON_RESERVOIR" if quality <= .20 else "AMBIGUOUS"
        results.append(dict(Well=w.Well, Label=status, NetReservoirFraction=quality,
                            Samples=supp.get("samples",0), Shale_source=supp.get("shale_source","UNKNOWN"),
                            Samples_QC=supp.get("cutoff_span",np.nan),IL=int(row.IL), XL=int(row.XL),
                            Distance_m=d, Reason="well_phie_vshale_screen"))
    return pd.DataFrame(results)


def _fit_centroids(values, labels):
    values = np.asarray(values, float)
    med = np.nanmedian(values, axis=0)
    med[~np.isfinite(med)] = 0.
    x = np.where(np.isfinite(values), values, med)
    sd = np.maximum(np.std(x, axis=0), 1e-9)
    zx = np.clip((x-med)/sd, -5, 5)
    p = zx[labels == 1].mean(axis=0)
    n = zx[labels == 0].mean(axis=0)
    direction = p-n
    denom = max(float(np.dot(direction, direction)), .10)
    return med, sd, direction, float(np.dot((p+n)/2, direction)), denom


def _score(model, values):
    med, sd, direction, midpoint, denom = model
    values = np.asarray(values, float)
    filled = np.where(np.isfinite(values), values, med)
    zx = np.clip((filled-med)/sd, -5, 5)
    raw = (zx @ direction-midpoint)/np.sqrt(denom)
    return expit(np.clip(raw, -8, 8))


def reservoir_screen(frame, labels, min_wells=4):
    """Conservative well-conditioned seismic facies similarity, NOT P(sand).

    Leave-one-WELL-out cross-validation is required before displaying reservoir
    and nonreservoir proxy classes; otherwise status INSufficient.
    """
    out = frame.copy()
    out["Reservoir_similarity_score"] = np.nan
    out["Reservoir_proxy_class"] = "Unclassified"
    if labels is None or labels.empty:
        return out, dict(status="INSUFFICIENT", reason="No usable measured well labels", wells=0)
    fit_rows = labels[labels.Label.isin(["RESERVOIR", "NON_RESERVOIR"])].copy()
    fit_rows = fit_rows.drop_duplicates("Well")
    if len(fit_rows) < min_wells or fit_rows.Label.value_counts().min() < 2 or fit_rows.Label.nunique() != 2:
        return out, dict(status="INSUFFICIENT", reason="Need ≥2 distinct wells of each facies class", wells=len(fit_rows))
    grid = out.set_index(["IL","XL"])
    available = []
    for row in fit_rows.itertuples(index=False):
        if (row.IL,row.XL) in grid.index:
            r = grid.loc[(row.IL,row.XL)]
            if isinstance(r, pd.DataFrame):r = r.iloc[0]
            available.append((row.Well, row.Label, r[list(STATIC_FEATURES)].to_numpy(float)))
    if len(available) < min_wells:
        return out, dict(status="INSUFFICIENT", reason="Too few well-matched attribute samples", wells=len(available))
    x = np.array([r[2] for r in available], float)
    y = np.array([int(r[1] == "RESERVOIR") for r in available])
    if min(np.bincount(y)) < 2:
        return out, dict(status="INSUFFICIENT", reason="Need both facies supported by ≥2 wells", wells=len(y))
    votes=[]
    for j in range(len(y)):
        keep = np.arange(len(y)) != j
        if len(np.unique(y[keep])) < 2:
            return out, dict(status="INSUFFICIENT", reason="Insufficient control for leave-one-well-out validation", wells=len(y))
        model = _fit_centroids(x[keep], y[keep])
        votes.append(float(_score(model, x[j:j+1])[0]))
    pred = np.array(votes) >= .5
    bacc = .5*(float(np.mean(pred[y==0]==0))+float(np.mean(pred[y==1]==1)))
    # A weak/inverted relationship cannot be relabelled as lithology.
    if bacc < .60:
        return out, dict(status="NOT_VALIDATED", reason=f"Leave-one-well-out balanced accuracy {bacc:.2f} < 0.60", wells=len(y), loo_balanced_accuracy=bacc)
    model = _fit_centroids(x, y)
    out["Reservoir_similarity_score"] = _score(model, out[list(STATIC_FEATURES)].to_numpy(float))
    out["Reservoir_proxy_class"] = np.select(
        [out.Reservoir_similarity_score>=.65, out.Reservoir_similarity_score<=.35],
        ["Reservoir-favourable proxy", "Non-reservoir-favourable proxy"],
        default="Ambiguous")
    return out, dict(status="EXPLORATORY", reason="Well-conditioned facies similarity; not calibrated probability", wells=len(y), loo_balanced_accuracy=bacc)


# V9.3.1 | Independent amplitude-based seismic facies HYPOTHESIS.
# This mode deliberately avoids well-training prerequisites, does not confuse
# bright amplitude with confirmed sand, and operates on all target-valid paired cells.
AMPLITUDE_FACIES_CODES = {
    "DIM_SHALE_PRONE": 0.0,
    "INTERMEDIATE_UNCERTAIN": 1.0,
    "BRIGHT_SAND_PRONE": 2.0,
}


def amplitude_facies_proxy(frame, *, low_percentile=30., high_percentile=70.,
                           attribute="B_RMS"):
    """Partition valid *baseline* seismic amplitude into exploratory facies bins.

    This is a FULL sampled survey classification: no spatial nearest-well buffer
    and no requirement for well lithology labels. It does NOT estimate sand or
    shale probability. QC-screening failures remain visible as annotations,
    not silently interpreted as validated facies observations.

    Returns (copy_of_frame, provenance/report). B_ fields use 1995 only to
    avoid time-lapse leakage into the lithology screening hypothesis.
    """
    if not (0 < low_percentile < high_percentile < 100):
        raise ValueError("Specify ordered percentiles within 0–100")
    if attribute not in ("B_RMS", "B_EnvelopeMean", "B_AbsolutePeak"):
        raise ValueError("Only 1995 baseline amplitude-strength attributes are allowed")
    out = frame.copy()
    out["AmplitudeFaciesClass"] = "UNCLASSIFIED"
    out["AmplitudeFaciesCode"] = np.nan
    out["AmplitudeFaciesAttribute"] = attribute
    out["AmplitudeFaciesQC"] = "TARGET_INVALID"
    report = {
        "status": "INSUFFICIENT", "method": "Amplitude rank proxy (unvalidated)",
        "attribute": attribute, "low_percentile": float(low_percentile),
        "high_percentile": float(high_percentile),
        "no_training_wells_required": True,
        "geological_validation": "NOT VALIDATED",
        "interpretation": "Bright=sand-prone / dim=shale-prone HYPOTHESIS only",
        "paired_attribute_cells": int(len(out)),
    }
    if out.empty or attribute not in out:
        report.update(reason="No computed 1995 amplitude attribute cells")
        return out, report
    vals = pd.to_numeric(out[attribute], errors="coerce").to_numpy(float)
    valid = np.isfinite(vals)
    n = int(valid.sum())
    report["valid_amplitude_cells"] = n
    report["excluded_amplitude_cells"] = int(len(out)-n)
    if n < 6:
        report.update(reason="Need at least six valid baseline amplitude cells to define percentiles")
        return out, report
    lo, hi = np.percentile(vals[valid], [low_percentile, high_percentile])
    report.update(low_cutoff=float(lo), high_cutoff=float(hi))
    # Constant/near-constant seismic strength carries no discrimination.
    if not np.isfinite([lo, hi]).all() or (hi-lo) <= max(1e-12, abs(float(hi))*1e-9):
        report.update(reason="No stable amplitude contrast; percentile cutoffs overlap")
        out.loc[valid, "AmplitudeFaciesClass"] = "INTERMEDIATE_UNCERTAIN"
        out.loc[valid, "AmplitudeFaciesCode"] = 1.0
        return out, report
    dim = valid & (vals <= lo)
    bright = valid & (vals >= hi)
    middle = valid & ~dim & ~bright
    out.loc[dim, "AmplitudeFaciesClass"] = "DIM_SHALE_PRONE"
    out.loc[bright, "AmplitudeFaciesClass"] = "BRIGHT_SAND_PRONE"
    out.loc[middle, "AmplitudeFaciesClass"] = "INTERMEDIATE_UNCERTAIN"
    for name, code in AMPLITUDE_FACIES_CODES.items():
        out.loc[out["AmplitudeFaciesClass"] == name, "AmplitudeFaciesCode"] = code
    if "QC_screening_pass" in out:
        approved = out["QC_screening_pass"].fillna(False).to_numpy(dtype=bool)
        out.loc[valid & ~approved, "AmplitudeFaciesQC"] = "RAW_ONLY_REFERENCE_QC_FAILED"
        out.loc[valid & approved, "AmplitudeFaciesQC"] = "REFERENCE_SUPPORTED"
    else:
        out.loc[valid, "AmplitudeFaciesQC"] = "REFERENCE_QC_UNAVAILABLE"
    if "Horizon_support" in out:
        model = valid & out["Horizon_support"].astype(str).eq("MODEL_ONLY").to_numpy()
        out.loc[model, "AmplitudeFaciesQC"] = "MODEL_ONLY_HORIZON_UNVALIDATED"
    report.update(
        status="EXPLORATORY_UNVALIDATED", reason="Percentile classification of 1995 amplitude strength",
        dim_cells=int(dim.sum()), intermediate_cells=int(middle.sum()),
        bright_cells=int(bright.sum()),
        raw_only_cells=int(out["AmplitudeFaciesQC"].eq("RAW_ONLY_REFERENCE_QC_FAILED").sum()),
        model_only_cells=int(out["AmplitudeFaciesQC"].eq("MODEL_ONLY_HORIZON_UNVALIDATED").sum()),
        reference_supported_cells=int(out["AmplitudeFaciesQC"].eq("REFERENCE_SUPPORTED").sum()),
    )
    return out, report


def candidate_regions(frame, stride, *, low=.70, high=1.60, min_cells=5,
                      require_reservoir=False, metric="Local_contrast", facies_filter=None):
    """Grid-faithful polygonization of full-survey sampled cells.

    metric=Local_contrast uses reference-supported 4D contrast.
    metric=Raw_RMS_percentile uses ALL target-valid paired cells but ranks
    raw amplitudes only as UNVALIDATED screening, never depletion evidence.
    """
    if frame.empty:
        return []
    f = frame.copy()
    il0, xl0 = f.IL.min(), f.XL.min()
    # Use physical grid step to preserve holes and gaps, not dense rank indices.
    u = np.round((f.IL.to_numpy()-il0)/stride).astype(int)
    v = np.round((f.XL.to_numpy()-xl0)/stride).astype(int)
    shape=(int(u.max())+1,int(v.max())+1)
    if shape[0]*shape[1] > 3_000_000:
        raise ValueError("Grid too large for pixel polygonization; increase sample spacing")
    regions=[]
    if metric == "Local_contrast":
        z = f.Local_contrast.to_numpy(float)
        base = np.isfinite(z) & np.isfinite(f.Reference_NRMS.to_numpy(float))
        category_names = ("Higher 4D response", "Low 4D response")
    elif metric == "Raw_RMS_percentile":
        z = f.D_RMS.rank(pct=True).to_numpy(float)
        base = np.isfinite(z)
        category_names = ("Higher raw 4D RMS response", "Low raw 4D RMS response")
    else:
        raise ValueError("Unrecognized opportunity detection metric")
    # Strict mode requires the reference-window gate, not just finite NRMS.
    if metric == "Local_contrast" and "QC_screening_pass" in f:
        base &= f.QC_screening_pass.fillna(False).to_numpy(dtype=bool)
    if require_reservoir:
        if "Reservoir_proxy_class" not in f.columns:
            raise ValueError("No validated well-conditioned reservoir screening available")
        base &= f.Reservoir_proxy_class.to_numpy() == "Reservoir-favourable proxy"
    if facies_filter is not None:
        if facies_filter not in AMPLITUDE_FACIES_CODES:
            raise ValueError("Unknown amplitude facies proxy class")
        if "AmplitudeFaciesClass" not in f.columns:
            raise ValueError("Calculate amplitude facies in Analysis Workflow 2 first")
        base &= f.AmplitudeFaciesClass.astype(str).to_numpy() == facies_filter
    for kind, condition in ((category_names[0], z>=high), (category_names[1], z<=low)):
        mask=np.zeros(shape,dtype=bool)
        mask[u[base&condition],v[base&condition]] = True
        components,n=label_components(mask,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        for lab in range(1,n+1):
            indices=np.where(components[u,v] == lab)[0]
            if len(indices) < min_cells:
                continue
            cells=f.iloc[indices]
            polygons=[box(float(r.IL)-stride/2,float(r.XL)-stride/2,
                          float(r.IL)+stride/2,float(r.XL)+stride/2)
                      for r in cells.itertuples(index=False)]
            geom=unary_union(polygons)
            regions.append({"ID":f"{kind[0]}{len(regions)+1:02}",
                            "Category":kind,"Cells":len(cells),
                            "Geometry":geom, "Keys":list(zip(cells.IL.astype(int),cells.XL.astype(int))),
                            "Detection_metric":metric,
                            "Mean_detection_score":float(np.nanmean(z[indices])),
                            "Mean_contrast":float(cells.Local_contrast.mean()) if cells.Local_contrast.notna().any() else np.nan,
                            "Median_reference_NRMS":float(cells.Reference_NRMS.median()) if cells.Reference_NRMS.notna().any() else np.nan,
                            "Mean_reservoir_score":float(cells.Reservoir_similarity_score.mean()) if "Reservoir_similarity_score" in cells and cells.Reservoir_similarity_score.notna().any() else np.nan,
                            "Mean_window_sensitivity":float(cells.Window_RMS_span.mean()) if "Window_RMS_span" in cells and cells.Window_RMS_span.notna().any() else np.nan,
                            "Bright_amplitude_proxy_fraction":float(cells.AmplitudeFaciesClass.eq("BRIGHT_SAND_PRONE").mean()) if "AmplitudeFaciesClass" in cells else np.nan,
                            "Facies_evidence_status":"UNVALIDATED amplitude hypothesis" if "AmplitudeFaciesClass" in cells else "No amplitude proxy"})
    return sorted(regions,key=lambda r:r["Cells"],reverse=True)


def rank_candidates(regions, frame, wells=None, weights=(30,35,20,15), scenarios=300, seed=2026):
    """Rank relative evidence with uncertainty over *weights*, not oil probability."""
    names=("Reservoir", "4D_Contrast", "Repeatability", "Well_Control")
    weights = np.asarray(weights,float)
    if weights.shape != (4,) or (weights<0).any() or not weights.sum():
        raise ValueError("Four nonnegative ranking weights with positive sum required")
    base = frame.set_index(["IL","XL"])
    data=[]
    for region in regions:
        cells=base.loc[region["Keys"]]
        if isinstance(cells,pd.Series): cells=cells.to_frame().T
        vals=np.asarray(cells.Local_contrast, float)
        if region.get("Detection_metric") == "Raw_RMS_percentile":
            # Raw amplitudes are only a relative map contrast, not depletion.
            raw=float(region["Mean_detection_score"])
            contrast=float(np.clip(1-raw if region["Category"].startswith("Low") else raw,0,1))
        elif not np.isfinite(vals).any():
            contrast=np.nan
        elif region["Category"].startswith("Low"):
            contrast=float(np.clip(1-np.nanmedian(vals)/1.6,0,1))
        else:
            contrast=float(np.clip(np.nanmedian(vals)/3.0,0,1))
        ref=float(np.nanmedian(cells.Reference_NRMS)) if np.isfinite(cells.Reference_NRMS.to_numpy(float)).any() else np.nan
        qc=float(np.clip(1-ref/100,0,1)) if np.isfinite(ref) else np.nan
        reservoir=region.get("Mean_reservoir_score",np.nan)
        if not np.isfinite(reservoir): reservoir=np.nan
        nearest=np.nan
        if wells is not None and {"X","Y"}.issubset(cells) and {"X","Y"}.issubset(wells):
            xy=cells[["X","Y"]].to_numpy(float)
            ww=wells[["X","Y"]].to_numpy(float)
            xy=xy[np.isfinite(xy).all(axis=1)]
            ww=ww[np.isfinite(ww).all(axis=1)]
            if len(xy) and len(ww):
                nearest=float(cKDTree(xy).query(ww)[0].min())
        wellcontrol=float(np.exp(-nearest/1000)) if np.isfinite(nearest) else np.nan
        data.append({"ID":region["ID"], "Category":region["Category"], "Cells":region["Cells"],
                     "Reservoir":reservoir, "4D_Contrast":contrast,
                     "Repeatability":qc, "Well_Control":wellcontrol,
                     "Closest_well_m":nearest,"Mean_4D_contrast":region["Mean_contrast"],
                     "Production_status":"Not supplied", "Hydrocarbon_confirmation":"Not established",
                     "Detection_method":region.get("Detection_metric","Local_contrast"),
                     "Bright_amplitude_proxy_fraction":region.get("Bright_amplitude_proxy_fraction",np.nan),
                     "Facies_evidence_status":region.get("Facies_evidence_status","No amplitude proxy")})
    table=pd.DataFrame(data)
    if table.empty:return table
    scores=table[list(names)].to_numpy(float)
    have=np.isfinite(scores)
    weight_coverage=have @ weights / weights.sum()
    score=np.nansum(scores*weights,axis=1)/np.maximum(have@weights,1e-12)
    table["Evidence_coverage_pct"] = 100*weight_coverage
    table["Evidence_score"] = 100*score
    rng=np.random.default_rng(seed)
    draws=rng.uniform(.75,1.25,size=(scenarios,len(weights)))*weights
    draws=draws/np.maximum(draws.sum(axis=1,keepdims=True),1e-12)
    raw=np.where(have[None,:,:],scores[None,:,:],0)
    denom=np.einsum("sn,kn->sk",draws,have.astype(float))
    numer=np.einsum("sn,kn->sk",draws,np.where(have,scores,0))
    trial=100*numer/np.maximum(denom,1e-12)
    trial_ranks=pd.DataFrame(trial.T).rank(axis=0,method="min",ascending=False).to_numpy().T
    table["Score_sensitivity_min"] = trial.min(axis=0)
    table["Score_sensitivity_max"] = trial.max(axis=0)
    table["Best_rank"] = trial_ranks.min(axis=0).astype(int)
    table["Worst_rank"] = trial_ranks.max(axis=0).astype(int)
    table["First_rank_scenario_share"] = np.mean(trial_ranks==1,axis=0)
    table["Screening_confidence"] = np.where((weight_coverage>=.75) & ((table.Worst_rank-table.Best_rank)<=1),
                                              "Moderate screening support", "Limited / missing evidence")
    return table.sort_values("Evidence_score",ascending=False).reset_index(drop=True)


# V9.2: descriptive seismic amplitude clusters. NOT a reservoir classifier.
def descriptive_seismic_facies(frame):
    """Split every full-coverage valid paired cell into three descriptive seismic
    signature bins. No geological labels are inferred; NOT probability of sand.
    Works on 1995 static attributes, avoiding 4D production leakage.
    """
    out=frame.copy()
    out["Seismic_facies_cluster"]=np.nan
    available=[k for k in ("B_RMS","B_EnvelopeMean","B_MostNegative") if k in out]
    if not available or out.empty:
        return out
    feats=out[available].replace([np.inf,-np.inf],np.nan)
    safe=feats.notna().all(axis=1)
    if safe.sum()<6:
        return out
    # Deterministic robust standardized amplitude index; three equal-frequency bins.
    x=feats.loc[safe].to_numpy(float)
    center=np.median(x,axis=0)
    scale=np.maximum(np.percentile(x,75,axis=0)-np.percentile(x,25,axis=0),1e-9)
    score=np.mean(np.clip((x-center)/scale,-5,5),axis=1)
    bins=np.quantile(score,[1/3,2/3])
    out.loc[safe,"Seismic_facies_cluster"]=np.digitize(score,bins,right=False)+1
    return out
