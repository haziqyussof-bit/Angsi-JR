"""Auditable trace-wise 4D diagnostics. Arrays use (sample, trace) order."""
import numpy as np
from scipy.ndimage import label
from scipy.signal import hilbert


def register(baseline, monitor):
    keys = ["IL", "XL"]
    pairs = baseline.rename(columns={"idx": "bi"}).merge(
        monitor[keys + ["idx"]].rename(columns={"idx": "mi"}),
        on=keys, how="outer", validate="one_to_one", indicator=True)
    return pairs


def sample_mask(values, zero_run=8):
    """Mask long exact-zero runs as suspected padding only when enabled."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if zero_run:
        runs, _ = label(values == 0)
        lengths = np.bincount(runs)
        valid &= ~((runs > 0) & (lengths[runs] >= zero_run))
    return valid


def window(time, bounds):
    start, stop = bounds
    if not np.isfinite([start, stop]).all() or start >= stop:
        raise ValueError("Window requires ordered finite endpoints")
    return (time >= start) & (time <= stop)


def metrics(a, b, minimum=8):
    a, b = np.asarray(a), np.asarray(b)
    good = np.isfinite(a) & np.isfinite(b)
    n = int(good.sum())
    names = ("baseline_rms", "monitor_rms", "rms_difference", "nrms",
             "predictability", "signed_difference", "relative_rms_change")
    result = dict.fromkeys(names, np.nan)
    result["valid_samples"] = n
    if n < minimum:
        return result
    x, y = a[good], b[good]
    rb, rm, rd = np.sqrt([np.mean(x*x), np.mean(y*y), np.mean((y-x)**2)])
    energy = np.dot(x, x) * np.dot(y, y)
    result.update(baseline_rms=rb, monitor_rms=rm, rms_difference=rd,
                  nrms=200*rd/(rb+rm) if rb+rm else np.nan,
                  predictability=100*np.dot(x, y)**2/energy if energy else np.nan,
                  signed_difference=np.mean(y-x),
                  relative_rms_change=100*(rm-rb)/rb if rb else np.nan)
    return result


def _lag_fit(a, b, ref, lags):
    idx = np.flatnonzero(ref)
    idx = idx[(idx + lags[0] >= 0) & (idx + lags[-1] < len(a))]
    if not len(idx):
        return None
    sampled = b[idx[:, None] + lags]
    common = np.isfinite(a[idx]) & np.isfinite(sampled).all(axis=1)
    if common.sum() < max(12, len(idx)//2):
        return None
    x, y = a[idx[common]], sampled[common]
    xc, yc = x-x.mean(), y-y.mean(axis=0)
    denominator = np.linalg.norm(xc)*np.linalg.norm(yc, axis=0)
    scores = np.divide(xc @ yc, denominator, out=np.full(len(lags), -1.0), where=denominator>0)
    best = int(np.argmax(scores))
    shifted = y[:, best]
    gain = np.dot(x, shifted)/np.dot(shifted, shifted) if np.any(shifted) else np.nan
    competing = np.abs(lags-lags[best]) > 1
    margin = scores[best]-max(scores[competing], default=-1.0)
    return int(lags[best]), gain, scores[best], margin


def _shift(values, lag):
    aligned = np.full_like(values, np.nan)
    target = np.arange(len(values))
    source = target + lag
    valid = (source >= 0) & (source < len(values))
    aligned[valid] = values[source[valid]]
    return aligned


def _spectral_fit(a, b, train):
    """Estimate a bounded, smooth amplitude-only filter from contiguous training data."""
    from scipy.ndimage import gaussian_filter1d
    indices = np.flatnonzero(train)
    if len(indices) < 32 or not np.all(np.isfinite(a[indices]) & np.isfinite(b[indices])):
        return None
    taper = np.hanning(len(indices))
    x = abs(np.fft.rfft(taper*a[indices]))
    y = abs(np.fft.rfft(taper*b[indices]))
    floor = 0.15*max(y)
    if floor <= 0:
        return None
    ratio = gaussian_filter1d((x*y+floor**2)/(y*y+floor**2), 1.5)
    impulse = np.fft.fftshift(np.fft.irfft(np.clip(ratio, .67, 1.5), n=len(indices)))
    middle = len(impulse)//2
    return impulse[middle-4:middle+5]*np.hanning(9)


def _phase(a, b):
    """Circular mean phase offset, energy weighted, for contiguous valid data."""
    if len(a) < 12 or not np.isfinite(a+b).all() or not np.any(a) or not np.any(b):
        return np.nan
    product = np.vdot(hilbert(a), hilbert(b))
    return np.degrees(np.angle(product)) if abs(product) else np.nan


def analyze_trace(time, baseline, monitor, reference, reservoir, max_lag=3,
                  zero_run=8, reference_limit=50, spectral=False, mute_before_ms=None):
    time = np.asarray(time, float)
    baseline, monitor = np.asarray(baseline, float), np.asarray(monitor, float)
    if baseline.shape != time.shape or monitor.shape != time.shape or time.ndim != 1 or len(time) < 2:
        raise ValueError("Expected equal 1D sample arrays")
    dt = time[1]-time[0]
    if not np.isfinite(time).all() or dt <= 0 or not np.allclose(np.diff(time), dt):
        raise ValueError("Correction requires uniform ascending TWT")
    if max_lag < 0 or max_lag > 12 or zero_run < 0 or reference_limit <= 0:
        raise ValueError("Invalid QC parameters")
    if max(reference[0], reservoir[0]) <= min(reference[1], reservoir[1]):
        raise ValueError("Reservoir and reference windows must be disjoint")
    ref, res = window(time, reference), window(time, reservoir)
    # Reserve a guard interval so shifted/filter inputs cannot cross into another window.
    guard = max_lag + (4 if spectral else 0)
    if np.any(ref & np.convolve(res.astype(int), np.ones(2*guard+1), mode="full")[guard:guard+len(time)].astype(bool)):
        raise ValueError("Reference and reservoir require a wider sample guard interval")
    live = time >= mute_before_ms if mute_before_ms is not None else np.ones(len(time),dtype=bool)
    a = np.where(live & sample_mask(baseline, zero_run), baseline, np.nan)
    b = np.where(live & sample_mask(monitor, zero_run), monitor, np.nan)
    samples = np.flatnonzero(ref)
    split = len(samples)//2
    train, test = ref.copy(), ref.copy()
    train[samples[max(0, split-guard):]] = False
    test[samples[:split+guard]] = False
    lags = np.arange(-max_lag, max_lag+1)
    fit = _lag_fit(a, b, train, lags)
    ref_result = metrics(a[ref], b[ref])
    raw = metrics(a[res], b[res])
    record = {**{f"raw_{k}": v for k, v in raw.items()},
              **{f"reference_{k}": v for k, v in ref_result.items()},
              "phase_offset_deg": _phase(a[ref], b[ref]),
              "shift_ms": np.nan, "gain": np.nan, "shift_split_ms": np.nan,
              "heldout_before_nrms": np.nan, "heldout_after_nrms": np.nan,
              "raw_common_reservoir_nrms": np.nan, "corrected_reservoir_nrms": np.nan,
              "spectral_applied": False, "status": "REVIEW", "reason": "Insufficient paired reference support"}
    corrected = np.full_like(b, np.nan)
    if fit is not None:
        lag, gain, corr, margin = fit
        record.update(shift_ms=lag*dt, gain=gain, lag_correlation=corr, lag_margin=margin)
        alternate = _lag_fit(a, b, test, lags)
        if alternate:
            record["shift_split_ms"] = abs(alternate[0]-lag)*dt
        aligned = gain*_shift(b, lag)
        kernel = _spectral_fit(a, aligned, train) if spectral else None
        if kernel is not None:
            aligned = np.convolve(aligned, kernel, mode="same")
        support = test & np.isfinite(a) & np.isfinite(b) & np.isfinite(aligned)
        before, after = metrics(a[support], b[support]), metrics(a[support], aligned[support])
        record.update(heldout_before_nrms=before["nrms"], heldout_after_nrms=after["nrms"])
        plausible = (.5 <= gain <= 2 and corr >= .65 and (max_lag == 0 or abs(lag) < max_lag)
                     and margin >= .02 and alternate is not None and abs(alternate[0]-lag) <= 1)
        if plausible and after["nrms"] <= before["nrms"]+1e-8 and after["nrms"] <= reference_limit:
            corrected = aligned
            common = res & np.isfinite(a) & np.isfinite(b) & np.isfinite(corrected)
            record.update(status="PASS", reason="Held-out reference repeatability gate",
                          spectral_applied=kernel is not None,
                          raw_common_reservoir_nrms=metrics(a[common], b[common])["nrms"],
                          corrected_reservoir_nrms=metrics(a[common], corrected[common])["nrms"])
        else:
            record.update(status="FAIL" if np.isfinite(after["nrms"]) else "REVIEW",
                          reason="Correction stability, fit or held-out NRMS gate")
    if res.any() and ref.any() and raw["valid_samples"] >= max(8, .8*res.sum()) and ref_result["valid_samples"] >= max(8, .8*ref.sum()):
        if np.isfinite(raw["nrms"]) and 0 < ref_result["nrms"] <= reference_limit:
            record["local_repeatability_contrast"] = raw["nrms"]/ref_result["nrms"]
    record.setdefault("local_repeatability_contrast", np.nan)
    record["raw_valid_fraction"] = raw["valid_samples"]/max(int(res.sum()), 1)
    record["reference_valid_fraction"] = ref_result["valid_samples"]/max(int(ref.sum()), 1)
    return record, a, b, corrected

