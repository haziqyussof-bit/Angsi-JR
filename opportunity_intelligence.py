"""ANGSI JR. V8.2 — auditable size, ranking, and *conditional* volume scenarios.

All volumes are hydrocarbon pore volume (HCPV), NOT oil/gas in-place,
recoverable reserves, production-derived remaining volumes or probabilities.
Assumptions MUST be user supplied or explicitly enabled as hypothetical.
No hidden default is allowed to produce real-field volume estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = (
    "NetPay_low_m", "NetPay_mode_m", "NetPay_high_m",
    "Phi_low", "Phi_mode", "Phi_high",
    "Sw_low", "Sw_mode", "Sw_high",
)


def label_and_rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a new table with friendly display names and separate rankings.

    Overall labels are regenerated after each scoring/filter change, while the
    persistent hash ID remains intact for audit and exports.
    """
    if frame.empty:
        return frame.copy()
    t = frame.copy()
    if "ID" in t.columns:
        t = t.set_index("ID", drop=True)
    if not t.index.is_unique:
        raise ValueError("Candidate IDs must be unique")
    idx = pd.Index(t.index.astype(str), name="ID")
    t.index = idx
    score = pd.to_numeric(t.get("Score", pd.Series(np.nan, index=idx)), errors="coerce")
    # Stable tie breaks: score DESC, persistent id ASC.
    order = sorted(idx, key=lambda key: (-score[key] if np.isfinite(score[key]) else np.inf, key))
    labels = {id_: f"OPP-{i:02d}" for i, id_ in enumerate(order, 1)}
    t["Opportunity"] = [labels[k] for k in idx]
    t["Overall_rank"] = pd.Series({k: i for i, k in enumerate(order, 1)}).reindex(idx).astype(int)

    area = pd.to_numeric(t.get("Area_km2", pd.Series(np.nan, index=idx)), errors="coerce")
    if "Area_km2" in t:
        t["Area_rank"] = area.where(area > 0).rank(method="min", ascending=False).astype("Int64")
    else:
        t["Area_rank"] = pd.Series(pd.array([pd.NA]*len(t), dtype="Int64"), index=idx)
    if "Cells" in t:
        t["Cell_count_rank"] = pd.to_numeric(t["Cells"], errors="coerce").rank(
            method="min", ascending=False
        ).astype("Int64")
    if "HCPV_P50_Mm3" in t:
        v = pd.to_numeric(t["HCPV_P50_Mm3"], errors="coerce")
        t["Volume_rank"] = v.where(v >= 0).rank(method="min", ascending=False).astype("Int64")
    else:
        t["Volume_rank"] = pd.Series(pd.array([pd.NA]*len(t), dtype="Int64"), index=idx)
    return t


def _validate_one(values, area):
    if not np.isfinite(area) or area <= 0:
        return "Physical area unavailable: verify a projected metre CRS and geometry fit"
    missing = [x for x in REQUIRED if x not in values]
    if missing:
        return "Scenario inputs missing: " + ", ".join(missing)
    try:
        a = np.array([float(values[k]) for k in REQUIRED], dtype=float)
    except (TypeError, ValueError):
        return "Scenario inputs must be finite numeric values"
    if not np.isfinite(a).all():
        return "Scenario inputs missing or non-finite"
    if a.min() < 0:
        return "Negative thickness/porosity/saturation is invalid"
    nlo, nmd, nhi = a[:3]
    plo, pmd, phi = a[3:6]
    slo, smd, shi = a[6:9]
    if nlo <= 0 or not nlo <= nmd <= nhi:
        return "Net pay bounds must satisfy 0 < low <= mode <= high"
    if not 0 < plo <= pmd <= phi <= 1:
        return "Porosity must satisfy 0 < low <= mode <= high <= 1"
    if not 0 <= slo <= smd <= shi <= 1:
        return "Water saturation must satisfy 0 <= low <= mode <= high <= 1"
    return "OK"


def _draw_triangular(rng, left: float, mode: float, right: float, draws: int) -> np.ndarray:
    # NumPy raises ValueError for point-valued (deterministic) assumptions.
    if left == right:
        return np.full(draws, left, dtype=float)
    return rng.triangular(left, mode, right, draws)


def hcpv_scenarios(candidates: pd.DataFrame, assumptions: pd.DataFrame,
                   draws: int = 2000, seed: int = 2026) -> pd.DataFrame:
    """Monte Carlo of *user-defined what-if* HCPV, output Mm³ reservoir pore volume.

    HCPV [million reservoir m3] = area [km²] * net pay [m] * phi * (1-Sw).
    P90/P50/P10 here are synthetic scenario *exceedance* percentiles, not
    geological resource distributions or calibrated confidence intervals.
    Candidates are drawn independently; spatial/geologic correlations are NOT modeled.
    """
    if draws < 200 or draws > 50000:
        raise ValueError("Draws must be between 200 and 50,000")
    c = candidates.set_index("ID") if "ID" in candidates.columns else candidates
    a = assumptions.set_index("ID") if "ID" in assumptions.columns else assumptions
    if not c.index.is_unique or not a.index.is_unique:
        raise ValueError("IDs must be unique")
    rng = np.random.default_rng(seed)
    rows, vectors = {}, {}
    for cid, row in c.iterrows():
        area = pd.to_numeric(pd.Series([row.get("Area_km2", np.nan)]), errors="coerce").iloc[0]
        vals = a.loc[cid] if cid in a.index else {}
        status = _validate_one(vals, area)
        rec = {
            "Volume_assessment": status,
            "HCPV_P90_Mm3": np.nan,
            "HCPV_P50_Mm3": np.nan,
            "HCPV_P10_Mm3": np.nan,
            "HCPV_range_Mm3": np.nan,
            "HCPV_relative_spread": np.nan,
            "Volume_scenario_top_rank_share_pct": np.nan,
        }
        if status == "OK":
            vals = {k: float(vals[k]) for k in REQUIRED}
            t = _draw_triangular(rng, vals["NetPay_low_m"], vals["NetPay_mode_m"], vals["NetPay_high_m"], draws)
            phi = _draw_triangular(rng, vals["Phi_low"], vals["Phi_mode"], vals["Phi_high"], draws)
            sw = _draw_triangular(rng, vals["Sw_low"], vals["Sw_mode"], vals["Sw_high"], draws)
            v = float(area) * t * phi * (1 - sw)
            if np.isfinite(v).all():
                q10, q50, q90 = np.quantile(v, [.10, .50, .90])
                rec.update(HCPV_P90_Mm3=q10, HCPV_P50_Mm3=q50, HCPV_P10_Mm3=q90,
                           HCPV_range_Mm3=q90-q10,
                           HCPV_relative_spread=(q90-q10)/q50 if q50 > 0 else np.nan)
                vectors[cid] = v
        rows[cid] = rec
    if vectors:
        ids = sorted(vectors)
        samples = np.stack([vectors[i] for i in ids], axis=0)
        # Equal-volume ties are shared rather than assigned arbitrarily.
        best = np.isclose(samples, samples.max(axis=0, keepdims=True), atol=1e-12, rtol=1e-10)
        win_share = best/best.sum(axis=0, keepdims=True)
        for i, cid in enumerate(ids):
            rows[cid]["Volume_scenario_top_rank_share_pct"] = 100*float(win_share[i].mean())
    return pd.DataFrame.from_dict(rows, orient="index").reindex(c.index)


def example_hypothetical_inputs(ids, net=(5., 15., 25.), phi=(.12, .20, .28),
                                sw=(.20, .40, .60)) -> pd.DataFrame:
    """Explicit hypothetical example only, NEVER field-derived parameters."""
    vals = dict(zip(REQUIRED, [*net, *phi, *sw]))
    return pd.DataFrame([dict(ID=cid, **vals) for cid in ids])


def blank_inputs(ids) -> pd.DataFrame:
    return pd.DataFrame([dict(ID=cid, **{k: np.nan for k in REQUIRED}) for cid in ids])


def evidence_gaps(row) -> str:
    gaps = []
    for field, label in (("Reservoir", "reservoir data"), ("Structure", "structure"),
                         ("Horizon", "horizon confidence"), ("Production", "production/drainage")):
        if not np.isfinite(row.get(field, np.nan)):
            gaps.append(label)
    if not np.isfinite(row.get("Area_km2", np.nan)):
        gaps.append("verified surface area")
    if float(row.get("Support_wells", 0) or 0) <= 0:
        gaps.append("local well control")
    return ", ".join(gaps) if gaps else "None identified in available score components"
