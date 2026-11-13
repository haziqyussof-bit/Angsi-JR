"""Streamlit entry points for V8 scientific workflows."""
from ui_help import help_text, VOLUME_HELP

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import cKDTree

import angsi_core as C
from opportunities import COMPONENTS, projected_geometry, rank, regions
from petrophysics import interval_support
from seismic_qc import analyze_trace, register
from survey_analysis import VERSION, evaluate, file_stamp, sample_pairs
from uncertainty import sensitivity
from opportunity_intelligence import (
    REQUIRED as HCPV_FIELDS, label_and_rank_candidates, hcpv_scenarios,
    example_hypothetical_inputs, blank_inputs, evidence_gaps,
)


def export(label, frame, name):
    st.download_button(label, frame.to_csv(index=False), name, "text/csv", key=name, help=help_text(label))


def manifest(name, **params):
    body = json.dumps(dict(version=VERSION, **params), indent=2, default=str)
    st.download_button("Export processing parameters", body, name, "application/json", key=name, help=help_text("Export processing parameters"))


def map_plot(frame, field, matched, polygons=(), polygon_labels=None):
    grid = frame.pivot(index="IL", columns="XL", values=field)
    stride = frame.attrs.get("stride")
    if stride:
        grid = grid.reindex(index=np.arange(grid.index.min(), grid.index.max()+1, stride),
                            columns=np.arange(grid.columns.min(), grid.columns.max()+1, stride))
    fig = go.Figure(go.Heatmap(x=grid.columns, y=grid.index, z=grid, colorscale="Viridis",
                               connectgaps=False, colorbar_title=field))
    for identifier, _, _, polygon in polygons:
        # Unions can be Polygon or MultiPolygon. Neither needs to be a rectangle.
        pieces = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
        pretty = (polygon_labels or {}).get(identifier, identifier)
        for piece in pieces:
            for ring in (piece.exterior, *piece.interiors):
                x, y = ring.xy
                fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines", name=pretty,
                                         line=dict(color="#ffdb72", width=2), showlegend=False,
                                         hovertemplate=f"{pretty}<extra></extra>"))
        p = polygon.representative_point()
        fig.add_annotation(x=p.x, y=p.y, text=pretty, showarrow=False,
                           bgcolor="#07293d", font=dict(color="white", size=12))
    if not matched.empty:
        fig.add_trace(go.Scatter(x=matched.XL, y=matched.IL, text=matched.Well, mode="markers+text",
                                 textposition="top center", marker_color="#ff8edf", name="Nearest well trace"))
    fig.update_layout(template="plotly_dark", height=560, xaxis_title="Crossline", yaxis_title="Inline",
                      paper_bgcolor="#031628")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def settings(prefix):
    a, b, c, d = st.columns(4)
    with a:
        ref_start = st.number_input("Reference start (ms)", value=700, key=prefix+"rs", help=help_text("Reference start (ms)"))
    with b:
        ref_end = st.number_input("Reference end (ms)", value=950, key=prefix+"re", help=help_text("Reference end (ms)"))
    with c:
        top = st.number_input("Target top (ms)", value=1500, key=prefix+"ts", help=help_text("Target top (ms)"))
    with d:
        base = st.number_input("Target base (ms)", value=1700, key=prefix+"te", help=help_text("Target base (ms)"))
    a, b, c = st.columns(3)
    with a:
        lag = st.slider("Maximum lag (samples)", 1, 12, 4, key=prefix+"lag", help=help_text("Maximum lag (samples)"))
    with b:
        zeros = st.slider("Suspected padding run (0 disables)", 0, 32, 8, key=prefix+"zeros", help=help_text("Suspected padding run (0 disables)"))
    with c:
        limit = st.number_input("Reference NRMS limit (%)", 1.0, 200.0, 50.0, key=prefix+"limit", help=help_text("Reference NRMS limit (%)"))
    spec = st.checkbox("Try bounded spectral matching", key=prefix+"spectral", help=help_text("Try bounded spectral matching"))
    mute = None
    if st.checkbox("Apply documented early-time mute", key=prefix+"mute", help=help_text("Apply documented early-time mute")):
        mute = st.number_input("Mask samples earlier than (ms)", value=0, key=prefix+"mute-time", help=help_text("Mask samples earlier than (ms)"))
    return dict(reference=(ref_start, ref_end), reservoir=(top, base), max_lag=lag,
                zero_run=zeros, reference_limit=limit, spectral=spec, mute_before_ms=mute)


def qc(data, section, section_id, params):
    bp, mp, _, wells, gb, gm, matched, _ = data
    st.subheader("Advanced seismic QC")
    st.caption("Choose a geologically reviewed non-reservoir reference. Gates are editable screening criteria. "
               "Monitor XY validation is unavailable in the supplied V7 loader. Registration is by exact IL/XL.")
    modules = ("4.1 Survey footprint", "4.2 Trace registration", "4.3 Amplitude and phase",
               "4.4 NRMS and predictability", "4.5 Reference analysis", "4.6 Shift and alignment",
               "4.7 Cross-equalization", "4.8 Sample masking", "4.9 Before and after", "4.10 Confidence")
    selection = st.selectbox("QC module", modules, help=help_text("QC module"))
    if selection.startswith("4.1 ") or selection.startswith("4.2 "):
        common = gb[["IL", "XL"]].merge(gm[["IL", "XL"]], validate="one_to_one")
        inventory = pd.DataFrame({"Survey": ["1995", "2006"],
                                  "Trace_count": [len(gb), len(gm)],
                                  "Matched": [len(common)]*2,
                                  "Unpaired": [len(gb)-len(common), len(gm)-len(common)]})
        st.dataframe(inventory, hide_index=True)
        st.caption("Exact header pairs confirm common grid keys. Survey navigation and monitor positional alignment require field verification.")
        if selection.startswith("4.1 ") and not common.empty:
            step = max(1, len(common)//45000)
            fig = go.Figure(go.Scattergl(x=common.XL[::step], y=common.IL[::step],
                                         mode="markers", marker=dict(size=2), name="Sampled paired keys"))
            fig.add_trace(go.Scatter(x=matched.XL, y=matched.IL, text=matched.Well,
                                     mode="markers+text", name="Wells"))
            fig.update_layout(template="plotly_dark", xaxis_title="Crossline", yaxis_title="Inline")
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, width="stretch")
        return
    if section is None:
        st.info("Choose an inline or crossline with paired coverage for trace diagnostics.")
        return
    pairs, time, b, m = section
    records, outputs = [], []
    try:
        for row, a, z in zip(pairs.itertuples(), b.T, m.T):
            record, a, z, corrected = analyze_trace(time, a, z, **params)
            records.append(dict(IL=row.IL, XL=row.XL, **record))
            outputs.append((a, z, corrected))
    except ValueError as exc:
        st.warning(str(exc))
        return
    table = pd.DataFrame(records)
    fields = {
        modules[2]: ["raw_baseline_rms", "raw_monitor_rms", "phase_offset_deg"],
        modules[3]: ["raw_nrms", "raw_predictability", "reference_nrms", "reference_predictability"],
        modules[4]: ["reference_nrms", "reference_valid_fraction", "heldout_before_nrms", "heldout_after_nrms"],
        modules[5]: ["shift_ms", "shift_split_ms"],
        modules[6]: ["gain", "spectral_applied", "reason"],
        modules[7]: ["raw_valid_fraction", "reference_valid_fraction"],
        modules[8]: ["raw_common_reservoir_nrms", "corrected_reservoir_nrms"],
        modules[9]: ["status", "reason", "shift_split_ms", "raw_valid_fraction"]
    }[selection]
    st.dataframe(table[["IL", "XL", *fields]], hide_index=True)
    trace = st.select_slider("Diagnostic trace", options=list(range(len(table))),
                             format_func=lambda i: f"IL {table.IL[i]} / XL {table.XL[i]}", help=help_text("Diagnostic trace"))
    a, z, corrected = outputs[trace]
    fig = go.Figure()
    for name, values in (("Baseline", a), ("Monitor", z), ("Raw difference", z-a),
                         ("Corrected difference", corrected-a)):
        if np.isfinite(values).any():
            fig.add_trace(go.Scatter(x=time, y=values, name=name, mode="lines"))
    for bounds, name, color in ((params["reference"], "Reference", "#15d1eb"),
                                (params["reservoir"], "Target", "#ffc95d")):
        fig.add_vrect(x0=bounds[0], x1=bounds[1], fillcolor=color, opacity=.14,
                      line_width=0, annotation_text=name)
    fig.update_layout(template="plotly_dark", xaxis_title="TWT (ms)", yaxis_title="Amplitude")
    st.plotly_chart(fig, width="stretch")
    if selection == modules[8]:
        from plotly.subplots import make_subplots
        raw = np.column_stack([z-a for a,z,_ in outputs])
        adjusted = np.column_stack([c-a for a,_,c in outputs])
        delta = make_subplots(rows=1, cols=2, shared_yaxes=True,
                              subplot_titles=["Raw 4D", "Corrected 4D (accepted traces)"])
        if np.isfinite(raw).any():
            limit = max(1e-8,np.nanpercentile(abs(raw),98))
            for col, values in enumerate((raw, adjusted),1):
                delta.add_trace(go.Heatmap(z=values,y=time,zmin=-limit,zmax=limit,
                                          colorscale="RdBu_r",showscale=col==2),row=1,col=col)
            delta.update_yaxes(autorange="reversed",title_text="TWT (ms)")
            delta.update_xaxes(title_text="Trace ordinal")
            delta.update_layout(template="plotly_dark")
            st.plotly_chart(delta,width="stretch")
    if selection == modules[6] and np.isfinite(corrected).any():
        spectrum = go.Figure()
        ref = (time >= params["reference"][0]) & (time <= params["reference"][1])
        valid = ref & np.isfinite(a+z+corrected)
        samples = np.flatnonzero(valid)
        if len(samples) >= 16 and np.all(np.diff(samples) == 1):
            for name, values in (("Baseline", a), ("Raw monitor", z), ("Corrected monitor", corrected)):
                series = values[samples]*np.hanning(len(samples))
                spectrum.add_trace(go.Scatter(x=np.fft.rfftfreq(len(series), (time[1]-time[0])/1000),
                                              y=abs(np.fft.rfft(series)), name=name))
            spectrum.update_layout(template="plotly_dark", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
            st.plotly_chart(spectrum, width="stretch")
    st.caption("PASS requires bounded gain, a stable interior shift, positive reference correlation and held-out NRMS within threshold. "
               "Long exact-zero runs are an optional padding heuristic. Individual zeros are valid.")
    st.session_state.v8_qc_evidence = {"selection": section_id, "params": params,
                                       "status_counts": table.status.value_counts().to_dict(),
                                       "selected_trace": table.iloc[trace].to_dict()}
    export("Export trace QC CSV", table, "angsi_trace_qc.csv")
    manifest("angsi_trace_qc.json", inputs=[file_stamp(bp), file_stamp(mp)], section=section_id, settings=params)


@st.cache_data(show_spinner="Evaluating paired survey traces…", max_entries=4)
def survey_cached(b_stamp, m_stamp, stride, params, top_spec=None, base_spec=None):
    bp, mp = b_stamp[0], m_stamp[0]
    gb, gm = C.seismic_grid(bp, xy=True), C.seismic_grid(mp)
    pairs = sample_pairs(gb, gm, stride)
    top = dict(top_spec) if top_spec is not None else None
    base = dict(base_spec) if base_spec is not None else None
    frame, qc = evaluate(bp, mp, pairs, params, top, base, return_qc=True)
    # Count sampled grid keys missing from either vintage as explicit coverage exclusions.
    common_ref = (gb.IL.min(), gb.XL.min())
    b_sample = gb.loc[((gb.IL-common_ref[0]) % stride == 0) & ((gb.XL-common_ref[1]) % stride == 0), ["IL", "XL"]]
    m_sample = gm.loc[((gm.IL-common_ref[0]) % stride == 0) & ((gm.XL-common_ref[1]) % stride == 0), ["IL", "XL"]]
    coverage = b_sample.merge(m_sample, on=["IL", "XL"], how="outer", indicator=True,
                              validate="one_to_one")
    missed = coverage.loc[coverage._merge != "both", ["IL", "XL", "_merge"]].copy()
    if not missed.empty:
        missed["reason"] = missed["_merge"].map({"left_only": "no_monitor_trace", "right_only": "no_baseline_trace"}).astype(str)
        missed["included"] = False
        qc = pd.concat([qc, missed[["IL", "XL", "reason", "included"]]], ignore_index=True)
    return frame, qc


def surface_dict(surface):
    base = surface["base"]
    usable = surface["confident"] & np.isfinite(base)
    for axis in (0,1):
        jumps = abs(np.diff(base, axis=axis)) > 20
        if axis == 0:
            usable[:-1] &= ~jumps
            usable[1:] &= ~jumps
        else:
            usable[:, :-1] &= ~jumps
            usable[:, 1:] &= ~jumps
    keys = zip(surface["IL"][usable], surface["XL"][usable])
    return tuple(zip(keys, surface["base"][usable]))


def _control_markers(tops):
    return tops[(tops.Checkshot_Donor == tops.Well) & tops.TWT_ms.notna() &
                ~tops.Depth_Source.str.contains("assumed", na=False)]


def _score_candidates(candidates, matched, controls, frame, conformable, proximity_scale=None):
    rows = []
    for identifier, category, cells, _ in candidates:
        xy = cells[["X", "Y"]].to_numpy()
        usable = np.isfinite(xy).all(axis=1) & (abs(xy).sum(axis=1) > 0)
        cell_xy = xy[usable]
        quality, cutoff = [], []
        distance = np.nan
        if len(cell_xy) and len(matched):
            well_xy = matched[["X", "Y"]].to_numpy(float)
            valid_wells = np.isfinite(well_xy).all(axis=1)
            if valid_wells.any():
                distance = float(cKDTree(cell_xy).query(well_xy[valid_wells])[0].min())
        cell_keys = cells[["IL","XL"]].to_numpy(float)
        half_spacing = cells.attrs.get("stride", 1)/2
        for tie in matched.itertuples():
            support = controls.get(tie.Well)
            if support is not None and tie.Distance_m <= 100 and (np.abs(cell_keys-[tie.IL,tie.XL]).max(axis=1) <= half_spacing).any():
                quality.append(support["quality"])
                cutoff.append(support["cutoff_span"])
        finite_quality = [q for q in quality if np.isfinite(q)]
        finite_cutoff = [q for q in cutoff if np.isfinite(q)]
        contrast = float(cells.local_repeatability_contrast.median())
        seismic = min(1., contrast/3) if category == "High contrast" else max(0., 1-contrast)
        ref_nrms = float(cells.reference_nrms.median())
        seismic *= max(0., 1-ref_nrms/200)
        proximity = np.exp(-distance/proximity_scale) if proximity_scale else np.nan
        stability = float(cells.shift_split_ms.median())
        rows.append(dict(ID=identifier, Category=category, Cells=len(cells),
                         Reservoir=np.mean(finite_quality) if finite_quality else np.nan,
                         Structure=np.nan, Seismic=seismic,
                         Proximity=proximity, Horizon=np.nan, Production=np.nan,
                         Nearest_well_coordinate_distance=distance,
                         Reference_NRMS=ref_nrms, NRMS_window_span=cells.nrms_window_span.median(),
                         RMS_difference_window_span=cells.rms_window_span.median(),
                         Shift_split_ms=stability, Support_wells=len(finite_quality),
                         Cutoff_span=max(finite_cutoff, default=np.nan)))
        if conformable:
            relative_time = (frame.top_ms.max()-cells.top_ms.median())/(frame.top_ms.max()-frame.top_ms.min()) if frame.top_ms.nunique()>1 else np.nan
            rows[-1]["Structure"] = relative_time
            rows[-1]["Horizon"] = (1-cells.horizon_sensitivity_ms/20).clip(0,1).median()
            rows[-1]["Horizon_sensitivity_ms"] = cells.horizon_sensitivity_ms.median()
            rows[-1]["Depth_conversion_span_m"] = cells.depth_conversion_span_m.median()
    return pd.DataFrame(rows)


def screening(data, surface_loader, log_loader):
    bp, mp, wd, wells, gb, gm, matched, tops = data
    st.subheader("4D attributes and opportunity screening")
    st.caption("Choose a TWT window, checkshot-derived depth window, or an interpreted horizon pair. "
               "The same selected target window is applied to both vintages.")
    mode = st.radio("Extraction mode", ["Time window (TWT)", "Depth window (TVDSS)",
                                        "Horizon-bounded"], horizontal=True, key="v81_extraction_mode", help=help_text("Extraction mode"))
    params = dict(reference=(700., 950.), reservoir=(1500., 1700.), max_lag=4,
                  zero_run=8, reference_limit=50., spectral=False, mute_before_ms=None)
    if mode == "Time window (TWT)":
        cols = st.columns(2)
        with cols[0]: target_top = st.number_input("Extraction top TWT (ms)", value=1500.0, step=10.0, key="v81_time_top", help=help_text("Extraction top TWT (ms)"))
        with cols[1]: target_base = st.number_input("Extraction base TWT (ms)", value=1700.0, step=10.0, key="v81_time_base", help=help_text("Extraction base TWT (ms)"))
        params["reservoir"] = (target_top, target_base)
        domain_label = f"TWT {target_top:g}–{target_base:g} ms"
    elif mode == "Depth window (TVDSS)":
        options = [well for well in wells.Well if len(C.read_checkshot(wd, well)) >= 2]
        if not options:
            st.error("Depth mode requires at least one validated checkshot.")
            return
        donor = st.selectbox("Checkshot for depth-to-time conversion", options, key="v81_depth_well", help=help_text("Checkshot for depth-to-time conversion"))
        cols = st.columns(2)
        with cols[0]: depth_top = st.number_input("Extraction top TVDSS (m)", value=1500.0, step=10.0, key="v81_depth_top", help=help_text("Extraction top TVDSS (m)"))
        with cols[1]: depth_base = st.number_input("Extraction base TVDSS (m)", value=1700.0, step=10.0, key="v81_depth_base", help=help_text("Extraction base TVDSS (m)"))
        transformed = C.depth_to_time(np.array([depth_top, depth_base]), C.read_checkshot(wd, donor))
        if not np.isfinite(transformed).all():
            st.error("The chosen depth interval falls outside the selected checkshot range. No extrapolation applied.")
            return
        params["reservoir"] = tuple(map(float, transformed))
        domain_label = f"TVDSS {depth_top:g}–{depth_base:g} m via {donor} ({transformed[0]:.1f}–{transformed[1]:.1f} ms)"
        st.caption("Approximate depth conversion from one well checkshot; not 3D depth migration.")
    else:
        domain_label = "Horizon pair"
    stride = st.radio("Sample spacing in IL/XL", [5, 10, 15], index=1, horizontal=True,
                      key="v81_grid_step", help=help_text("Sample spacing in IL/XL"))
    markers = _control_markers(tops)
    available = markers.groupby("Marker").Well.nunique()
    choices = sorted(available[available >= 2].index.tolist())
    top_name = base_name = None
    if mode == "Horizon-bounded":
        if len(choices) < 2:
            st.error("Two independently controlled marker horizons are required.")
            return
        a, b = st.columns(2)
        with a: top_name = st.selectbox("Top interpreted horizon", choices, key="v81_horizon_top", help=help_text("Top interpreted horizon"))
        with b: base_name = st.selectbox("Base interpreted horizon", choices, index=min(1, len(choices)-1), key="v81_horizon_base", help=help_text("Base interpreted horizon"))
        if top_name == base_name:
            st.warning("Choose different top and base markers.")
            return
        st.caption("Only cells with valid, non-crossing baseline horizon picks are evaluated. Missing picks are counted as excluded.")
        domain_label = f"Horizon {top_name} → {base_name}"
    with st.expander("Advanced QC reference / cross-equalization settings", expanded=False):
        st.caption("Reference window must be a reviewed non-reservoir interval. It is distinct from extraction target.")
        refa, refb = st.columns(2)
        with refa: ref_top = st.number_input("Reference start (ms)", value=700.0, key="v81_ref_top", help=help_text("Reference start (ms)"))
        with refb: ref_base = st.number_input("Reference end (ms)", value=950.0, key="v81_ref_base", help=help_text("Reference end (ms)"))
        params["reference"] = (ref_top, ref_base)
        aa, bb, cc = st.columns(3)
        with aa: params["max_lag"] = st.slider("Maximum lag (samples)", 0, 12, 4, key="v81_max_lag", help=help_text("Maximum lag (samples)"))
        with bb: params["zero_run"] = st.slider("Suspected padding run (0 disables)", 0, 32, 8, key="v81_zero_run", help=help_text("Suspected padding run (0 disables)"))
        with cc: params["reference_limit"] = st.number_input("Reference NRMS limit (%)", 1., 200., 50., key="v81_nrms_gate", help=help_text("Reference NRMS limit (%)"))
        params["spectral"] = st.checkbox("Try bounded spectral matching", value=False, key="v81_spectral", help=help_text("Try bounded spectral matching"))
        if st.checkbox("Mask early-time samples using verified mute", value=False, key="v81_early_mute", help=help_text("Mask early-time samples using verified mute")):
            params["mute_before_ms"] = st.number_input("Mute before TWT (ms)", 0., 4000., 0., key="v81_mute_ms", help=help_text("Mute before TWT (ms)"))
        st.caption("Correction status and raw sample validity are reported separately.")
    if params["reservoir"][0] >= params["reservoir"][1] or params["reference"][0] >= params["reference"][1]:
        st.warning("Top must precede base for both extraction and reference windows.")
        return
    if st.button("Calculate survey diagnostics", type="primary", key="v81_survey_calculate", help=help_text("Calculate survey diagnostics")):
        st.session_state.pop("v8_candidate_evidence", None)
        st.session_state.pop("v8_attribute_evidence", None)
        st.session_state.v8_survey_request = (params, stride, top_name, base_name, mode, domain_label)
    req = st.session_state.get("v8_survey_request")
    if req is None:
        st.info("Select a mode and click Calculate survey diagnostics. Default: 1500–1700 ms, 10 IL/XL.")
        return
    # Support saved requests from the previous app version without breaking the page.
    if len(req) == 4:
        params, stride, top_name, base_name = req
        saved_mode, saved_label = "Legacy selection", str(params["reservoir"])
    else:
        params, stride, top_name, base_name, saved_mode, saved_label = req
    st.caption(f"Calculated settings · {saved_label} · Reference {params['reference']} ms · Grid step {stride}")
    b_stamp, m_stamp = file_stamp(bp), file_stamp(mp)
    try:
        a, b = None, None
        if top_name:
            a = surface_loader(str(bp), str(mp), top_name, stride, 20, 250)
            b = surface_loader(str(bp), str(mp), base_name, stride, 20, 250)
            if a is None or b is None:
                st.warning("Selected markers lack sufficient well control. Choose a different horizon pair or use Time window.")
                return
        frame, cell_qc = survey_cached(b_stamp, m_stamp, stride, params,
                                       surface_dict(a) if a else None, surface_dict(b) if b else None)
    except (ValueError, OSError) as exc:
        st.warning(str(exc))
        return
    st.markdown("#### Extraction coverage & excluded-cell QC")
    if cell_qc.empty:
        st.warning("No sampled survey cells found. Check IL/XL sampling origin and grid spacing.")
        return
    counts = cell_qc.reason.value_counts().rename_axis("QC reason").reset_index(name="Cells")
    total = len(cell_qc)
    eligible = int(cell_qc.included.fillna(False).sum())
    paired = int((~cell_qc.reason.isin(["no_monitor_trace", "no_baseline_trace"])).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sampled cells", f"{total:,}")
    c2.metric("Paired traces", f"{paired:,}")
    c3.metric("Screening-valid cells", f"{eligible:,}")
    c4.metric("Excluded / flagged", f"{total-eligible:,}")
    st.dataframe(counts, hide_index=True, width="stretch")
    with st.expander("Inspect or download per-cell QC classifications"):
        st.dataframe(cell_qc, hide_index=True, width="stretch")
        export("Download extraction cell QC CSV", cell_qc, "angsi_extraction_cell_qc.csv")
    st.caption("A paired trace can be excluded from local-repeatability screening while retaining a valid raw amplitude attribute. "
               "QC correction PASS/FAIL is separate from sample validity; excluded cells are not interpreted as zero 4D change.")
    if frame.empty:
        st.warning("No paired traces were processed for the selected interval. Check the QC reason counts above. "
                   "Try Time window 1500–1700 ms if horizon coverage is insufficient.")
        return
    if top_name:
        horizon_span = {}
        for surface in (a, b):
            for il, xl, value in zip(surface["IL"].ravel(), surface["XL"].ravel(),
                                     surface["sensitivity"].ravel()):
                horizon_span[il, xl] = max(horizon_span.get((il, xl), 0), value)
        frame["horizon_sensitivity_ms"] = [horizon_span.get(key, np.nan) for key in zip(frame.IL, frame.XL)]
        depths = []
        for well in wells.Well:
            cs = C.read_checkshot(wd, well)
            if len(cs) > 1:
                depths.append(C.time_to_depth(frame.top_ms, cs))
        span = np.full(len(frame), np.nan)
        if len(depths) > 1:
            values = np.asarray(depths)
            supported = np.isfinite(values).sum(axis=0) > 1
            span[supported] = np.nanmax(values[:,supported],axis=0)-np.nanmin(values[:,supported],axis=0)
        frame["depth_conversion_span_m"] = span
    frame["correction_pass"] = (frame.status == "PASS").astype(int)
    fields = ["raw_rms_difference", "raw_nrms", "raw_signed_difference", "raw_relative_rms_change",
              "reference_nrms", "reference_predictability", "raw_predictability", "phase_offset_deg",
              "local_repeatability_contrast", "shift_ms", "raw_common_reservoir_nrms",
              "corrected_reservoir_nrms", "correction_pass", "shift_split_ms", "nrms_window_span",
              "rms_window_span", "raw_valid_fraction"]
    if top_name:
        fields.extend(["horizon_sensitivity_ms","depth_conversion_span_m"])
        st.caption("Depth spread compares the same TWT across available own checkshots. It measures sensitivity to the chosen conversion curve.")
    field = st.selectbox("Attribute map", fields, help=help_text("Attribute map"))
    frame.attrs["stride"] = stride
    fig = map_plot(frame, field, matched)
    st.plotly_chart(fig, width="stretch")
    st.download_button("Export attribute figure (HTML)", fig.to_html(include_plotlyjs=True),
                       "angsi_attribute_map.html", "text/html", help=help_text("Export attribute figure (HTML)"))
    if not frame[field].notna().any():
        st.info("The selected attribute is unavailable for the calculated cells.")
    il, xl = st.columns(2)
    with il:
        chosen_il = st.selectbox("Map inline", sorted(frame.IL.unique()), help=help_text("Map inline"))
    with xl:
        chosen_xl = st.selectbox("Map crossline", sorted(frame.loc[frame.IL == chosen_il, "XL"].unique()), help=help_text("Map crossline"))
    if st.button("Use this trace in Overview and Seismic QC", help=help_text("Use this trace in Overview and Seismic QC")):
        st.session_state.v8_pending_trace = (int(chosen_il), int(chosen_xl))
        st.success("Trace selected. Open Overview or Seismic QC.")
    export("Export survey diagnostics CSV", frame, "angsi_survey_qc.csv")
    st.session_state.v8_attribute_evidence = {"field": field, "valid_cells": int(frame[field].count()),
                                              "screening_eligible_cells": eligible,
                                              "excluded_qc_counts": counts.set_index("QC reason")["Cells"].to_dict(),
                                              "mode": saved_mode, "domain": saved_label,
                                              "median": frame[field].median(), "settings": params}
    st.markdown("#### Scientific uncertainty inventory (always visible)")
    # This report is displayed even if the candidate detector subsequently returns no polygons.
    def _usable_count(column):
        return int(frame[column].notna().sum()) if column in frame.columns else 0
    inventory = pd.DataFrame([
        {"Uncertainty source": "4D window sensitivity", "Data status":
         f"Diagnostics for {_usable_count('nrms_window_span'):,}/{len(frame):,} processed cells",
         "Geological status": "Sensitivity diagnostic; not production validation"},
        {"Uncertainty source": "Correction/time-shift stability", "Data status":
         f"Diagnostics for {_usable_count('shift_split_ms'):,}/{len(frame):,} processed cells",
         "Geological status": "Split-reference stability only"},
        {"Uncertainty source": "Horizon reflector picking", "Data status":
         f"Diagnostics for {_usable_count('horizon_sensitivity_ms'):,}/{len(frame):,} cells" if top_name
         else "Not constrained in fixed-window extraction",
         "Geological status": "Automated reflector preview requires well tie / fault QC"},
        {"Uncertainty source": "Time-to-depth conversion", "Data status":
         f"Checkshot curve sensitivity for {_usable_count('depth_conversion_span_m'):,}/{len(frame):,} cells" if top_name
         else "Not evaluated for this extraction",
         "Geological status": "Not a calibrated spatial velocity uncertainty model"},
        {"Uncertainty source": "Survey polygon area", "Data status": "Pending user-verified metre CRS",
         "Geological status": "Absolute km² and volume unavailable until affine geometry verified"},
        {"Uncertainty source": "Production, pressure, drainage and fluid", "Data status": "Not supplied",
         "Geological status": "Cannot confirm bypassed/untapped oil from 4D alone"},
    ])
    st.dataframe(inventory, hide_index=True, width="stretch")
    st.markdown("#### Candidate screening")
    st.caption("Candidates are connected sampled cells that pass reference QC. A low response is a seismic observation. "
               "Candidate footprints follow the sampled grid. Reservoir, drainage and production require independent evidence.")
    x, y, z = st.columns(3)
    with x:
        low = st.number_input("Low contrast cutoff", .01, .99, .70, help=help_text("Low contrast cutoff"))
    with y:
        high = st.number_input("High contrast cutoff", 1., 10., 1.6, help=help_text("High contrast cutoff"))
    with z:
        minimum = st.number_input("Minimum cells", 2, 200, 4, help=help_text("Minimum cells"))
    manifest("angsi_survey_processing.json", inputs=[b_stamp, m_stamp], stride=stride,
             extraction_mode=saved_mode, domain=saved_label,
             cell_qc_counts=counts.set_index("QC reason")["Cells"].to_dict(), settings=params,
             top_marker=top_name, base_marker=base_name, horizon_search_ms=20,
             horizon_anchor_limit=250, low=low, high=high, minimum=minimum)
    candidates = regions(frame, stride, low, high, minimum)
    if not candidates:
        st.session_state.pop("v8_candidate_evidence", None)
        st.info("No connected regions meet the current criteria.")
        return
    candidate_map_placeholder = st.empty()
    controls = {}
    phie_min, shale_max = .10, .40
    if top_name:
        pcol, scol = st.columns(2)
        phie_min = pcol.number_input("Minimum PHIE (fraction)", .02, .98, .10, step=.01, help=help_text("Minimum PHIE (fraction)"))
        shale_max = scol.number_input("Maximum VSHALE (fraction)", .05, .95, .40, step=.01, help=help_text("Maximum VSHALE (fraction)"))
        for well in wells.itertuples():
            local = markers[markers.Well == well.Well]
            start, stop = local[local.Marker == top_name], local[local.Marker == base_name]
            if len(start) == len(stop) == 1:
                try:
                    controls[well.Well] = interval_support(log_loader(str(wd), well.Well), well.RT,
                                                            start.TVDSS_m.iloc[0], stop.TVDSS_m.iloc[0], phie_min, shale_max)
                except (OSError, ValueError, KeyError):
                    continue
    epsg = st.text_input("Verified projected EPSG code (optional)", placeholder="From survey metadata", help=help_text("Verified projected EPSG code (optional)"))
    area, features = {}, []
    if epsg:
        try:
            for identifier, category, _, polygon in candidates:
                value, geometry = projected_geometry(polygon, gb, int(epsg))
                area[identifier] = value
                features.append(dict(type="Feature", id=identifier, geometry=geometry,
                                     properties=dict(category=category, area_km2=value)))
        except (ValueError, TypeError, RuntimeError) as exc:
            st.warning(f"Projected export unavailable. {exc}")
            area, features = {}, []
    distance_scale = None
    if area and st.checkbox("Use verified well proximity as a coverage score", help=help_text("Use verified well proximity as a coverage score")):
        distance_scale = st.number_input("Proximity decay distance (m)", min_value=50., value=1000., help=help_text("Proximity decay distance (m)"))
    table = _score_candidates(candidates, matched, controls, frame, bool(top_name), distance_scale).set_index("ID")
    table["Area_km2"] = pd.Series(area, dtype=float)
    st.caption("Structure uses relative top TWT. The horizon score decreases linearly with tracking sensitivity over 20 ms. "
               "Optional proximity measures distance to well control. Production and fluid evidence are missing.")
    weights = [col.number_input(f"{name} %", 0, 100, default, key=f"weight-{name}", help=help_text(f"{name} %"))
               for col, name, default in zip(st.columns(6), COMPONENTS, (25, 15, 35, 5, 15, 5))]
    try:
        score = rank(table, weights)
    except ValueError as exc:
        st.warning(str(exc))
        return
    spans = {"Reservoir": table.Cutoff_span}
    if top_name:
        spans["Horizon"] = (table.Horizon_sensitivity_ms/20).clip(0,1)
    spread = sensitivity(table, weights, spans)
    result = table.join(score).join(spread).sort_values(["Score"], ascending=False)
    result["Confidence"] = np.where((result.Evidence_weight_pct >= 65) &
                                     (result.Worst_rank-result.Best_rank <= 1),
                                     "Moderate screening support", "Limited screening support")
    result = label_and_rank_candidates(result)
    result["Evidence_gaps"] = result.apply(evidence_gaps, axis=1)
    labels = result["Opportunity"].to_dict()
    if features:
        for feature in features:
            label = labels.get(feature["id"], feature["id"])
            feature["properties"].update(display_name=label,
                                         overall_rank=int(result.loc[feature["id"], "Overall_rank"]),
                                         evidence_confidence=result.loc[feature["id"], "Confidence"])
        st.download_button("Export WGS84 candidate GeoJSON with OPP names",
                           json.dumps(dict(type="FeatureCollection", features=features), allow_nan=False),
                           "angsi_candidates_v82.geojson", "application/geo+json", help=help_text("Export WGS84 candidate GeoJSON with OPP names"))
    candidate_map_placeholder.plotly_chart(
        map_plot(frame, "local_repeatability_contrast", matched, candidates, polygon_labels=labels),
        width="stretch")

    st.markdown("#### Candidate size & rankings")
    st.caption("**Three distinct rankings**: opportunity attractiveness (scored evidence), "
               "physical polygon area (only after verified metre CRS), and conditional HCPV scenario volume. "
               "OPP labels follow the CURRENT overall rank; original hash IDs stay in CSV for traceability.")
    all_area = result.Area_km2.notna() & (result.Area_km2 > 0)
    largest = result.loc[all_area].sort_values("Area_km2", ascending=False).head(1)
    best = result.sort_values("Overall_rank").head(1)
    a, b, c = st.columns(3)
    a.metric("Best evidence-ranked", best.Opportunity.iloc[0] if not best.empty else "Not available")
    b.metric("Largest verified polygon", largest.Opportunity.iloc[0] if not largest.empty else "Area unavailable")
    c.metric("Largest area (km²)", f"{largest.Area_km2.iloc[0]:.3f}" if not largest.empty else "Not verified")
    if not all_area.any():
        st.warning("**Physical area cannot be calculated yet.** Supply and verify the projected-metre EPSG "
                   "and affine XY fit. Screening cell count is displayed only as a sampling proxy; "
                   "it must NOT be labeled km² or used to calculate absolute volume.")
    core_cols = ["Opportunity", "Category", "Overall_rank", "Area_rank", "Area_km2", "Cells",
                 "Cell_count_rank", "Score", "Evidence_weight_pct", "Confidence", "Evidence_gaps"]
    st.dataframe(result[[x for x in core_cols if x in result]], hide_index=True, width="stretch")
    st.markdown("##### Top 3 by verified mapped area")
    if all_area.any():
        st.dataframe(result.loc[all_area].sort_values("Area_km2", ascending=False)[
            ["Opportunity", "Area_km2", "Area_rank", "Score", "Confidence"]].head(3),
            hide_index=True, width="stretch")
    else:
        st.info("Verified area ranking: NOT AVAILABLE. Largest screened regions by cell count are "
                "shown only as a grid-sampling proxy.")
        st.dataframe(result.sort_values("Cells", ascending=False)[
            ["Opportunity", "Cells", "Cell_count_rank"]].head(3), hide_index=True)

    st.markdown("#### Optional What-if Volume Potential & Sensitivity")
    st.caption("This is **NOT remaining oil**, reserves, recoverable barrels, or production-calibrated prediction. "
               "HCPV = verified polygon area × NET PAY × POROSITY × (1 − Sw), reported as "
               "million reservoir cubic metres (Mm³). You must supply or explicitly enable hypothetical inputs. "
               "Net pay, porosity and water saturation distributions are NOT spatially mapped by this method.")
    calculate_volume = st.checkbox("Enable user-defined hydrocarbon pore-volume what-if analysis",
                                    value=False, key="v82_enable_volume", help=help_text("Enable user-defined hydrocarbon pore-volume what-if analysis"))
    if calculate_volume:
        example_mode = st.checkbox("Populate clearly HYPOTHETICAL example inputs for testing (not Angsi measurements)",
                                   value=False, key="v82_volume_example", help=help_text("Populate clearly HYPOTHETICAL example inputs for testing (not Angsi measurements)"))
        sample_ids = result.index.tolist()
        assumed = example_hypothetical_inputs(sample_ids) if example_mode else blank_inputs(sample_ids)
        assumed.insert(1, "Opportunity", assumed.ID.map(labels))
        st.caption("Edit each candidate separately. Net pay low/mode/high (m); "
                   "phi and water saturation low/mode/high (fractions). "
                   "When using example inputs, all polygons share identical distributions and volume order "
                   "will largely follow area order.")
        data = st.data_editor(assumed, hide_index=True, width="stretch", num_rows="fixed",
                              column_config={
                                  name: st.column_config.TextColumn(name, help=VOLUME_HELP[name])
                                  if name in ("ID", "Opportunity") else
                                  st.column_config.NumberColumn(name, help=VOLUME_HELP[name],
                                      min_value=0.0, format="%.3f")
                                  for name in assumed.columns if name in VOLUME_HELP
                              },
                              disabled=["ID", "Opportunity"],
                              key="v82_assumptions_"+("hyp" if example_mode else "blank")+"_"+
                              sha256(";".join(sorted(sample_ids)).encode()).hexdigest()[:12])
        sc1, sc2 = st.columns(2)
        with sc1:
            draws = st.select_slider("What-if simulations per candidate", options=[500, 1000, 2000, 5000],
                                     value=2000, key="v82_vol_draws", help=help_text("What-if simulations per candidate"))
        with sc2:
            seed = st.number_input("Scenario reproducibility seed", min_value=0,
                                   max_value=1000000000, value=2026, step=1, key="v82_vol_seed", help=help_text("Scenario reproducibility seed"))
        scenario = hcpv_scenarios(result, data.drop(columns="Opportunity"), draws=int(draws), seed=int(seed))
        result = result.join(scenario)
        result = label_and_rank_candidates(result)
        valid = result.HCPV_P50_Mm3.notna()
        if valid.any():
            st.markdown("##### Top 3 by hypothetical hydrocarbon pore volume")
            vol_cols = ["Opportunity", "Volume_rank", "HCPV_P90_Mm3", "HCPV_P50_Mm3", "HCPV_P10_Mm3",
                        "HCPV_relative_spread", "Volume_scenario_top_rank_share_pct", "Confidence"]
            st.dataframe(result.loc[valid].sort_values("HCPV_P50_Mm3", ascending=False)[vol_cols].head(3),
                         hide_index=True, width="stretch")
            largest_vol = result.loc[valid].sort_values("HCPV_P50_Mm3", ascending=False).iloc[0]
            st.metric("Largest conditional HCPV (P50 what-if)",
                      f"{largest_vol.Opportunity} · {largest_vol.HCPV_P50_Mm3:.2f} million reservoir m³")
            st.caption("P90 / P50 / P10 are **scenario exceedance percentiles** from bounded triangular "
                       "assumptions, not validated oil-in-place probabilities. Top-rank share measures "
                       "the frequency of leading size across the selected what-if simulations; "
                       "it is not probability of discovery. Independent scenario draws ignore geology correlation.")
        else:
            st.info("No volume scenario can be computed yet. Verify physical area, then enter valid "
                    "per-candidate net pay, phi and Sw assumptions. No invented values are used.")
        with st.expander("Volume input diagnostics / unavailable reasons", expanded=False):
            st.dataframe(result[["Opportunity", "Volume_assessment"]], hide_index=True, width="stretch")
        export("Export volume scenario CSV (NOT reserves)", result.reset_index(),
               "angsi_whatif_volume_screening.csv")
        manifest("angsi_volume_scenarios.json", units="million reservoir m3 HCPV",
                 simulation="Independent triangular what-if, not calibrated", assumptions=data.to_dict("records"),
                 seed=int(seed), draws=int(draws), verified_epsg=epsg, area_availability=int(all_area.sum()),
                 volume_interpretation="Not in-place oil, gas or recoverable reserves")
    else:
        st.info("Conditional volume ranking is intentionally unavailable until scenario inputs are enabled. "
                "Area and evidence-ranked opportunity results remain available.")

    st.markdown("#### Opportunity uncertainty / evidence coverage")
    uncertainty_cols = ["Opportunity", "Score", "Evidence_weight_pct", "Best_rank", "Worst_rank",
                        "First_place_fraction", "NRMS_window_span", "RMS_difference_window_span",
                        "Horizon_sensitivity_ms", "Depth_conversion_span_m", "Cutoff_span",
                        "Volume_assessment", "HCPV_P90_Mm3", "HCPV_P50_Mm3", "HCPV_P10_Mm3",
                        "HCPV_relative_spread", "Confidence", "Evidence_gaps"]
    st.dataframe(result[[x for x in uncertainty_cols if x in result]], hide_index=True, width="stretch")
    st.caption("Weight-scenario ranks are sensitivity to score assumptions only. Missing production/drainage evidence "
               "is never interpreted as zero reservoir risk. Unavailable uncertainty inputs remain blank.")
    with st.expander("Full auditable opportunity ranking / diagnostics", expanded=False):
        st.dataframe(result, width="stretch")
    chosen = st.selectbox("Candidate evidence card", result.index.tolist(),
                          format_func=lambda cid: f"{result.at[cid, 'Opportunity']} — {result.at[cid, 'Category']}", help=help_text("Candidate evidence card"))
    st.markdown(f"**{result.at[chosen, 'Opportunity']} — {result.at[chosen, 'Category']}**")
    st.dataframe(result.loc[[chosen]].T.rename(columns={chosen: "Evidence"}), width="stretch")
    st.caption("Ranking sensitivity uses 300 fixed-seed weight scenarios with factors 0.8–1.2 and "
               "renormalization. Sample-window sensitivity shifts ±1 sample; petrophysical cutoff "
               "sensitivity uses PHIE ±0.02 and shale ±0.05.")
    st.session_state.v8_candidate_evidence = result.reset_index().head(20).to_dict("records")
    export("Export ranked candidates CSV", result.reset_index(), "angsi_ranked_candidates_v82.csv")
    manifest("angsi_ranking_processing.json", inputs=[b_stamp, m_stamp], weights=dict(zip(COMPONENTS, weights)),
             scenario_seed=2026, scenarios=300, low=low, high=high, minimum=minimum, epsg=epsg, proximity_scale_m=distance_scale,
             stride=stride, settings=params, top_marker=top_name, base_marker=base_name,
             phie_min=phie_min, shale_max=shale_max)

