"""East–West seismic fence and well projections."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from interpretation import fence_seed, reflector
from seismic_qc import sample_mask


def figure(distance, vertical, baseline, monitor, difference, palettes, domain, vertical_exaggeration=1):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, shared_yaxes=False,
                       vertical_spacing=.055, subplot_titles=("1995 Baseline", "2006 Monitor", "4D Difference"))
    common = np.abs(np.r_[baseline.ravel(), monitor.ravel()])
    delta = np.abs(difference.ravel())
    amp, diff = (max(1e-8, np.percentile(v[np.isfinite(v)], 98)) if np.isfinite(v).any() else 1
                 for v in (common, delta))
    for row, (values, limit, palette) in enumerate(zip((baseline, monitor, difference),
                                                     (amp, amp, diff), palettes), 1):
        fig.add_trace(go.Heatmap(x=distance, y=vertical, z=values, zmin=-limit, zmax=limit,
                                 colorscale=palette, showscale=False, connectgaps=False), row=row, col=1)
    fig.update_yaxes(autorange="reversed", matches="y",
                     title_text="TWT (ms)" if domain == "TWT" else "TVDSS (m)")
    fig.update_xaxes(title_text="Along fence (km)", row=3, col=1)
    fig.update_layout(template="plotly_dark", height=1050, dragmode="pan", uirevision=f"fence-{domain}-ve{vertical_exaggeration}")
    if vertical_exaggeration>1:
        finite=np.asarray(vertical,float)[np.isfinite(vertical)]
        if len(finite)>2:
            middle=(float(finite.min())+float(finite.max()))/2
            half=(float(finite.max())-float(finite.min()))/(2*vertical_exaggeration)
            fig.update_yaxes(range=[middle+half,middle-half],autorange=False)
    return fig


def render(data, mode, cs, names, markers, show_logs, show_tops, log_loader, cs_loader, palettes,show_rt=False):
    import pandas as pd
    import streamlit as st
    import angsi_core as C

    bp, mp, wd, wells, gb, gm, matched, tops = data
    ordered = wells.sort_values("X", ascending=False)
    extension_m = int(st.session_state.get("v10_fence_extension",1250))
    route = C.nearest_trace_section(bp, mp, gb, gm, ordered[["X", "Y"]].to_numpy(),
                                    extension_m=extension_m)
    if route is None:
        st.warning("No paired seismic coverage along the East–West route.")
        return
    pairs, stations = route
    result = C.read_arbitrary(bp, mp, pairs, max_traces=400)
    if result is None:
        return
    d, time, b, m = result
    axis = C.time_to_depth(time, cs) if mode == "TVDSS" else time
    valid = np.isfinite(axis)
    if not valid.any():
        st.warning("The selected checkshot has no valid overlap with the seismic samples.")
        return
    # Blank the midpoint of large gaps so the heatmap and picks cannot bridge them.
    gaps = np.flatnonzero(np.diff(d) > max(.25, 2.5*np.median(np.diff(d))))+1 if len(d)>1 else []
    if len(gaps):
        d = np.insert(d, gaps, (d[gaps]+d[gaps-1])/2)
        b, m = (np.insert(values, gaps, np.nan, axis=1) for values in (b, m))
    b, m = (np.column_stack([np.where(sample_mask(trace), trace, np.nan) for trace in v.T])
            for v in (b, m))
    fig = figure(d, axis[valid], b[valid], m[valid], (m-b)[valid], palettes, mode,
                 vertical_exaggeration=int(st.session_state.get("v9_vertical_exaggeration",1)))
    colors = ["#26dfea", "#ff93e1", "#f7b848", "#a2eb7c", "#adb5ff"]
    domain_col = "TVDSS_m" if mode == "TVDSS" else "TWT_ms"
    for j, marker in enumerate(markers):
        source = tops[(tops.Marker == marker) & tops.TWT_ms.notna() &
                      (tops.Checkshot_Donor == tops.Well)]
        covered = matched.loc[(matched.Distance_m <= 100) & matched.Monitor_Index.notna(), "Well"]
        source = source[source.Well.isin(covered)].groupby("Well").TWT_ms.median()
        seed = fence_seed(d, stations, ordered.Well.map(source).to_numpy(float))
        pick, _, _, _ = reflector(time, b, seed, radius=22)
        y = C.time_to_depth(pick, cs) if mode == "TVDSS" else pick
        for row in (1, 2, 3):
            fig.add_trace(go.Scatter(x=d, y=y, name=marker, mode="lines", showlegend=row == 1,
                                     line=dict(color=colors[j % len(colors)], width=2)),
                          row=row, col=1)
    for well, xpos in zip(ordered.itertuples(), stations):
        if well.Well not in names:
            continue
        markers_at_well = tops[(tops.Well == well.Well) & tops[domain_col].notna()]
        logs = None
        resistivity_log=None
        if show_logs or show_rt:
            try:
                log = log_loader(str(wd), well.Well)
                depth = pd.to_numeric(log.iloc[:,0], errors="coerce").to_numpy(float)-well.RT
                gr = pd.to_numeric(log.GR_P, errors="coerce").to_numpy(float) if "GR_P" in log else np.full(len(depth),np.nan)
                rt = pd.to_numeric(log.RT_P, errors="coerce").to_numpy(float) if "RT_P" in log else np.full(len(depth),np.nan)
                if mode == "TWT":
                    own, donor, _, _ = cs_loader(well.Well, wd, wells)
                    depth = C.depth_to_time(depth, own) if donor == well.Well else np.full_like(depth, np.nan)
                keep = np.flatnonzero(np.isfinite(depth+gr) & (depth >= axis[valid].min()) &
                                      (depth <= axis[valid].max()))[::4]
                if show_logs and len(keep):
                    logs = (xpos-np.clip(gr[keep]/150, 0, 1)*max(stations[-1]*.018, .05), depth[keep])
                if show_rt:
                    chosen=np.flatnonzero(np.isfinite(depth+rt)&(rt>0) & (depth >= axis[valid].min()) & (depth <= axis[valid].max()))[::4]
                    if len(chosen):
                        resistivity_log=(xpos+np.clip((np.log10(np.clip(rt[chosen],.02,200))-np.log10(.02))/4.,0,1)*max(stations[-1]*.018,.05),depth[chosen])
            except (OSError, ValueError, KeyError, AttributeError):
                pass
        for row in (1, 2, 3):
            fig.add_vline(x=xpos, line_dash="dot", line_color="#ffe068", row=row, col=1)
            fig.add_trace(go.Scatter(x=[xpos], y=[axis[valid].min()], text=[well.Well],
                                     mode="markers+text", textposition="bottom center",
                                     showlegend=False, marker_color="#ffe068"), row=row, col=1)
            if show_tops and len(markers_at_well):
                fig.add_trace(go.Scatter(x=[xpos]*len(markers_at_well), y=markers_at_well[domain_col],
                                         text=markers_at_well.Marker, mode="markers",
                                         marker=dict(symbol="diamond", color="#ff76d3"), showlegend=False),
                              row=row, col=1)
            if logs is not None:
                fig.add_trace(go.Scatter(x=logs[0], y=logs[1], mode="lines", name=f"{well.Well} GR left",
                                         showlegend=False, line=dict(color=st.session_state.get("v102_qc_gr_colour","#a3f37c"), width=1)), row=row, col=1)
            if resistivity_log is not None:
                fig.add_trace(go.Scatter(x=resistivity_log[0], y=resistivity_log[1], mode="lines", name=f"{well.Well} RT right",
                                         showlegend=False, line=dict(color=st.session_state.get("v102_qc_rt_colour","#ff8462"), width=1.2)), row=row, col=1)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False})
    st.slider("Extend seismic fence beyond outermost wells (metres)",0,3000,1250,250,
              key="v10_fence_extension",help="Placed below the seismic viewer. Only actual matched SEG-Y traces are shown. Drag and Streamlit will rerun the fence.")
    st.caption("Fence ends extend beyond the outermost wells only where BOTH surveys have real traces. "
               "Dotted well paths and log depths use vertical assumptions. Colored lines are baseline reflector picks projected onto all panels. TWT overlays require in-range checkshot coverage for each log.")

