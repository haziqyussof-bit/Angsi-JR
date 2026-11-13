"""ANGSI JR. V9 interface: horizon -> seismic attributes -> facies -> Opportunities.

This page is additive to the V8.3 app. All outputs are read-only analyses.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import analysis_engine as E
import angsi_core as C
import v10_features as V10F
import v10_volumetrics as V10V
import v10_auto_volumetrics as V10A
from survey_analysis import sample_pairs, file_stamp

DARK = "#061e31"
TEXT = "#f1fbff"
COLORS = [[0, "#512994"], [.17, "#355acc"], [.38, "#2fb3d7"], [.55, "#4bce8b"], [.77, "#f5df46"], [1, "#eb3838"]]
WELL_MARKERS = dict(symbol="triangle-up",size=9,color="#f8d066",line=dict(color="#071824",width=1))


def _frame_grid(frame, key, stride):
    selected = frame[["IL", "XL", key]].drop_duplicates(["IL","XL"]).copy()
    pivot = selected.pivot(index="IL", columns="XL", values=key)
    # Respect the ACTUAL sampled IL/XL coordinates rather than assuming a new
    # origin from the first valid cell (which can silently drop map samples).
    # Every observed grid row/column is retained. Missing intersections stay NaN.
    return pivot.sort_index().sort_index(axis=1)


def plot_attribute_map(frame, column, matched, *, stride, label, limits=None, show_wells=True):
    table = _frame_grid(frame, column, stride)
    values = table.to_numpy(float)
    finite = values[np.isfinite(values)]
    symmetrical = column.startswith("D_") and column.endswith(("Mean", "MostNegative", "MaxPositive"))
    if limits is None:
        if symmetrical and len(finite):
            c = max(np.nanpercentile(abs(finite),98),1e-12)
            limits=(-c,c)
        elif len(finite):
            limits=(float(np.nanpercentile(finite,2)),float(np.nanpercentile(finite,98)))
    palette = "RdBu_r" if symmetrical else COLORS
    fig=go.Figure(go.Heatmap(x=table.columns.to_numpy(), y=table.index.to_numpy(),
                             z=values, colorscale=palette, zmin=limits[0] if limits else None,
                             zmax=limits[1] if limits else None, connectgaps=False,
                             colorbar=dict(title=label),hovertemplate="IL %{y} / XL %{x}<br>Value %{z:.4g}<extra></extra>"))
    if show_wells and not matched.empty:
        fig.add_trace(go.Scatter(x=matched.XL,y=matched.IL,mode="markers+text",text=matched.Well,
                                 textposition="top center",marker=WELL_MARKERS,
                                 textfont=dict(color="white",size=9),name="Well (nearest trace)",
                                 hovertemplate="%{text} · IL %{y}, XL %{x}<extra></extra>"))
    fig.update_layout(template="plotly_dark",title=label,height=435,
                      paper_bgcolor=DARK,plot_bgcolor=DARK,
                      xaxis_title="Crossline",yaxis_title="Inline",
                      margin=dict(l=35,r=5,t=45,b=40),font=dict(color=TEXT,size=10))
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    return fig


def frame_coverage(frame, qc, gb, gm, stride):
    common_origin=(int(gb.IL.min()), int(gb.XL.min()))
    def count(g):
        sample = ((g.IL-common_origin[0]) % stride == 0) & ((g.XL-common_origin[1]) % stride == 0)
        return int(sample.sum())
    b, m = count(gb), count(gm)
    counts=qc.reason.value_counts().rename_axis("Reason").reset_index(name="Cells") if not qc.empty else pd.DataFrame(columns=["Reason","Cells"])
    return dict(Baseline_sampled=b, Monitor_sampled=m, Paired_sampled=len(qc),
                Attributes_valid=len(frame),
                Screening_valid=int(frame.QC_screening_pass.sum()) if "QC_screening_pass" in frame else 0,
                Raw_only=int((~frame.QC_screening_pass).sum()) if "QC_screening_pass" in frame else 0,
                Paired_excluded=int((~qc.included.fillna(False)).sum()) if not qc.empty else 0), counts


def _make_surface_pair(data, surface_loader, top_name, base_name, stride):
    import v8_dashboard as V8
    bp,mp,wd,wells,gb,gm,matched,tops=data
    a=surface_loader(str(bp),str(mp),top_name,stride,20,250)
    b=surface_loader(str(bp),str(mp),base_name,stride,20,250)
    if a is None or b is None:
        raise ValueError("One or both horizon previews unavailable. Check AI Horizon 2D & 3D interpretation QC.")
    # Extraction domain and interpretation confidence are DIFFERENT masks.
    # The old V8.surface_dict retained only well-tie high-confidence cells;
    # this unintentionally cut the seismic attribute map into well-centred islands.
    # For exploratory coverage, use picks on actual paired seismic cells, with a
    # trend-model fallback labelled MODEL_ONLY when reflector tracking fails.
    # NEVER interpolate a trace or invent pixels outside the paired footprint.
    def extract(surface):
        mask = np.asarray(surface["has_data"], bool)
        picked = np.asarray(surface["base"],float)
        trend = np.asarray(surface["trend"],float)
        values = np.where(np.isfinite(picked), picked, trend)
        usable = mask & np.isfinite(values)
        confidence = np.asarray(surface["confident"],bool) & np.isfinite(picked)
        status = np.where(confidence,"HIGH_CONFIDENCE",
                 np.where(np.isfinite(picked),"EXPLORATORY_REFLECTOR","MODEL_ONLY"))
        times = {(int(il),int(xl)):float(z)
                 for il,xl,z in zip(surface["IL"][usable],surface["XL"][usable],values[usable])}
        labels = {(int(il),int(xl)):str(s)
                  for il,xl,s in zip(surface["IL"][usable],surface["XL"][usable],status[usable])}
        return times,labels
    aa,astat = extract(a)
    bb,bstat = extract(b)
    if not aa or not bb:
        raise ValueError("No paired seismic footprint has usable horizon predictions")
    keys=set(aa)&set(bb)
    status = {k:("HIGH_CONFIDENCE" if astat[k]==bstat[k]=="HIGH_CONFIDENCE"
                 else "MODEL_ONLY" if "MODEL_ONLY" in (astat[k],bstat[k])
                 else "EXPLORATORY_REFLECTOR") for k in keys}
    st.session_state["v92_surface_qc"] = status
    report=dict(Top_surface=top_name,Base_surface=base_name,
                Top_cells=len(aa),Base_cells=len(bb),Paired_horizon_cells=len(keys),
                High_confidence_cells=sum(v=="HIGH_CONFIDENCE" for v in status.values()),
                Model_only_cells=sum(v=="MODEL_ONLY" for v in status.values()),
                Exploratory_reflector_cells=sum(v=="EXPLORATORY_REFLECTOR" for v in status.values()),
                Coverage_method="whole actual paired trace footprint; predicted picks are NOT validated")
    return aa,bb,report


def _render_attribute_triptych(frame, attr, stride, matched):
    a,b,c=st.columns(3)
    union=pd.concat([frame[f"B_{attr}"],frame[f"M_{attr}"]],ignore_index=True)
    common_limits=(float(np.nanpercentile(union,2)),float(np.nanpercentile(union,98))) if union.notna().any() else None
    with a:
        st.plotly_chart(plot_attribute_map(frame,f"B_{attr}",matched,stride=stride,
                                           label=f"1995 · {E.ATTRIBUTE_CATALOG[attr][0]}",limits=common_limits),
                        width="stretch",key=f"v9_map_b_{attr}")
    with b:
        st.plotly_chart(plot_attribute_map(frame,f"M_{attr}",matched,stride=stride,
                                           label=f"2006 · {E.ATTRIBUTE_CATALOG[attr][0]}",limits=common_limits),
                        width="stretch",key=f"v9_map_m_{attr}")
    with c:
        st.plotly_chart(plot_attribute_map(frame,f"D_{attr}",matched,stride=stride,
                                           label=f"4D raw difference · {E.ATTRIBUTE_CATALOG[attr][0]}"),
                        width="stretch",key=f"v9_map_d_{attr}")
    st.caption("Maps cover all sampled paired seismic traces with usable target samples, NOT just wells or horizon-control buffers. Target-only raw results survive reference QC failures but are explicitly RAW_ONLY. 4D RMS is RMS(monitor-baseline), not RMS(monitor)-RMS(baseline).")



def plot_amplitude_facies_map(frame, matched, *, stride, label):
    """Categorical full-footprint display: grey dim / GREEN intermediate / YELLOW bright."""
    grid=_frame_grid(frame,"AmplitudeFaciesCode",stride)
    categorical=V10F.facies_colorscale()
    fig=go.Figure(go.Heatmap(
        x=grid.columns.to_numpy(),y=grid.index.to_numpy(),z=grid.to_numpy(float),
        colorscale=categorical,zmin=-.5,zmax=2.5,connectgaps=False,
        colorbar=dict(title="Hypothesis",tickvals=[0,1,2],
                      ticktext=["Dim / shale-prone","Intermediate","Bright / sand-prone"]),
        hovertemplate="IL %{y}, XL %{x}<br>Facies code %{z}<extra></extra>"))
    if not matched.empty:
        fig.add_trace(go.Scatter(x=matched.XL,y=matched.IL,mode="markers+text",
            text=matched.Well,textposition="top center",marker=WELL_MARKERS,
            textfont=dict(color="white",size=9),name="Wells (overlay only)"))
    fig.update_layout(template="plotly_dark",title=label,height=510,
        paper_bgcolor=DARK,plot_bgcolor=DARK,xaxis_title="Crossline",yaxis_title="Inline",
        margin=dict(l=40,r=8,t=50,b=40),font=dict(color=TEXT,size=10))
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    return fig

def render(data, surface_loader, log_loader, checkshot_loader):
    bp,mp,wd,wells,gb,gm,matched,tops=data
    st.subheader("Engine 4.5 · Integrated Reservoir Analysis")
    st.caption("Inputs: QC-paired 1995/2006 seismic → AI Horizon surfaces or fixed windows → trace attributes → measured well-log facies control → Opportunities. Auditable, read-only processing.")
    wf1,wf2 = st.tabs(["Workflow 1 · Seismic attribute extraction", "Workflow 2 · Reservoir / non-reservoir screening"])
    with wf1:
        st.markdown("#### Choose an extraction geometry")
        mode=st.radio("Extraction window",["Time window (TWT)","Depth interval (TVDSS)","Interpreted horizon pair"],
                      key="v9_analysis_mode",horizontal=True,help="Time/depth: uniform window. Horizon pair: trace-dependent conformable interval bounded by two independent QC-controlled seismic reflectors.")
        choices=sorted(tops.loc[tops.TWT_ms.notna(),"Marker"].unique().tolist())
        topname=basename=None
        selected_cs=None
        top,bottom=1500.,1700.
        if mode=="Time window (TWT)":
            p,q=st.columns(2)
            with p: top=st.number_input("Top TWT (ms)",value=1500.,step=10.,key="v9_timetop")
            with q: bottom=st.number_input("Base TWT (ms)",value=1700.,step=10.,key="v9_timebase")
            time_interval=(top,bottom)
            domain="Time (TWT)"
        elif mode=="Depth interval (TVDSS)":
            donors=[w for w in wells.Well if len(checkshot_loader(w))>=2]
            if not donors:
                st.error("Depth conversion requires a real checkshot. Stop rather than invent a velocity model.")
                return
            p,q,r=st.columns(3)
            with p: selected_cs=st.selectbox("Reference checkshot",donors,key="v9_depth_reference")
            with q: depth_top=st.number_input("Top TVDSS (m)",value=1500.,step=10.,key="v9_depthtop")
            with r: depth_base=st.number_input("Base TVDSS (m)",value=1700.,step=10.,key="v9_depthbase")
            time_interval=tuple(map(float,C.depth_to_time([depth_top,depth_base],checkshot_loader(selected_cs))))
            domain=f"Depth (TVDSS {depth_top:g}–{depth_base:g} m; {selected_cs} checkshot)"
            st.caption("One checkshot provides an approximate time-depth conversion; the app does not claim full spatial depth migration.")
        else:
            if len(choices)<2:
                st.warning("Two controlled horizon markers needed. Build/review horizons in AI Horizon 2D & 3D or use fixed time.")
                return
            p,q=st.columns(2)
            with p:topname=st.selectbox("Top interpreted horizon",choices,
                                        index=choices.index("I-25") if "I-25" in choices else 0,key="v9_topmarker")
            with q:basename=st.selectbox("Base interpreted horizon",choices,
                                         index=choices.index("I-68") if "I-68" in choices else min(1,len(choices)-1),key="v9_basemarker")
            time_interval=(1500.,1700.) # fallback; dynamic top/base independently picked for every cell.
            domain=f"Horizon {topname} → {basename}"
            st.warning("Whole-surface exploratory extraction: ALL actual paired seismic cells are evaluated. High-confidence reflector ties, tentative reflector picks and trend-model predictions are reported separately. Predicted horizons are NOT validated geology or reliable reservoir evidence.")
        p,q=st.columns([1.15,2])
        with p:stride=st.radio("Sampling (IL/XL)",[5,10,15],index=1,horizontal=True,key="v9_analysis_stride")
        with q:attr=st.selectbox("Attribute to visualize",list(E.ATTRIBUTE_CATALOG),
                                 format_func=lambda key:E.ATTRIBUTE_CATALOG[key][0],key="v9_attr_choice",
                                 help="15+ physical trace attributes; not all proprietary Petrel attributes are implemented.")
        with st.expander("Advanced QC & cross-equalization",expanded=False):
            p,q=st.columns(2)
            with p: ref0=st.number_input("Reference TWT top (ms)",value=700.,key="v9_ref0")
            with q: ref1=st.number_input("Reference TWT base (ms)",value=950.,key="v9_ref1")
            p,q,r=st.columns(3)
            with p:zeros=st.slider("Suspected zero-padding run",0,32,8,key="v9_zeros")
            with q:lag=st.slider("Maximum sample lag",0,12,4,key="v9_lag")
            with r:limit=st.number_input("Reference NRMS gate (%)",min_value=1.,max_value=200.,value=50.,key="v9_gate")
            do_correct=st.checkbox("Run V8.3 held-out cross-equalization QC",value=True,key="v9_run_correction")
            spectral=st.checkbox("Allow bounded spectral matching",value=False,key="v9_spectral")
            mute=st.checkbox("Apply confirmed early-time mute",value=False,key="v9_apply_mute")
            mute_before=st.number_input("Mute before TWT (ms)",value=0.,key="v9_mute_ms") if mute else None
        st.info(f"**Source:** AI Horizon / QC-paired SEG-Y · **Geometry:** {domain} · **Default:** 1500–1700 ms / 10 IL–XL. Source files stay read-only.")
        if st.button("Calculate seismic attributes",type="primary",key="v9_calculate_analysis"):
            if (mode!="Interpreted horizon pair" and (not np.isfinite(time_interval).all() or time_interval[0]>=time_interval[1])) or ref0>=ref1 or topname==basename and topname is not None:
                st.error("Invalid or reversed extraction interval/reference/horizon pair.")
            else:
                try:
                    surface_top=surface_base=None
                    surf_report={}
                    if mode=="Interpreted horizon pair":
                        with st.spinner("Loading AI Horizon picks and confidence mask…"):
                            surface_top,surface_base,surf_report=_make_surface_pair(data,surface_loader,topname,basename,stride)
                    with st.spinner("Extracting actual baseline, monitor and difference seismic attributes…"):
                        pairs=sample_pairs(gb,gm,stride)
                        frame,cell_qc=E.compute_attribute_volume(
                            bp,mp,pairs,bounds=time_interval,top_map=surface_top,base_map=surface_base,
                            reference=(ref0,ref1),zero_run=zeros,max_lag=lag,
                            reference_limit=limit,spectral=spectral,
                            mute_before_ms=mute_before,use_correction=do_correct)
                    if mode=="Interpreted horizon pair":
                        quality=st.session_state.get("v92_surface_qc",{})
                        frame["Horizon_support"]=[quality.get((int(i),int(x)),"MISSING")
                                                  for i,x in zip(frame.IL,frame.XL)]
                        cell_qc["Horizon_support"]=[quality.get((int(i),int(x)),"MISSING")
                                                    for i,x in zip(cell_qc.IL,cell_qc.XL)]
                    else:
                        frame["Horizon_support"]="NOT_HORIZON_MODE"
                        cell_qc["Horizon_support"]="NOT_HORIZON_MODE"
                    summary,counts=frame_coverage(frame,cell_qc,gb,gm,stride)
                    spec=dict(Mode=mode,Description=domain,Top_ms=time_interval[0],Base_ms=time_interval[1],
                              Top_marker=topname,Base_marker=basename,Depth_reference=selected_cs,
                              Depth_bounds_m=(depth_top,depth_base) if mode=="Depth interval (TVDSS)" else None,
                              Reference_TWT=(ref0,ref1),Stride=stride,Correction=do_correct,Spectral=spectral,
                              Zero_run=zeros,Mute_before_ms=mute_before,Max_lag=lag,NRMS_limit=limit,
                              Files=[file_stamp(bp),file_stamp(mp)],Surface_QC=surf_report)
                    st.session_state.pop("v9_opps",None)
                    st.session_state.pop("v10_volumes",None)
                    st.session_state.pop("v10_auto_volumes",None)
                    st.session_state.v9_analysis={"frame":frame,"qc":cell_qc,"spec":spec,
                                                  "summary":summary,"counts":counts,
                                                  "well_labels":pd.DataFrame(),"facies_report":None,
                                                  "frame_facies":None}
                    st.success(f"Survey-wide extraction: {len(frame):,} target-valid paired cells; "
                               f"{summary['Screening_valid']:,} reference-supported; "
                               f"{summary['Raw_only']:,} RAW_ONLY; "
                               f"{summary['Paired_excluded']:,} target-invalid excluded. No well-radius cropping.")
                except Exception as exc:
                    st.error(f"Extraction failed ({type(exc).__name__}): {exc}")
        store=st.session_state.get("v9_analysis")
        if store is not None:
            frame,qc,spec=store["frame"],store["qc"],store["spec"]
            st.markdown("#### QC coverage & exclusion reasons")
            l,m,n,o=st.columns(4)
            l.metric("Sampled paired cells",store["summary"]["Paired_sampled"])
            m.metric("Valid attributes",store["summary"]["Attributes_valid"])
            n.metric("Excluded paired cells",store["summary"]["Paired_excluded"])
            o.metric("Reference-supported",store["summary"].get("Screening_valid",0))
            st.caption(f"**Raw-only cells:** {store['summary'].get('Raw_only',0):,}. These have target data but failed reference quality. Raw maps display them; QC-based interpretation must not treat them as trusted reservoir change.")
            if spec.get("Mode")=="Interpreted horizon pair":
                st.markdown("#### Horizon extraction support (whole survey coverage ≠ confidence)")
                st.dataframe(pd.DataFrame([spec.get("Surface_QC",{})]),width="stretch",hide_index=True)
                if "Horizon_support" in qc:
                    st.dataframe(qc.Horizon_support.value_counts().rename_axis("Horizon support").reset_index(name="Cells"),hide_index=True,width="stretch")
                st.warning("Areas outside high-confidence well ties can appear on attribute maps through tentative reflectors or model-only horizon extrapolations. They must not be interpreted as confirmed reservoir/non-reservoir or bypassed pay.")
            st.dataframe(store["counts"],hide_index=True,width="stretch")
            st.caption(f"Calculation in memory: {spec['Description']} / grid {spec['Stride']} IL–XL; previously calculated results remain until user recalculates. Missing baseline/monitor sampled keys are not counted as paired.")
            if not frame.empty:
                st.markdown("#### 1995 Baseline / 2006 Monitor / raw 4D response")
                _render_attribute_triptych(frame,attr,spec["Stride"],matched)
                st.markdown("#### QC-filtered 4D change")
                corrected_options=[x for x in frame.columns if x.startswith("C_") and frame[x].notna().any()]
                if corrected_options:
                    corrected=st.selectbox("Corrected trace attribute (PASS only)",corrected_options,key="v9_corrected_choice")
                    st.plotly_chart(plot_attribute_map(frame,corrected,matched,stride=spec["Stride"],label=f"Corrected: {corrected}"),width="stretch")
                else:
                    st.caption("No cross-equalization PASS traces for a corrected map, or correction disabled. Raw maps remain available; do not substitute failed corrections silently.")
                a,b=st.columns(2)
                with a:st.download_button("Export 4D seismic attribute CSV",frame.to_csv(index=False),
                                          "angsi_v9_trace_attributes.csv","text/csv")
                with b:st.download_button("Export per-cell QC CSV",qc.to_csv(index=False),
                                          "angsi_v9_analysis_cell_qc.csv","text/csv")
                st.download_button("Export calculation manifest",json.dumps(spec,indent=2,default=str),
                                   "angsi_v9_analysis_manifest.json","application/json")
            else:
                st.warning("No usable attribute cells passed the current QC; see exclusion counts before changing geology assumptions.")
    with wf2:
        st.markdown("#### Reservoir / non-reservoir screening")
        st.caption("Two independent pathways. A well-calibrated model requires independent well labels and blind-well QC. An amplitude-only proxy can scan the WHOLE paired seismic footprint without wells, but provides a hypothesis — not measured sand, shale, oil or gas.")
        store=st.session_state.get("v9_analysis")
        if store is None or store["frame"].empty:
            st.info("Run Workflow 1 first to calculate 1995/2006/4D amplitude attributes. Then choose a Workflow 2 pathway.")
            return
        spec=store["spec"]
        approach=st.radio("Screening pathway",[
            "Amplitude Facies Proxy — whole survey / no training wells",
            "Well-calibrated reservoir screening — requires independent well controls"
        ],horizontal=False,index=0,key="v94_facies_method")
        if approach.startswith("Amplitude"):
            st.info("**Exploratory hypothesis:** brighter 1995 RMS may be sand-prone; dimmer amplitude may be shale-prone. This is NOT validated lithology, not probability of sand, and not an oil/gas predictor. Rock physics, tuning, fluid, processing and stratigraphy may invert this trend.")
            p,q,r=st.columns(3)
            with p:
                strength=st.selectbox("Baseline amplitude-strength attribute",[
                    "B_RMS", "B_EnvelopeMean", "B_AbsolutePeak"],
                    format_func=lambda v:{"B_RMS":"1995 RMS amplitude (recommended)",
                        "B_EnvelopeMean":"1995 mean instantaneous amplitude",
                        "B_AbsolutePeak":"1995 absolute peak amplitude"}[v],
                    key="v94_amp_attribute")
            with q:
                dim_pct=st.slider("Dim / shale-prone threshold (percentile)",10,40,30,5,key="v94_dim_pct")
            with r:
                bright_pct=st.slider("Bright / sand-prone threshold (percentile)",60,90,70,5,key="v94_bright_pct")
            st.caption("Cutoffs are percentiles calculated from every target-valid sampled baseline seismic cell. No well-buffer mask, no required lithology markers, and no 4D amplitude difference is used to define facies. Coverage comes from Workflow 1.")
            if st.button("Calculate amplitude facies proxy · FULL seismic coverage",type="primary",key="v94_amp_run"):
                try:
                    result,report=E.amplitude_facies_proxy(store["frame"],
                        low_percentile=dim_pct,high_percentile=bright_pct,attribute=strength)
                    store["frame_amplitude_proxy"]=result
                    store["amplitude_proxy_report"]=report
                    st.session_state.v9_analysis=store
                    st.session_state.pop("v9_opps",None)
                    if report["status"]=="EXPLORATORY_UNVALIDATED":
                        st.success(f"Survey-wide amplitude proxy complete: {report['valid_amplitude_cells']:,} valid cells. Classification is UNVALIDATED.")
                    else:
                        st.warning("Insufficient amplitude contrast for facies separation; no confident bright/dim distinction possible.")
                except Exception as exc:
                    st.error(f"Amplitude proxy failed ({type(exc).__name__}): {exc}")
            proxy=store.get("frame_amplitude_proxy")
            report=store.get("amplitude_proxy_report")
            if proxy is not None and report is not None:
                st.markdown("##### Whole-survey amplitude facies preview")
                l,m,n,o=st.columns(4)
                l.metric("Valid amplitude cells",report.get("valid_amplitude_cells",0))
                m.metric("Bright / sand-prone hypothesis",report.get("bright_cells",0))
                n.metric("Dim / shale-prone hypothesis",report.get("dim_cells",0))
                o.metric("Intermediate / uncertain",report.get("intermediate_cells",0))
                st.plotly_chart(plot_amplitude_facies_map(proxy,matched,stride=spec["Stride"],
                    label="1995 amplitude-based seismic facies — UNVALIDATED"),width="stretch",key="v94_amp_map")
                st.caption(f"Actual {report.get('attribute','B_RMS')} cutoffs: dim ≤ {report.get('low_cutoff',float('nan')):,.4g}, bright ≥ {report.get('high_cutoff',float('nan')):,.4g}. These are relative survey percentiles, NOT petrophysical cutoffs.")
                qcols=proxy["AmplitudeFaciesQC"].value_counts().rename_axis("QC label").reset_index(name="Cells")
                with st.expander("Classification / exclusions and uncertainty QC",expanded=False):
                    st.dataframe(qcols,hide_index=True,width="stretch")
                    st.json(report)
                    st.warning("RAW_ONLY and MODEL_ONLY sampled cells can be drawn for exploration but are NOT validated reservoir evidence. A blank cell remains unclassified; it is not shale.")
                st.download_button("Export full survey amplitude facies CSV",proxy.to_csv(index=False),
                                   "angsi_v94_unvalidated_amplitude_facies.csv","text/csv",key="v94_amp_export")
                manifest=dict(version="V9.3.1",analysis_spec=spec,facies_proxy=report,
                    scope="Every target-valid paired seismic cell in input analysis frame",
                    geological_validation="NOT VALIDATED", production_inference="NONE")
                st.download_button("Export facies hypothesis provenance",json.dumps(manifest,indent=2,default=str),
                    "angsi_v94_amplitude_facies_manifest.json","application/json",key="v94_amp_manifest")
                st.info("Go to **Opportunities** and choose the amplitude proxy overlay or filter. Screening polygons remain investigation candidates; amplitude facies is never used as a confirmed reservoir probability.")
        else:
            st.caption("Strict independent well control: PHIE/VSHALE/GR proxy → one label per well → leave-one-well-out validation → full-survey seismic similarity. This may legitimately return INSUFFICIENT.")
            a,b=st.columns(2)
            with a:phie=st.number_input("PHIE reservoir threshold (fraction)",min_value=.01,max_value=.60,value=.10,step=.01,key="v9_phi_cut")
            with b:vsh=st.number_input("Maximum VSHALE (fraction)",min_value=.05,max_value=.95,value=.40,step=.05,key="v9_vsh_cut")
            gr_proxy=st.checkbox("Allow explicitly flagged GR-derived shale proxy if VSHALE missing",value=True,key="v92_gr_proxy")
            if st.button("Calculate WELL-CALIBRATED reservoir screening",type="primary",key="v9_facies_run"):
                try:
                    mode={"Time window (TWT)":"Time (TWT)","Depth interval (TVDSS)":"Depth (TVDSS)",
                          "Interpreted horizon pair":"Horizon pair"}[spec["Mode"]]
                    depth_bounds=spec.get("Depth_bounds_m") or (1500.,1700.)
                    time_bounds=(spec["Top_marker"],spec["Base_marker"]) if mode=="Horizon pair" else (spec["Top_ms"],spec["Base_ms"])
                    labels=E.well_labels(store["frame"],wells,matched,tops,wd=wd,mode=mode,
                                         time_window=time_bounds,depth_window=depth_bounds,
                                         cs_loader=checkshot_loader,log_loader=log_loader,
                                         phie_min=phie,vshale_max=vsh,allow_gr_proxy=gr_proxy)
                    enriched,report=E.reservoir_screen(store["frame"],labels)
                    store.update(well_labels=labels,facies_report=report,frame_facies=enriched)
                    st.session_state.pop("v9_opps",None)
                    st.session_state.v9_analysis=store
                except Exception as exc:
                    st.error(f"Well-calibrated facies screening failed ({type(exc).__name__}): {exc}")
            report=store.get("facies_report")
            if report is not None:
                st.markdown(f"**Status:** {report['status']} · **Reason:** {report['reason']}")
                if "loo_balanced_accuracy" in report:
                    st.metric("Leave-one-well-out balanced accuracy (exploratory)",f"{report['loo_balanced_accuracy']:.0%}")
                if not store["well_labels"].empty:
                    st.dataframe(store["well_labels"],hide_index=True,width="stretch")
                facies=store["frame_facies"]
                if report["status"]=="EXPLORATORY":
                    st.plotly_chart(plot_attribute_map(facies,"Reservoir_similarity_score",matched,
                        stride=spec["Stride"],label="Well-conditioned seismic similarity · NOT P(sand)",limits=(0,1)),width="stretch")
                    st.download_button("Export well-calibrated seismic facies CSV",facies.to_csv(index=False),
                                       "angsi_v9_well_screening.csv","text/csv")
                else:
                    st.warning("Insufficient independent lithology controls. Do NOT force well labels by changing cutoffs. Use the separate unvalidated Amplitude Facies Proxy pathway to explore full coverage instead.")
        with st.expander("Optional: whole-survey UNSUPERVISED seismic signature clusters (unlabelled)",expanded=False):
            st.caption("Clusters 1–3 are amplitude signatures, not confirmed sand, shale or reservoir labels.")
            if st.button("Calculate unlabelled seismic clusters",key="v92_cluster"):
                store["seismic_facies"]=E.descriptive_seismic_facies(store["frame"])
                st.session_state.v9_analysis=store
            exploratory=store.get("seismic_facies")
            if exploratory is not None and not exploratory.empty:
                st.plotly_chart(plot_attribute_map(exploratory,"Seismic_facies_cluster",matched,
                    stride=spec["Stride"],label="UNVALIDATED seismic signature clusters (1–3)",limits=(1,3)),width="stretch")
        st.markdown("#### Integration with Opportunities")
        st.caption("Workflow 2 results remain available to Opportunities for full-footprint overlays and optional exploratory masks. Re-run Workflow 1 to reset all derived facies when the interval or horizons change.")


def _geom_paths(geom):
    parts=list(geom.geoms) if hasattr(geom,"geoms") else [geom]
    for part in parts:
        yield part.exterior.xy
        for interior in part.interiors:
            yield interior.xy


def opportunities_bridge(data, surface_loader=None):
    """Full common survey coverage, independent of nearest-well selection."""
    bp,mp,wd,wells,gb,gm,matched,tops=data
    st.markdown("### V9 Workflow · Whole-cube Opportunities")
    st.caption("Scans every sampled baseline/monitor pair across the FULL common seismic footprint. Wells provide overlays/control only; there is NO proximity/radius mask. Horizon mode may legitimately have smaller QC-controlled coverage.")
    with st.expander("Run a fresh whole-survey extraction here (no need to visit Analysis)",expanded=True):
        c1,c2,c3=st.columns(3)
        with c1: opp_top=st.number_input("Top TWT (ms)",min_value=0.,value=1500.,step=10.,key="opp_full_top")
        with c2: opp_base=st.number_input("Base TWT (ms)",min_value=0.,value=1700.,step=10.,key="opp_full_base")
        with c3: opp_stride=st.radio("Full-footprint sample spacing IL/XL",[5,10,15],index=1,horizontal=True,key="opp_full_stride")
        if st.button("Calculate FULL seismic coverage + 4D attributes",type="primary",key="opp_full_run"):
            if opp_top >= opp_base:
                st.error("Top TWT must be shallower than Base TWT")
            else:
                try:
                    pairs=sample_pairs(gb,gm,opp_stride)
                    with st.spinner(f"Calculating {len(pairs):,} survey-wide paired traces..."):
                        frame,qc=E.compute_attribute_volume(bp,mp,pairs,bounds=(opp_top,opp_base),
                            reference=(700.,950.),zero_run=8,use_correction=False)
                    summary,counts=frame_coverage(frame,qc,gb,gm,opp_stride)
                    spec=dict(Mode="Time window (TWT)", Description=f"Full survey TWT {opp_top:g}–{opp_base:g} ms",
                        Top_ms=opp_top,Base_ms=opp_base,Top_marker=None,Base_marker=None,
                        Depth_bounds_m=None,Stride=opp_stride,Correction=False,
                        Reference_TWT=(700.,950.),Files=[file_stamp(bp),file_stamp(mp)],
                        Coverage="whole paired seismic footprint; not well-limited")
                    st.session_state.v9_analysis=dict(frame=frame,qc=qc,spec=spec,
                        summary=summary,counts=counts,well_labels=pd.DataFrame(),
                        facies_report=None,frame_facies=None)
                    st.session_state.pop("v9_opps",None)
                    st.success(f"Whole-survey scan completed: {summary['Attributes_valid']:,} target-valid; "
                               f"{summary['Screening_valid']:,} reference-supported; "
                               f"{summary['Raw_only']:,} raw-only; "
                               f"{summary['Paired_excluded']:,} excluded.")
                except Exception as exc:
                    st.error(f"Full survey extraction failed ({type(exc).__name__}): {exc}")
    store=st.session_state.get("v9_analysis")
    if store is None or store["frame"].empty:
        st.info("No V9 Analysis result yet. Run **Analysis → Workflow 1**, then optionally train reservoir / non-reservoir facies in Workflow 2. Existing V8.3 4D screening below still works independently.")
        return
    frame=store["frame"].copy()
    enriched=store.get("frame_facies")
    if enriched is not None and not enriched.empty:
        for key in ("Reservoir_similarity_score","Reservoir_proxy_class"):
            if key in enriched:
                frame[key]=enriched[key].to_numpy()
    amplitude=store.get("frame_amplitude_proxy")
    if amplitude is not None and not amplitude.empty:
        for key in ("AmplitudeFaciesCode","AmplitudeFaciesClass", "AmplitudeFaciesQC", "AmplitudeFaciesAttribute"):
            if key in amplitude:
                frame[key]=amplitude[key].to_numpy()
    report=store.get("facies_report")
    amp_report=store.get("amplitude_proxy_report")
    st.caption(f"Inputs: {store['spec']['Description']} · grid {store['spec']['Stride']} · {len(frame):,} TARGET-VALID sampled paired cells (not all reference QC-passed). Reservoir model: {report.get('status','NOT_RUN') if report else 'NOT_RUN'}. Production/pressure histories are not supplied.")
    if store["spec"].get("Mode")=="Interpreted horizon pair":
        counts=frame.Horizon_support.value_counts().to_dict() if "Horizon_support" in frame else {}
        st.info("**Horizon confidence across this survey-wide map:** " + ", ".join(f"{k}: {v:,}" for k,v in counts.items()) + ". Candidate ranking from MODEL_ONLY areas is NOT a validated reservoir decision.")
    st.markdown("#### Full common survey coverage · 4D raw RMS")
    st.plotly_chart(plot_attribute_map(frame,"D_RMS",matched,stride=store["spec"]["Stride"],
                       label="Full paired seismic footprint: 4D RMS(monitor − baseline)"),
                    width="stretch",key="v91_full_raw_opp_map")
    qc = store.get("qc",pd.DataFrame())
    if not qc.empty:
        counts = qc.reason.value_counts().rename_axis("QC status").reset_index(name="Sampled cells")
        with st.expander("Coverage and exclusions by reason",expanded=False):
            st.dataframe(counts,hide_index=True,width="stretch")
            st.caption("RAW_ONLY cells are mapped for descriptive attribute visualization but are not validated repeatability-controlled 4D signals.")
    detection=st.radio("Whole-survey anomaly detector",[
        "Raw RMS percentile (whole target-valid survey)",
        "Reference-supported local 4D contrast (strict QC)"],
        index=0,key="v9_opp_detector",horizontal=True,
        help="Raw scan covers target-valid traces even when reference QC fails. Strict mode only ranks cells with trustworthy reference repeatability.")
    raw_mode=detection.startswith("Raw")
    p,q,r=st.columns(3)
    if raw_mode:
        with p: low=st.number_input("Low-change cutoff (percentile)",.01,.49,.20,step=.05,key="v9_opp_raw_low")
        with q: high=st.number_input("High-change cutoff (percentile)",.51,.99,.80,step=.05,key="v9_opp_raw_high")
    else:
        with p: low=st.number_input("Low local contrast cutoff",.05,.99,.70,step=.05,key="v9_opp_low")
        with q: high=st.number_input("High local contrast cutoff",1.,9.,1.6,step=.1,key="v9_opp_high")
    with r: minimum=st.number_input("Minimum connected cells",min_value=2,max_value=500,value=5,key="v9_opp_min")
    st.caption("Candidate polygons are computed over ALL eligible sampled IL/XL cells, not within buffers around wells. Raw-only polygons represent relative amplitude changes, not validated pressure/saturation response.")
    supported=bool(report and report.get("status")=="EXPLORATORY")
    require=st.checkbox("Require exploratory well-supported reservoir facies",value=False,disabled=not supported,key="v9_require_res")
    amp_supported=bool(amp_report and amp_report.get("status")=="EXPLORATORY_UNVALIDATED"
                       and "AmplitudeFaciesClass" in frame)
    proxy_options={"No facies restriction":None,
                   "Bright / sand-prone HYPOTHESIS only":"BRIGHT_SAND_PRONE",
                   "Dim / shale-prone HYPOTHESIS only":"DIM_SHALE_PRONE"}
    proxy_choice=st.selectbox("Optional amplitude-facies hypothesis filter for candidate polygons",
        list(proxy_options),disabled=not amp_supported,key="v94_opp_proxy_filter",
        help="Filters full-survey sampled cells by UNVALIDATED 1995 amplitude hypothesis. This is not confirmed lithology or drainage.")
    facies_filter=proxy_options[proxy_choice] if amp_supported else None
    if amp_supported:
        st.caption("Amplitude facies is independent of wells; filtering to bright/dim is a subjective exploration hypothesis and does not establish reservoir presence.")
    wcols=st.columns(4)
    weights=[]
    for name,default,col in zip(["Reservoir","4D response","Repeatability","Well control"],[30,35,20,15],wcols):
        with col: weights.append(st.number_input(name+" weight",min_value=0,max_value=100,value=default,key=f"v9_w_{name}"))
    if st.button("Build ranked opportunity screening polygons",type="primary",key="v9_generate_opps"):
        try:
            region=E.candidate_regions(frame,store["spec"]["Stride"],low=low,high=high,
                                       min_cells=minimum,require_reservoir=require,
                                       metric="Raw_RMS_percentile" if raw_mode else "Local_contrast",
                                       facies_filter=facies_filter)
            ranked=E.rank_candidates(region,frame,wells,weights=weights)
            st.session_state.pop("v10_volumes",None)
            st.session_state.pop("v10_auto_volumes",None)
            st.session_state.pop("v10_auto_well_qc",None)
            st.session_state.v9_opps={"regions":region,"ranking":ranked,"settings":dict(low=low,high=high,
                         minimum=minimum,require_reservoir=require,weights=weights,
                         facies_filter=facies_filter,
                         facies_interpretation="UNVALIDATED amplitude hypothesis" if facies_filter else "None",
                         metric="Raw_RMS_percentile" if raw_mode else "Local_contrast")}
            # Automatically run V10.1 candidate-footprint screening at the same
            # time as opportunity ranking. Missing logs/PVT are reported per
            # candidate; an auto-volume error never blocks the 4D ranking.
            if region:
                try:
                    auto_table,auto_qc,auto_geo=V10A.auto_candidate_volumes(
                        region,gb,wells,wd,tops,store["spec"],
                        bo=float(st.session_state.get("v101_auto_bo",1.25)),
                        bg=float(st.session_state.get("v101_auto_bg",.005)),
                        scf_per_boe=float(st.session_state.get("v101_auto_scf",6000.)),
                        uncertainty=float(st.session_state.get("v101_auto_uncert",25))/100.,
                        attribute_frame=frame)
                    st.session_state.v10_auto_volumes=auto_table
                    st.session_state.v10_auto_well_qc=auto_qc
                    st.session_state.v10_auto_spatial_qc=auto_geo
                    st.info("V10.1 automatic preliminary volumes calculated with opportunity ranking. "
                            "Missing log/phase controls remain explicitly flagged; all numbers are conditional scenarios.")
                except Exception as auto_exc:
                    st.warning(f"Ranked 4D candidates are valid as screening candidates, "
                               f"but auto volumetrics could not run: {type(auto_exc).__name__}: {auto_exc}")
        except Exception as exc:
            st.error(f"Opportunity screening failed ({type(exc).__name__}): {exc}")
    result=st.session_state.get("v9_opps")
    if not result:
        st.caption("Select controls and calculate to draw connected screening polygons. Strong 4D response is not automatically produced; low response is not automatically undrained.")
        return
    regions=result["regions"]
    ranked=result["ranking"]
    if not regions:
        st.warning("No contiguous QC-supported sampled cells meet thresholds. Change only defensible settings after inspecting noise/reference QC.")
        return
    # Keep native IL/XL polygon geometry: do not claim area km² without verified CRS.
    available_maps=["D_RMS","B_RMS","M_RMS","Local_contrast","D_Mean","D_MostNegative","Reference_NRMS"]
    if "Reservoir_similarity_score" in frame and frame["Reservoir_similarity_score"].notna().any():
        available_maps.append("Reservoir_similarity_score")
    if amp_supported:
        available_maps.append("AmplitudeFaciesCode")
    map_key=st.selectbox("Opportunity map background",available_maps,key="v9_opp_background",
                         help="Choose RMS baseline, monitor, 4D difference, reservoir similarity, or local 4D contrast. Ranked polygon footprints remain fixed.")
    show_facies=st.toggle("Overlay well-conditioned reservoir / non-reservoir screening",value=False,
                          disabled=not supported,key="v9_opp_facies_overlay")
    if st.checkbox("Compare RMS baseline / monitor / 4D maps",value=False,key="v9_opportunity_triptych"):
        _render_attribute_triptych(frame,"RMS",store["spec"]["Stride"],matched)
    st.markdown("#### Composite map · facies beneath transparent 4D response")
    composite=st.toggle("Show Facies (bottom) + 4D (top) as spatially aligned layers",
        value=amp_supported,key="v10_composite_on",disabled=not amp_supported)
    transparency=st.slider("4D map transparency (%)",0,100,45,5,
        key="v10_4d_transparency",disabled=not composite,
        help="0% opaque 4D; 100% facies-only. Spatial masks remain unchanged.")
    fig=(plot_amplitude_facies_map(frame,matched,stride=store["spec"]["Stride"],
            label="UNVALIDATED 1995 amplitude facies proxy + candidates") if map_key=="AmplitudeFaciesCode"
         else plot_attribute_map(frame,map_key,matched,stride=store["spec"]["Stride"],label=f"{map_key} + candidate polygons"))
    if composite:
        bottom_fig=plot_amplitude_facies_map(frame,matched,stride=store["spec"]["Stride"],
                           label="1995 facies (unvalidated) beneath 4D response")
        selected_top=map_key if map_key!="AmplitudeFaciesCode" else "D_RMS"
        top_fig=plot_attribute_map(frame,selected_top,matched,stride=store["spec"]["Stride"],
                                  label=selected_top,show_wells=False)
        fig=go.Figure()
        bottom_trace=bottom_fig.data[0]
        bottom_trace.colorbar=dict(title="Facies*",x=1.14,thickness=13,
                    tickvals=[0,1,2],ticktext=["Dim", "Intermediate", "Bright"])
        fig.add_trace(bottom_trace)
        change_trace=top_fig.data[0]
        change_trace.opacity=1.-transparency/100.
        change_trace.colorbar=dict(title=selected_top,x=1.02,thickness=13)
        fig.add_trace(change_trace)
        fig.add_trace(go.Scatter(x=matched.XL,y=matched.IL,mode="markers+text",text=matched.Well,
            textposition="top center",marker=WELL_MARKERS,
            textfont=dict(color="white",size=9),name="Well locations"))
        fig.update_layout(template="plotly_dark",title=f"Facies* + {selected_top} (opacity {100-transparency}%)",
            paper_bgcolor=DARK,plot_bgcolor=DARK,height=560,
            xaxis_title="Crossline",yaxis_title="Inline",
            margin=dict(l=50,r=55,t=50,b=42))
        fig.update_yaxes(scaleanchor="x",scaleratio=1)
        st.caption("*Facies is an UNVALIDATED 1995 amplitude hypothesis, not measured lithology. "
                   "4D opacity changes display only, not the numerical screening or ranking.")
    if amp_supported and map_key!="AmplitudeFaciesCode":
        if st.checkbox("Overlay bright/dim amplitude hypothesis on current map",value=False,key="v94_opp_overlay"):
            zone=frame[frame.AmplitudeFaciesClass.isin(["BRIGHT_SAND_PRONE","DIM_SHALE_PRONE"])]
            if not zone.empty:
                colors=zone.AmplitudeFaciesClass.map({"BRIGHT_SAND_PRONE":V10F.BRIGHT,
                    "DIM_SHALE_PRONE":V10F.DIM}).tolist()
                fig.add_trace(go.Scattergl(x=zone.XL,y=zone.IL,mode="markers",
                    marker=dict(size=4,color=colors,opacity=.65),name="UNVALIDATED amplitude facies",
                    customdata=zone.AmplitudeFaciesClass,
                    hovertemplate="IL %{y}, XL %{x}<br>%{customdata}<extra></extra>"))
    if show_facies and "Reservoir_proxy_class" in frame:
        zone=frame[frame.Reservoir_proxy_class.isin(["Reservoir-favourable proxy","Non-reservoir-favourable proxy"])]
        if not zone.empty:
            colors=zone.Reservoir_proxy_class.map({"Reservoir-favourable proxy":"#4dfca4",
                 "Non-reservoir-favourable proxy":"#ff78a3"}).tolist()
            fig.add_trace(go.Scattergl(x=zone.XL,y=zone.IL,mode="markers",name="Well-conditioned seismic facies",
                         marker=dict(size=5,color=colors,opacity=.65),
                         customdata=zone.Reservoir_proxy_class,
                         hovertemplate="IL %{y}, XL %{x}<br>%{customdata}<extra></extra>"))
    by_id={r["ID"]:r for r in regions}
    top_labels=st.select_slider("Show labels for highest-ranked candidates",
        options=[0,5,10,20,50],value=10,key="v10_candidate_label_limit")
    all_outlines=st.checkbox("Draw every candidate outline (may be cluttered)",value=False,
                             key="v10_all_polygon_outlines")
    for rank,(i,row) in enumerate(ranked.iterrows(),1):
        if rank>50 and not all_outlines:continue
        region=by_id[row.ID]
        col="#ff6c64" if region["Category"].startswith("Higher") else "#e1ff72"
        for px,py in _geom_paths(region["Geometry"]):
            fig.add_trace(go.Scatter(x=list(py),y=list(px),mode="lines",name=row.ID,
                                     showlegend=False,line=dict(color=col,width=2.3),
                                     hovertemplate=row.ID+"<extra></extra>"))
        point=region["Geometry"].representative_point()
        if rank<=top_labels:
            fig.add_annotation(x=point.y,y=point.x,text=f"{row.ID} · #{rank}",showarrow=False,
                               bgcolor="#092a3d",font=dict(color="white",size=9))
    st.plotly_chart(fig,width="stretch",key="v9_opp_map")
    st.caption("Full sampled common seismic footprint is evaluated; NO candidate detection well-buffer mask. Polygons preserve data gaps. Reference-failed raw-only anomalies and all amplitude-facies proxy labels are UNVALIDATED. Areas are in grid cells, NOT km² unless XY/CRS verified.")
    cols=st.columns(3)
    cols[0].metric("Connected candidates",len(ranked))
    cols[1].metric("Reservoir-labelled pixels",int((frame.Reservoir_proxy_class=="Reservoir-favourable proxy").sum()) if "Reservoir_proxy_class" in frame else 0)
    cols[2].metric("Target-valid paired cells",len(frame))
    st.markdown("#### Auto Volumetrics · LOG-DERIVED PRELIMINARY STOIIP / GIIP / BOE")
    st.caption("AUTO runs when you build ranked opportunity polygons; use RECALCULATE only after editing assumptions. AUTO mode uses 4D candidate footprint area from verified survey XY affine geometry, interval-specific PHIE/SWE/GR/VSHALE/VOIL/VGAS from the LAS wells, and interval thickness from measured markers or own checkshots. It is NOT an oil/gas closure, not reserves, and not remaining recoverable hydrocarbons.")
    st.warning("**PVT caveat:** This dataset has not supplied confirmed Bo/Bg/fluid contacts. Volume estimation uses explicit screening FVF assumptions. The 4D anomaly extent is NOT an independent reservoir area or net-pay map. Candidate volumes must not be booked or summed into field reserves.")
    with st.expander("Automatic calculation settings · PVT assumptions & uncertainty", expanded=False):
        pc1,pc2,pc3,pc4=st.columns(4)
        with pc1: auto_bo=st.number_input("ASSUMED Bo (res m³/ST m³)",min_value=.1,value=1.25,step=.05,key="v101_auto_bo")
        with pc2: auto_bg=st.number_input("ASSUMED Bg (res m³/std m³)",min_value=.00001,value=.005,format="%.5f",key="v101_auto_bg")
        with pc3: auto_scf=st.number_input("Gas scf per BOE",min_value=100.,value=6000.,step=500.,key="v101_auto_scf")
        with pc4: auto_uncert=st.slider("Scenario input variation ±%",0,60,25,5,key="v101_auto_uncert")
        st.caption("PHIE/SWE/VSHALE net-pay cutoffs are 0.10 / 0.70 maximum Sw / 0.40 maximum shale. Bo and Bg ARE ASSUMPTIONS, not interpreted from LAS. Percentiles P90/P50/P10 are sensitivity bounds, not reserves certainty.")
    if st.button("🧮 RECALCULATE ALL CANDIDATE VOLUMES (UPDATED ASSUMPTIONS)",type="primary",key="v101_auto_run"):
        try:
            with st.spinner("Reading available LAS/checkshots, QC of physical area and calculating candidate scenarios…"):
                av, well_qc, geo_qc=V10A.auto_candidate_volumes(
                    regions,gb,wells,wd,tops,store["spec"],bo=auto_bo,bg=auto_bg,
                    scf_per_boe=auto_scf,uncertainty=auto_uncert/100.,
                    attribute_frame=frame)
            st.session_state.v10_auto_volumes=av
            st.session_state.v10_auto_well_qc=well_qc
            st.session_state.v10_auto_spatial_qc=geo_qc
            st.success(f"Automatic provisional screening complete: {len(av):,} candidates. "
                       f"{av.Inplace_MMboe.notna().sum()} with log-model phase split; "
                       f"{av.Oil_only_STOIIP_MMstb.notna().sum()} with oil/gas alternative end-member volumes. "
                       "No unsupported field reserve estimate was made.")
        except Exception as exc:
            st.error(f"Auto volumetrics failed: {type(exc).__name__}: {exc}")
    auto_vol=st.session_state.get("v10_auto_volumes")
    if isinstance(auto_vol,pd.DataFrame) and not auto_vol.empty:
        st.markdown("##### Direct preliminary volumes from available wells + candidate area")
        st.dataframe(auto_vol[["ID","Area_km2","Centroid_Lat_WGS84","Centroid_Lon_WGS84","Thickness_m","NTG","PHIE","Sw",
            "Oil_fraction","STOIIP_MMstb","GIIP_Bscf","Inplace_MMboe",
            "Oil_only_STOIIP_MMstb","Gas_only_GIIP_Bscf","P90_MMboe","P50_MMboe","P10_MMboe",
            "Analog_wells","Nearest_analog_m","Thickness_source","Status"]].round(4),width="stretch",hide_index=True)
        qcgeo=st.session_state.get("v10_auto_spatial_qc",{}).get("crs",{})
        st.caption(f"Survey geometry check: assumed {qcgeo.get('assumed_crs','unknown')} | "
                   f"median survey centre {qcgeo.get('lat',float('nan')):.5f}°N / "
                   f"{qcgeo.get('lon',float('nan')):.5f}°E. "
                   f"{qcgeo.get('geographic_plausibility','unknown')}. Exact source CRS needs confirmation.")
        with st.expander("Inspect real LAS interval controls, missing data and assumptions"):
            st.dataframe(st.session_state.get("v10_auto_well_qc",pd.DataFrame()),width="stretch",hide_index=True)
        st.download_button("Export automatic preliminary candidate volumes CSV",auto_vol.to_csv(index=False),
            "ANGSI_V10_1_AUTO_PRELIMINARY_INPLACE.csv","text/csv",key="v101_auto_export")
        st.caption("**Volume provenance:** a nearby well is a lithology/petrophysics ANALOG, not demonstrated connectivity. No phase model → STOIIP/GIIP are separate hypothetical end members; do NOT add them together. Distant analogs must be independently verified.")
    v102_opportunities_panel(data,store,regions,ranked,auto_vol)
    st.markdown("#### Evidence ranking and uncertainty")
    ranked=ranked.copy()
    ranked.insert(0,"Rank",range(1,len(ranked)+1))
    if isinstance(auto_vol,pd.DataFrame) and not auto_vol.empty:
        ranked=ranked.merge(auto_vol[["ID","Area_km2","STOIIP_MMstb","GIIP_Bscf","Inplace_MMboe","P90_MMboe","P50_MMboe","P10_MMboe","Status"]],on="ID",how="left")
    st.dataframe(ranked.round(3),hide_index=True,width="stretch")
    st.markdown("#### Top-ranked candidates · decision-support summary")
    volume_memory=st.session_state.setdefault("v10_volumes",{})
    cards=st.columns(3)
    for card_col,(_,cand) in zip(cards,ranked.head(3).iterrows()):
        with card_col:
            with st.container(border=True):
                st.markdown(f"**#{int(cand.Rank)} · {cand.ID}**")
                st.caption(str(cand.Category))
                st.metric("Relative evidence score",f"{float(cand.Evidence_score):.1f}/100")
                st.caption(f"Coverage: {float(cand.Evidence_coverage_pct):.0f}% · "
                           f"Ranking sensitivity: #{int(cand.Best_rank)}–#{int(cand.Worst_rank)}")
                if cand.ID in volume_memory:
                    vv=volume_memory[cand.ID]
                    st.caption(f"Scenario in-place: {vv.get('Inplace_MMboe',float('nan')):.2f} MMboe "
                               "(user-supplied geological assumptions; NOT reserves)")
                elif isinstance(auto_vol,pd.DataFrame) and cand.ID in set(auto_vol.ID):
                    ar=auto_vol.loc[auto_vol.ID.eq(cand.ID)].iloc[0]
                    if np.isfinite(ar.Inplace_MMboe):
                        st.caption(f"**AUTO PRELIMINARY:** STOIIP {ar.STOIIP_MMstb:.3f} MMstb · "
                                   f"GIIP {ar.GIIP_Bscf:.3f} Bscf · BOE {ar.Inplace_MMboe:.3f} MMboe "
                                   "(log-model phase, assumed PVT, anomaly area)")
                    elif np.isfinite(ar.Oil_only_STOIIP_MMstb):
                        st.caption(f"**Alternative end members** (not additive): oil-only "
                                   f"{ar.Oil_only_STOIIP_MMstb:.3f} MMstb OR gas-only "
                                   f"{ar.Gas_only_GIIP_Bscf:.3f} Bscf. Fluid phase not confirmed.")
                    else:
                        st.caption("AUTO: "+str(ar.Status))
                else:
                    st.caption("STOIIP / GIIP / BOE: click AUTO-CALCULATE to use available LAS and area")
    st.markdown("#### Bubble map · top-ranked opportunity polygons")
    areas,spatial_qc=V10F.candidate_areas_km2(regions,gb,store["spec"]["Stride"])
    map_bubble=plot_attribute_map(frame,"D_RMS",matched,stride=store["spec"]["Stride"],
                                  label="Candidate bubble map · 4D RMS background")
    V10F.map_bubble_layer(map_bubble,ranked,regions,max_labels=min(10,len(ranked)),areas=areas)
    st.plotly_chart(map_bubble,width="stretch",key="v10_opportunity_bubbles")
    if spatial_qc.get("valid"):
        st.caption(f"Physical-area proxy uses baseline affine XY headers: "
                   f"RMSE {spatial_qc['rmse_m']:.1f} m; "
                   f"{spatial_qc['cell_area_m2']:.1f} m² per original IL/XL bin. "
                   "Polygon area is anomaly FOOTPRINT, not geological closure or net pay.")
    else:
        st.warning("Unable to estimate a reliable XY bin area: "+spatial_qc.get('reason','Unknown'))
    st.caption("Evidence_score is a relative weighted index on available components. Raw RMS mode is a **whole-survey anomaly search**, NOT a validated depletion or remaining-oil map. First-rank scenario share is from weight perturbations, not geological probability.")
    chosen=st.selectbox("Inspect candidate evidence",ranked.ID.tolist(),key="v9_opp_selected")
    record=ranked.loc[ranked.ID==chosen].iloc[0]
    st.markdown(f"**{chosen}:** {record.Category} · {int(record.Cells)} sampled cells · Evidence coverage {record.Evidence_coverage_pct:.0f}% · {record.Screening_confidence}.")
    st.markdown("**Evidence → Observation → Alternatives → Check → Confidence → Recommendation**")
    st.write(f"4D observation: median local-contrast index {record.Mean_4D_contrast:.2f}. Independent reservoir evidence: {record.Reservoir if np.isfinite(record.Reservoir) else 'not available'}. Reference QC score: {record.Repeatability:.2f}.")
    st.write("Alternatives: pressure or saturation response, lithologic variability, non-reservoir noise, survey-footprint edge and false absence due to poor sensitivity.")
    st.write("Check: horizon ties and faults, PHIE/VSHALE/SWE calibration, fluid contacts, production history, pressure decline and QC before drilling.")
    st.warning("Candidate is an **investigation target**, NOT proven bypassed oil or a recoverable reserves estimate.")
    with st.expander("🛢️ Volumetric scenarios · GRV / STOIIP / GIIP / BOE (requires geological assumptions)",expanded=False):
        st.warning("The polygons were detected from 4D SEISMIC, not from reservoir closures. "
                   "Without geological closure, thickness, contacts, fluid properties and saturation "
                   "the in-place volumes below are HYPOTHETICAL sensitivity calculations, not discoveries or reserves.")
        approx=areas.get(chosen)
        if approx is not None:
            st.caption(f"Approximate 4D anomaly footprint: {approx:.3f} km², NOT mapped closure. "
                       "Set your independent geological area below.")
        p1,p2,p3=st.columns(3)
        with p1:
            area=st.number_input("Assumed hydrocarbon-bearing area (km²)",min_value=.001,
                value=float(max(.001,approx or 1.0)),step=.1,key=f"v10_area_{chosen}")
            gross=st.number_input("Gross reservoir thickness (m)",min_value=.1,value=30.,step=5.,key=f"v10_gross_{chosen}")
            ntg=st.number_input("Net-to-gross (fraction)",min_value=0.,max_value=1.,value=.55,step=.05,key=f"v10_ntg_{chosen}")
        with p2:
            phie=st.number_input("Effective porosity (fraction)",min_value=0.,max_value=1.,value=.18,step=.01,key=f"v10_porosity_{chosen}")
            sw=st.number_input("Water saturation (fraction)",min_value=0.,max_value=1.,value=.40,step=.05,key=f"v10_sw_{chosen}")
            phase=st.selectbox("Hydrocarbon phase scenario",["Oil","Gas","Mixed"],key=f"v10_phase_{chosen}")
            oil_fraction=1. if phase=="Oil" else 0. if phase=="Gas" else st.slider("Oil fraction of HCPV",0.,1.,.50,.05,key=f"v10_oil_fraction_{chosen}")
        with p3:
            bo=st.number_input("Bo (reservoir m³ / ST m³)",min_value=.01,value=1.25,step=.05,key=f"v10_bo_{chosen}")
            bg=st.number_input("Bg (reservoir m³ / standard m³)",min_value=.00001,value=.005,step=.001,format="%.5f",key=f"v10_bg_{chosen}")
            scf_per_boe=st.number_input("Gas conversion (scf / BOE)",min_value=100.,value=6000.,step=500.,key="v10_scf_boe")
            uncertainty=st.slider("Triangular input uncertainty ±%",0,70,20,5,key="v10_volume_uncertainty")
        confirmed=st.checkbox("I understand these are assumed prospective IN-PLACE scenarios, not oil/gas discoveries, recoverable volumes or reserves.",
            key=f"v10_volumetric_consent_{chosen}")
        if st.button("Calculate STOIIP / GIIP / BOE scenarios",disabled=not confirmed,
                     key=f"v10_volume_calc_{chosen}"):
            try:
                values=V10V.in_place(area,gross,ntg,phie,sw,bo,bg,oil_fraction,scf_per_boe)
                scenario=V10V.sensitivity_scenarios(area_km2=area,thickness_m=gross,ntg=ntg,
                     porosity=phie,sw=sw,bo=bo,bg=bg,oil_fraction=oil_fraction,
                     scf_per_boe=scf_per_boe,uncertainty=uncertainty/100.)
                volume_memory[chosen]=values
                st.session_state.v10_volumes=volume_memory
                st.session_state[f"v10_last_volume_{chosen}"]=scenario
                st.success("Illustrative in-place scenario calculated from explicit assumptions")
            except Exception as exc:
                st.error(f"Volumetric inputs rejected: {type(exc).__name__}: {exc}")
        if chosen in volume_memory:
            vv=volume_memory[chosen]
            cs1,cs2,cs3=st.columns(3)
            cs1.metric("STOIIP · MMstb",f"{vv['STOIIP_MMstb']:.3f}")
            cs2.metric("GIIP · Bscf",f"{vv['GIIP_Bscf']:.3f}")
            cs3.metric("In-place · MMboe",f"{vv['Inplace_MMboe']:.3f}")
            output=st.session_state.get(f"v10_last_volume_{chosen}")
            if output:
                tab=pd.DataFrame([{ "Property":k, **v} for k,v in output.items()])
                st.dataframe(tab.round(4),hide_index=True,width="stretch")
                st.download_button("Export V10 assumed volumetric scenarios CSV",tab.to_csv(index=False),
                    f"ANGSI_{chosen}_ASSUMED_volumetrics.csv","text/csv",key=f"v10_vol_export_{chosen}")
                st.caption("P90=low, P50=median, P10=high under illustrative independent triangular inputs. "
                           "These quantiles are NOT calibrated uncertainty or reserves estimates.")
    st.download_button("Export V9 candidate ranking CSV",ranked.to_csv(index=False),
                       "angsi_v9_opportunity_ranking.csv","text/csv")
    st.download_button("Export V9 candidate polygon WKT (IL/XL)",
                       pd.DataFrame([{"ID":r["ID"],"Category":r["Category"],
                                     "Cells":r["Cells"],"Polygon_IL_XL_WKT":r["Geometry"].wkt}
                                    for r in regions]).to_csv(index=False),
                       "angsi_v9_candidate_grid_polygons.csv","text/csv")
    st.download_button("Export V9 opportunity settings / provenance",
                       json.dumps({"analysis":store["spec"],"screening":result["settings"]},default=str,indent=2),
                       "angsi_v9_opportunity_manifest.json","application/json")
    st.session_state.v9_candidate_evidence=ranked.head(5).drop(columns=[]).to_dict("records")

# ============================================================================
# V10.2 ADDITIVE PANEL: facies-weighted, multi-reservoir volume SCENARIOS
# ============================================================================
import v10_2_reservoir_engine as V102


def _v102_volume_plot(table, regions, volume_column, unit_label, *, max_bubbles=25):
    """Volume sized markers located at real candidate representative-point IL/XL.

    No field value is added; this is an OVERLAY on a spatially calibrated map.
    """
    byid={str(r['ID']):r for r in regions}
    valid=table.copy()
    valid[volume_column]=pd.to_numeric(valid[volume_column],errors='coerce')
    valid=valid[valid[volume_column].notna() & (valid[volume_column]>=0)]
    valid=valid.sort_values(volume_column,ascending=False).head(max_bubbles)
    fig=go.Figure()
    if valid.empty:return fig
    loc=[byid.get(str(r.ID),{}).get('Geometry') for r in valid.itertuples()]
    keep=[j for j,g in enumerate(loc) if g is not None]
    if not keep:return fig
    v=valid.iloc[keep]
    loc=[loc[j] for j in keep]
    vals=v[volume_column].to_numpy(float)
    vmax=max(float(vals.max()),1e-9)
    sizes=14+67*np.sqrt(np.maximum(vals,0)/vmax)
    fig.add_trace(go.Scatter(
        x=[g.representative_point().y for g in loc],
        y=[g.representative_point().x for g in loc],
        mode='markers+text',
        text=[f'{row.ID}<br>{float(getattr(row,volume_column)):.2f} {unit_label}' for row in v.itertuples()],
        textposition='top center',
        textfont=dict(color='#f1faff',size=11),
        marker=dict(size=sizes,color='#26aa61',opacity=.64,line=dict(width=2,color='#f4ffc9')),
        customdata=np.array([[str(r.Reservoir),float(getattr(r,volume_column)),float(r.Area_km2)]
                              for r in v.itertuples()],object),
        hovertemplate='%{text}<br>Reservoir %{customdata[0]}<br>Area %{customdata[2]:.3f} km²<extra></extra>',
        name='Volume-ranked scenarios'))
    fig.update_layout(template='plotly_dark',paper_bgcolor=DARK,plot_bgcolor=DARK,
        title=f'Bubble size = {volume_column} · screening SCENARIOS · not reserves',
        xaxis_title='Crossline',yaxis_title='Inline',
        height=560,margin=dict(l=50,r=25,t=60,b=45),
        font=dict(color=TEXT,size=12),showlegend=False)
    fig.update_yaxes(scaleanchor='x',scaleratio=1)
    return fig


def _v102_run_volume(regions,frame,auto_vol,store,name,phase,settings, production=None,well_qc=None):
    uniform, log_count=V102.uniform_well_parameters(well_qc,
        {'Thickness_m':settings['fallback_thickness'],'NTG':settings['fallback_ntg'],
         'PHIE':settings['fallback_phi'],'Sw':settings['fallback_sw']})
    data=V102.enrich_candidate_volumes(
        regions,frame,auto_vol,reservoir=name,phase=phase,
        property_mode=settings.get('property_mode','Uniform reservoir log medians'),
        uniform_log_params=uniform,uniform_log_count=log_count,
        fallback_thickness=settings['fallback_thickness'],
        fallback_ntg=settings['fallback_ntg'],fallback_phi=settings['fallback_phi'],
        fallback_sw=settings['fallback_sw'],bo=settings['bo'],bg=settings['bg'],
        oil_rf=settings['oil_rf'],gas_rf=settings['gas_rf'],
        gross_cap_m=settings['gross_cap'],weight_sensitivity=settings['uncertainty'],
        multipliers=settings.get('multipliers'))
    if production is not None and not data.empty:
        data=V102.remaining_with_production(data,{name:production},phase=phase)
    return data


def v102_opportunities_panel(data,store,regions,ranked,auto_vol):
    """Additive; leaves V9.3 Opportunities evidence ranking and polygons unchanged."""
    bp,mp,wd,wells,gb,gm,matched,tops=data
    with st.container(border=True):
        st.markdown('### 🧮 V10.2 · All Candidates & Multi-Reservoir Screening Volumetrics')
        st.info('**Every qualifying polygon is evaluated** over the sampled seismic footprint; wells are only property analogs. '
                'Yellow bright facies / green intermediate / grey dim are *unvalidated lithology hypotheses*. '
                'These numbers represent **conditional in-place scenarios**, not validated remaining reserves.')
        spec=store['spec']
        is_horizon=(spec.get('Mode')=='Interpreted horizon pair')
        proposed=(f"{spec.get('Top_marker')} → {spec.get('Base_marker')}" if is_horizon else
                  f"TWT {float(spec.get('Top_ms',1500)):g}–{float(spec.get('Base_ms',1700)):g} ms [NOT a defined reservoir]")
        p,q=st.columns([2,2])
        with p:reservoir=st.text_input('Current analysis interval / reservoir ID',value=proposed,key='v102_res_name')
        with q:
            phase=st.selectbox('**Fluid PHASE SCENARIO** (not a fluid test)',
                ['Gas scenario','Oil scenario','Model phase when valid','Unknown phase · alternatives'],
                index=0,key='v102_phase')
        p,q,r=st.columns(3)
        with p:
            rf_g=st.slider('ASSUMED gas recovery fraction',0.,1.,.50,.05,key='v102_gas_rf')
            fallback_thick=st.number_input('Default gross thickness if LAS unavailable (m)',.5,200.,20.,2.5,key='v102_gross')
        with q:
            rf_o=st.slider('ASSUMED oil recovery fraction',0.,1.,.25,.05,key='v102_oil_rf')
            fallback_phi=st.number_input('Default PHIE when log missing',.01,.40,.18,.01,key='v102_phi')
        with r:
            fallback_ntg=st.number_input('Default NTG when log missing',.01,1.,.45,.05,key='v102_ntg')
            fallback_sw=st.number_input('Default Sw when log missing',.01,.95,.4,.05,key='v102_sw')
        st.caption('Defaults are *visible modelling assumptions*. Existing validated local PHIE, Sw and NTG are used first; '
                   'facies multipliers are applied consistently, not tuned to a desired 25 MMscf outcome.')
        cc1,cc2,cc3=st.columns(3)
        with cc1:default_bo=st.number_input('Assumed Bo (res m³ / ST m³)',.1,5.,1.25,.05,key='v102_bo')
        with cc2:default_bg=st.number_input('Assumed Bg (res m³ / std m³)',.0001,.2,.005,.001,format='%.5f',key='v102_bg')
        with cc3:
            apply_gross_cap=st.toggle('Optional user gross-thickness cap',value=False,key='v102_thickcap_enabled',
                help='OFF by default so valid LAS-derived gross thickness is not silently reduced. A cap is a user scenario assumption, not measured geology.')
            gross_cap=(st.number_input('Maximum gross thickness (m)',.5,300.,20.,2.,key='v102_thickcap')
                        if apply_gross_cap else None)
            uncert=st.slider('Low/base/high sensitivity ±%',0,70,25,5,key='v102_uncert')/100.
        prop_mode=st.radio('Reservoir property handling',
              ['Uniform reservoir log medians','Spatial local analog per polygon'],
              index=0,horizontal=True,key='v102_property_mode',
              help='Uniform uses the SAME median measured PHIE, Sw, NTG and gross thickness for all polygons in this reservoir. Facies proportions then adjust net hydrocarbon pore volume; spatial mode permits nearby-well analog variations.')
        settings={'bo':default_bo,'bg':default_bg,'gross_cap':gross_cap,'oil_rf':rf_o,'gas_rf':rf_g,
                  'property_mode':prop_mode,
                  'fallback_thickness':fallback_thick,'fallback_phi':fallback_phi,
                  'fallback_ntg':fallback_ntg,'fallback_sw':fallback_sw,'uncertainty':uncert}
        with st.expander('Facies-dependent properties · edit multipliers, all polygons use SAME assumptions',expanded=False):
            st.caption('Modifiers applied to actual log analog when present, otherwise to displayed default values. '
                       'Sw delta uses additive saturation points, not a multiplier. NOT a trained reservoir model.')
            multipliers={}
            for name,label,defaults in [('BRIGHT_SAND_PRONE','Bright / yellow sand hypothesis',(1.10,1.15,-.08)),
                                        ('INTERMEDIATE_UNCERTAIN','Intermediate / green',(0.70,.90,.04)),
                                        ('DIM_SHALE_PRONE','Dim / grey shale hypothesis',(.15,.65,.14)),
                                        ('NO_VALID_FACIES','Unknown facies',(1.,1.,0.))]:
                cx=st.columns(3)
                vals=[]
                for column,short,default,mini,maxi in zip(cx,['NTG ×','PHIE ×','Sw offset'],defaults,[0.,0.,-.5],[2.,2.,.5]):
                    with column:vals.append(st.number_input(f'{label} {short}',min_value=mini,max_value=maxi,value=float(default),step=.05,
                        key=f'v102_fac_{name}_{short}'))
                multipliers[name]=tuple(vals)
        settings['multipliers']=multipliers
        st.markdown('#### Production history (optional, VERIFIED INPUT ONLY)')
        verified_prod=st.toggle('I have verified reservoir-level cumulative produced volume',value=False,
                               key='v102_verified_prod')
        produced=None
        if verified_prod:
            st.warning('Cumulative reservoir production does not identify which individual candidate polygons produced it. Remaining outputs use an **ILLUSTRATIVE pro-rata allocation**, not measured candidate-level depletion.')
            unit='MMscf' if phase=='Gas scenario' else 'MMstb' if phase=='Oil scenario' else None
            if unit is None:
                st.warning('Choose Oil or Gas scenario to subtract matching phase production (do not mix oil/gas units).')
            else:
                produced=st.number_input(f'Cumulative produced in this reservoir ({unit})',min_value=0.,value=0.,step=5.,
                    key='v102_prod_input')
        if st.button('🧮 CALCULATE ALL CURRENT CANDIDATE VOLUMES & BUBBLE RANKING',
                     type='primary',key='v102_calc'):
            if auto_vol is None or auto_vol.empty:
                st.error('Auto-volume inputs missing. Calculate ranked polygons and V10.1 Auto Volumetrics first; valid survey XY must be available.')
            else:
                result=_v102_run_volume(regions,frame=store['frame_amplitude_proxy'] if isinstance(store.get('frame_amplitude_proxy'),pd.DataFrame) else store['frame'],
                                  auto_vol=auto_vol,store=store,name=reservoir,phase=phase,
                                  settings=settings,production=produced,
                                  well_qc=st.session_state.get('v10_auto_well_qc'))
                runs=st.session_state.setdefault('v102_reservoir_runs',{})
                runs[reservoir]={'volumes':result,'regions':regions,'phase':phase,
                                 'spec':dict(spec),'settings':settings,'production_verified':produced is not None}
                st.session_state['v102_reservoir_runs']=runs
                st.success(f'Calculated {len(result):,} candidate scenarios for {reservoir}. '
                           'Remaining recoverable shown ONLY with verified cumulative production input.')
        runs=st.session_state.get('v102_reservoir_runs',{})
        current=runs.get(reservoir)
        if current and isinstance(current.get('volumes'),pd.DataFrame) and not current['volumes'].empty:
            tab=current['volumes']
            ranked_v=V102.volume_ranking(tab,phase=current['phase'])
            main_col={'Gas scenario':'GIIP_MMscf','Oil scenario':'STOIIP_MMstb',
                      'Model phase when valid':'Inplace_MMboe','Unknown phase · alternatives':'Oil_only_MMstb'}[current['phase']]
            unit={'GIIP_MMscf':'MMscf','STOIIP_MMstb':'MMstb','Inplace_MMboe':'MMboe',
                  'Oil_only_MMstb':'MMstb (oil-only end member)'}[main_col]
            if ranked_v.empty:
                st.warning('No quantifiable candidate; check survey XY geometry and input QC rather than forcing numbers.')
            else:
                numerical_total=ranked_v[main_col].sum()
                metrics=st.columns(3)
                metrics[0].metric(f'**Total current-interval in-place** · {unit}',f'{numerical_total:,.2f}')
                metrics[1].metric('Candidates with volumes',len(ranked_v))
                metrics[2].metric('Mapped candidate area (km²)',f'{ranked_v.Area_km2.sum():.3f}')
                st.caption('Sum is across non-overlapping classified candidate sampled cells for this ONE analysis interval; '
                           'this does NOT include the entire reservoir outside selected candidate polygons. '
                           'Different time-window/marker-pair runs may overlap; do not blindly add them.')
                showing=['Volume_rank','Reservoir','ID','Category','Area_km2',main_col,
                         'Recoverable_gas_MMscf' if current['phase']=='Gas scenario' else 'Recoverable_oil_MMstb',
                         'Facies_breakdown','Input_source','Structural_closure','Status']
                st.dataframe(ranked_v[[c for c in showing if c in ranked_v]],width='stretch',hide_index=True)
                # Add bubbles to true IL/XL position, sized by computed volume not evidence score.
                background=plot_attribute_map(store['frame'],'D_RMS',matched,stride=int(spec['Stride']),
                              label='4D RMS + volume-proportional green bubbles',show_wells=True)
                bubble=_v102_volume_plot(ranked_v,current['regions'],main_col,unit)
                for trace in bubble.data: background.add_trace(trace)
                background.update_layout(title='4D RMS map · bubble diameter reflects preliminary volume',height=650,
                                         margin=dict(l=55,r=60,t=60,b=50))
                st.plotly_chart(background,width='stretch',key='v102_volume_bubbles')
                st.markdown('##### Top 3 by calculated VOLUME (not 4D evidence score)')
                cols=st.columns(3)
                for col,(_,row) in zip(cols,ranked_v.head(3).iterrows()):
                    with col:
                        with st.container(border=True):
                            st.markdown(f'**Volume Rank #{int(row.Volume_rank)} · {row.ID}**')
                            st.metric(f'In-place screening ({unit})',f'{float(row[main_col]):,.2f}')
                            st.caption(f"Area {float(row.Area_km2):.3f} km² · {row.Facies_breakdown}")
                            st.caption(str(row.Input_source))
            st.download_button('Export all candidate volume scenarios CSV',tab.to_csv(index=False),
                f'ANGSI_{reservoir.replace("/","_").replace(" ","_")}_V102.csv','text/csv',key='v102_csv')
        # Across multiple independent configured reservoir intervals.
        st.markdown('#### All evaluated reservoir intervals · scenario totals')
        selected_tables=[v['volumes'] for v in runs.values() if isinstance(v.get('volumes'),pd.DataFrame) and not v['volumes'].empty]
        if selected_tables:
            allrows=pd.concat(selected_tables,ignore_index=True)
            totals=V102.volumes_by_reservoir(allrows)
            st.dataframe(totals.round(3),width='stretch',hide_index=True)
            st.download_button('Export ALL reservoir candidate tables CSV',allrows.to_csv(index=False),
                               'ANGSI_V102_ALL_RESERVOIR_SCREENING.csv','text/csv',key='v102_csv_all')
            if len(totals)>1:
                st.warning('**Multi-interval aggregation caution:** check marker ordering, stratigraphic overlap, '
                           'mutually exclusive reservoirs and fluid phase before summing a field total. '
                           'In-place volumes are not recoverable reserves.')
                valid, reason=V102.check_stratigraphic_disjoint_specs(runs,tops)
                all_phase={v.get('phase') for v in runs.values()}
                st.caption('Portfolio interval QC: '+reason)
                if len(all_phase)!=1:
                    st.warning('Mixed gas/oil assumptions across stored intervals; cannot display an additive field volume.')
                elif valid and st.checkbox('Show illustrative portfolio sum of NON-OVERLAPPING measured marker intervals',
                                         value=False,key='v102_portfolio_sum'):
                    current_phase=next(iter(all_phase))
                    desired={'Gas scenario':'GIIP_MMscf','Oil scenario':'STOIIP_MMstb',
                              'Model phase when valid':'Inplace_MMboe',
                              'Unknown phase · alternatives':'Oil_only_MMstb'}[current_phase]
                    if desired in totals:
                        portfolio=pd.to_numeric(totals[desired],errors='coerce').dropna()
                        if len(portfolio)==len(totals):
                            unit={'GIIP_MMscf':'MMscf','STOIIP_MMstb':'MMstb',
                                  'Inplace_MMboe':'MMboe','Oil_only_MMstb':'MMstb (oil-only alternative)'}[desired]
                            st.metric(f'ILLUSTRATIVE total across {len(totals)} NON-OVERLAPPING stratigraphic intervals ({unit})',
                                      f'{float(portfolio.sum()):,.2f}')
                            st.caption('Scenario sum only — 4D anomaly footprints are not geological closures; '
                                       'net reservoir, hydrocarbon contacts and economic recoverability remain unvalidated.')
                        else:st.warning('Some intervals lack quantifiable, consistent phase estimates; total suppressed.')
                elif not valid:
                    st.caption('Field sum is intentionally blocked until mutually exclusive reservoir intervals are verified.')
        else:
            st.caption('Run at least one reservoir interval above. Multi-zone batch calculation available below.')
        with st.expander('🚀 Batch run MULTIPLE interpreted reservoir intervals (full paired cube sampling)',expanded=False):
            # Every pair is merely a consecutive measured stratigraphic boundary interval;
            # it is not automatically proven hydrocarbon-bearing reservoir.
            valid_tops=tops.loc[tops.TWT_ms.notna(),['Marker','TWT_ms']].copy()
            median=(valid_tops.groupby('Marker').TWT_ms.agg(['median','count'])
                    .query('count >= 2').sort_values('median'))
            boundary_names=median.index.tolist()
            pairs=[(a,b) for a,b in zip(boundary_names[:-1],boundary_names[1:])]
            lookup={f'{a} → {b}':(a,b) for a,b in pairs}
            typical=[k for k in lookup if any(x in k.upper() for x in ('I-25','I-35','I-68'))]
            choose_all=st.toggle('Select ALL eligible adjacent measured-marker intervals',value=False,key='v102_batch_all')
            chosen=(list(lookup) if choose_all else st.multiselect(
                 'Interpretation intervals (adjacent measured markers; verify each is a reservoir)',
                 list(lookup),default=typical[:3] if typical else list(lookup)[:2],key='v102_batch_pairs'))
            st.caption(f'{len(chosen)} selected stratigraphic interval(s); some may be non-reservoir units.')
            batch_stride=st.radio('Batch IL/XL sample step',[5,10,15],index=1,horizontal=True,key='v102_batch_stride')
            st.warning('Batch will read SEG-Y for EVERY selected marker pair; expensive on HPC. '
                       'Each pair is a *candidate stratigraphic interval*, not necessarily net sand or an independent accumulation. '
                       'Horizon trend-only picks are uncertain. Use conservative QC.')
            if st.button('RUN SELECTED INTERVALS · whole paired survey',key='v102_batch_run',disabled=not chosen):
                from survey_analysis import sample_pairs
                import analysis_engine as E
                import v10_auto_volumetrics as V10A
                last_surface_qc=st.session_state.get('v92_surface_qc')
                output=st.session_state.setdefault('v102_reservoir_runs',{})
                prog=st.progress(0.,text='Extracting interpreted intervals…')
                results=[]
                try:
                    for i,name in enumerate(chosen):
                        t,b=lookup[name]
                        try:
                            # Use exactly the same well-guided AI horizon surfaces as Analysis.
                            # Full trace coverage, not circular well buffers.
                            if surface_loader is None: raise RuntimeError('AI horizon surface loader not connected')
                            a,z,diag=_make_surface_pair(data,surface_loader,t,b,batch_stride)
                            pairs_df=sample_pairs(gb,gm,batch_stride)
                            f,qc=E.compute_attribute_volume(bp,mp,pairs_df,bounds=(1500.,1700.),
                                    top_map=a,base_map=z,reference=(700.,950.),use_correction=False)
                            if f.empty: raise ValueError('No valid paired samples within interpreted interval')
                            f['Horizon_support']=[st.session_state.get('v92_surface_qc',{}).get((int(il),int(xl)),'MISSING')
                                                   for il,xl in zip(f.IL,f.XL)]
                            f,fac=E.amplitude_facies_proxy(f,low_percentile=30.,high_percentile=70.,attribute='B_RMS')
                            regions_batch=E.candidate_regions(f,batch_stride,low=.20,high=.80,
                                               min_cells=5,metric='Raw_RMS_percentile')
                            if not regions_batch:raise ValueError('No connected 4D screening polygons at default thresholds')
                            batch_spec=dict(Mode='Interpreted horizon pair',Top_marker=t,Base_marker=b,Stride=batch_stride,
                                            Top_ms=1500.,Base_ms=1700.)
                            auto,batch_well_qc,_=V10A.auto_candidate_volumes(regions_batch,gb,wells,wd,tops,batch_spec,
                                          bo=default_bo,bg=default_bg,attribute_frame=f)
                            tab=_v102_run_volume(regions_batch,frame=f,auto_vol=auto,store={'spec':batch_spec},
                                               name=name,phase=phase,settings=settings,well_qc=batch_well_qc)
                            output[name]={'volumes':tab,'regions':regions_batch,'phase':phase,
                                          'spec':batch_spec,'settings':settings,'production_verified':False}
                            results.append((name,'DONE',len(tab)))
                        except Exception as exc:
                            results.append((name,type(exc).__name__,str(exc)[:160]))
                        prog.progress((i+1)/len(chosen),text=f'Completed {i+1} / {len(chosen)} interval(s)')
                    st.session_state.v102_reservoir_runs=output
                    st.dataframe(pd.DataFrame(results,columns=['Interval','Status','Candidates or diagnostic']),hide_index=True)
                finally:
                    if last_surface_qc is not None:st.session_state['v92_surface_qc']=last_surface_qc
                    else:st.session_state.pop('v92_surface_qc',None)
