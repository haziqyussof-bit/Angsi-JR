"""Field-facing review of well controls and seismic reflector sensitivity."""
from ui_help import help_text

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import angsi_core as C
import v10_tie_viewer as V10T
from interpretation import marker_qc, reflector, synthetic
from seismic_qc import sample_mask
from survey_analysis import VERSION, file_stamp
from v8_dashboard import export, manifest


def _control_rows(tops, matched, marker):
    rows = tops[(tops.Marker == marker) & (tops.Checkshot_Donor == tops.Well) &
                tops.TWT_ms.notna() & ~tops.Depth_Source.str.contains("assumed", na=False)]
    rows = rows.merge(matched, on="Well")
    rows = rows[(rows.Distance_m <= 100) & rows.Monitor_Index.notna()]
    return rows[~rows.duplicated("Well", keep=False)]


def validation(bp, controls, radius, polarity):
    records = []
    with C.segyio.open(str(bp), "r", strict=False) as survey:
        time = np.asarray(survey.samples, float)
        for well in controls.itertuples():
            others = controls[controls.Well != well.Well]
            if len(others) < 2:
                continue
            seed, nearest = C.trend_surface([[well.IL, well.XL]], others[["IL","XL"]],
                                            others.TWT_ms)
            trace = np.asarray(survey.trace[int(well.Base_Index)], float)
            trace[~sample_mask(trace)] = np.nan
            pick, margin, _, _ = reflector(time, trace[:, None], seed, radius, polarity, 0)
            alternative, _, _, _ = reflector(time, trace[:, None], seed, radius*1.5, polarity, 0)
            records.append(dict(Well=well.Well, Marker=well.Marker, Measured_tie_ms=well.TWT_ms,
                                Heldout_trend_ms=seed[0], Heldout_pick_ms=pick[0],
                                Residual_ms=pick[0]-well.TWT_ms, Peak_score_margin=margin[0],
                                Search_window_span_ms=abs(pick[0]-alternative[0]),
                                Nearest_control_grid_units=nearest[0],
                                Status="REVIEW" if not np.isfinite(pick[0]) or
                                abs(pick[0]-well.TWT_ms)>radius/2 else "PASS"))
    return pd.DataFrame(records)


def review(data, cs_loader, log_loader):
    bp, mp, wd, wells, gb, gm, matched, tops = data
    with st.expander("Marker validation and seismic-to-well tie"):
        well = st.selectbox("Review well", wells.Well.tolist(), key="v8-tie-well", help=help_text("Review well"))
        local = tops[tops.Well == well]
        table = marker_qc(local, lambda _: cs_loader(well, wd, wells)[0])
        st.dataframe(table, hide_index=True)
        export("Export marker QC", table, "angsi_marker_qc.csv")
        st.caption("A checkshot round trip checks interpolation consistency. Depth accuracy depends on datum, trajectory and checkshot coverage.")
        st.session_state.v8_horizon_evidence = {"marker_checks": table.to_dict("records")}
        # A text box with an unknown mnemonic previously disabled the button.
        # Display ACTUAL LAS curve options and a diagnostic table first.
        try:
            log=log_loader(str(wd),well)
        except (OSError,ValueError,KeyError) as exc:
            st.error(f"Cannot open {well} LAS for synthetic tie: {exc}")
            log=pd.DataFrame()
        mnemonics=[str(c) for c in log.columns[1:]] if not log.empty else []
        density_options=[c for c in mnemonics if c.upper() in ("DENS_P","RHOB","RHOZ","DENS","RHOB_P")]
        sonic_candidates=[c for c in mnemonics if (c.upper().startswith("DT") or
                          c.upper() in ("AC","AC_P","SONIC","DTC","DTP","DTCO"))]
        st.caption("**LAS curve inventory:** "+(", ".join(mnemonics) if mnemonics else "No LAS curves available"))
        with st.expander("Preview actual well logs & QC",expanded=False):
            if len(log):
                depth_values=pd.to_numeric(log.iloc[:,0],errors="coerce").to_numpy(float)-float(wells.set_index("Well").loc[well,"RT"])
                subset=sonic_candidates+density_options+[c for c in ("GR_P","RT_P") if c in mnemonics]
                qc_rows=[]
                for key in dict.fromkeys(subset):
                    raw=pd.to_numeric(log[key],errors="coerce").to_numpy(float)
                    finite=raw[np.isfinite(raw)]
                    qc_rows.append({"Curve":key,"Valid samples":len(finite),"Missing %":100*(1-len(finite)/max(1,len(raw))),
                                    "P05":np.percentile(finite,5) if len(finite) else np.nan,
                                    "Median":np.median(finite) if len(finite) else np.nan,
                                    "P95":np.percentile(finite,95) if len(finite) else np.nan})
                st.dataframe(pd.DataFrame(qc_rows),hide_index=True,width="stretch")
                if subset:
                    figlog=go.Figure()
                    for key in dict.fromkeys(subset[:4]):
                        figlog.add_trace(go.Scattergl(x=pd.to_numeric(log[key],errors="coerce"),
                            y=depth_values,mode="lines",name=key))
                    figlog.update_layout(height=390,template="plotly_dark",yaxis=dict(title="TVDSS (m)",autorange="reversed"),
                        xaxis_title="Curve values (different units; QC only)")
                    st.plotly_chart(figlog,width="stretch")
        if not sonic_candidates:
            st.warning("No obvious sonic mnemonic (DT/AC/DTC) was found in this LAS. A true synthetic cannot be made from GR, RHOB or PHIE alone. You can manually review available curves below if the sonic is stored under another name.")
        sonic_options=["(no sonic selected)"]+sonic_candidates+[v for v in mnemonics if v not in sonic_candidates]
        sonic=st.selectbox("Compressional sonic curve · choose real LAS mnemonic",sonic_options,index=1 if sonic_candidates else 0,key="v92_sonic_mnemonic")
        density=st.selectbox("Bulk density curve",density_options or ["(missing density)"],key="v92_density_mnemonic")
        sonic_unit=st.selectbox("Confirm sonic unit",["µs/ft","µs/m","s/m"],key="v92_sonic_unit")
        frequency=st.number_input("Ricker peak frequency (Hz)",5.,100.,30.,key="v92_ricker_frequency",help=help_text("Ricker peak frequency (Hz)"))
        cs, donor, _, status = cs_loader(well,wd,wells)
        tie = matched.set_index("Well").loc[well] if well in set(matched.Well) else None
        why=[]
        if sonic.startswith("("):why.append("Select a real sonic curve")
        if density.startswith("("):why.append("Density curve unavailable")
        if donor!=well or len(cs)<2:why.append("Own checkshot required")
        if tie is None or tie.Distance_m>100 or not np.isfinite(tie.Base_Index):why.append("No nearby baseline seismic tie")
        if why:st.info("Synthetic tie prerequisites: "+" · ".join(why))
        if st.button("Calculate synthetic seismic tie",disabled=bool(why),key="v92_run_synthetic"):
            try:
                depth=pd.to_numeric(log.iloc[:,0],errors="coerce").to_numpy(float)-float(wells.set_index("Well").loc[well,"RT"])
                dt=pd.to_numeric(log[sonic],errors="coerce").to_numpy(float)
                rho=pd.to_numeric(log[density],errors="coerce").to_numpy(float)
                if sonic_unit=="µs/ft":
                    dt=dt*1e-6/0.3048
                elif sonic_unit=="µs/m":
                    dt=dt*1e-6
                # otherwise s/m
                from scipy.ndimage import label
                mask=np.isfinite(depth+dt+rho)&(dt>0)&(dt<.004)&(rho>1)&(rho<4)
                seg,num=label(mask)
                counts=np.bincount(seg)
                counts[0]=0
                if counts.max()<10:raise ValueError("No continuous positive sonic/density segment with plausible units; inspect LAS units and gaps.")
                keep=seg==counts.argmax()
                dep,slow,den=depth[keep],dt[keep],rho[keep]
                order=np.argsort(dep);dep,slow,den=dep[order],slow[order],den[order]
                if np.any(np.diff(dep)<=0):raise ValueError("Duplicated depth samples in sonic/density overlap")
                with C.segyio.open(str(bp),"r",strict=False) as survey:
                    time=np.asarray(survey.samples,float)
                    trace=np.asarray(survey.trace[int(tie.Base_Index)],float)
                model=synthetic(time,dep,slow,den,cs.TVDSS_m,cs.TWT_ms,frequency)
                common=np.isfinite(model+trace)&sample_mask(trace)
                if common.sum()<8 or np.std(model[common])<=0 or np.std(trace[common])<=0:
                    raise ValueError("Insufficient common live seismic-synthetic signal in the calibrated interval")
                st.session_state["v10_synthetic_result"] = dict(
                    well=well,time=time,trace=trace,model=model,
                    common=common,il=int(tie.IL),xl=int(tie.XL),
                    sonic=sonic,density=density,frequency=frequency,unit=sonic_unit,
                    selected_marker_times=[(r.Marker,float(r.TWT_ms)) for r in local.itertuples()
                        if np.isfinite(r.TWT_ms)],source_file=str(bp))
            except (KeyError,ValueError,OSError,TypeError) as exc:
                st.error(f"Synthetic tie QC failed: {exc}")
    saved=st.session_state.get("v10_synthetic_result")
    if saved is not None:
        st.markdown("#### Seismic-to-well tie · two synchronized TWT views")
        if saved["well"] != well:
            st.info(f"Showing last calculated synthetic for {saved['well']}; run Calculate for {well} to update.")
        ctl1,ctl2,ctl3=st.columns(3)
        with ctl1: scale=st.slider("Synthetic wiggle scale (XL)",1.,15.,5.,.5,key="v10_wiggle_scale")
        with ctl2: shift=st.slider("Display-only synthetic time shift (ms)",-30,30,0,2,key="v10_synthetic_shift")
        with ctl3: pol=st.radio("Synthetic display polarity",["Normal","Reverse"],horizontal=True,key="v10_syn_polarity")
        time=saved["time"]
        shifted=V10T.shifted_model(time,saved["model"],shift)
        common=np.asarray(saved["common"],bool)&np.isfinite(shifted)
        if common.sum()>6 and np.std(shifted[common])>0 and np.std(saved["trace"][common])>0:
            corr=float(np.corrcoef(saved["trace"][common],shifted[common])[0,1])
            st.metric("Signed correlation (with display shift)",f"{corr:.3f}")
        left,right=st.columns([1,1.3])
        with left:
            st.plotly_chart(V10T.qc_trace_figure(time,saved["trace"],shifted,common,
                saved["selected_marker_times"]),width="stretch",key="v10_tie_trace")
        with right:
            try:
                section=C.coverage_pairs(data[4],data[5],"Inline",saved["il"])
                section=section.loc[section.XL.between(saved["xl"]-75,saved["xl"]+75)]
                if section.empty:raise ValueError("No paired inline traces close to well")
                # Keep a high enough trace density but never invent outside-coverage traces.
                real=C.read_paired_section(bp,mp,section,125)
                pairs,t,bs,_=real
                fig=V10T.seismic_tie_figure(t,pairs.XL.to_numpy(float),bs,saved["xl"],shifted,common,
                    wiggle_scale=scale,polarity=1 if pol=="Normal" else -1,
                    marker_times=saved["selected_marker_times"])
                st.plotly_chart(fig,width="stretch",key="v10_tie_seismic")
            except Exception as exc:
                st.warning(f"Seismic section for synthetic overlay unavailable: {type(exc).__name__}: {exc}")
        st.caption(f"{saved['well']} · LAS sonic {saved['sonic']} ({saved['unit']}) · density {saved['density']} · "
                   "own checkshot. Time shift and polarity are DISPLAY/QC settings, not automatic calibration. "
                   "Both panels use TWT (ms); legend stays above figures.")
        export("Export synthetic tie CSV",pd.DataFrame({"TWT_ms":time,"synthetic":saved["model"],
               "baseline":saved["trace"],"common":saved["common"]}),"angsi_v10_synthetic_tie.csv")
        manifest("angsi_synthetic_tie_v10.json",input=file_stamp(bp),well=saved["well"],
                 sonic=saved["sonic"],density=saved["density"],frequency_hz=saved["frequency"],
                 wavelet="Ricker",display_shift_ms=shift,wiggle_scale=scale)
    with st.expander("Leave-one-well-out reflector validation"):
        names = sorted(tops.Marker.unique())
        if not names:
            st.info("No provided markers.")
            return
        marker = st.selectbox("Validation marker", names, help=help_text("Validation marker"))
        radius = st.slider("Reflector search radius (ms)", 8, 50, 22, help=help_text("Reflector search radius (ms)"))
        polarity = st.selectbox("Expected reflector polarity", ["Either", "Peak", "Trough"], help=help_text("Expected reflector polarity"))
        control = _control_rows(tops, matched, marker)
        st.caption("Each validation leaves the selected well out of the structural trend. At least three own-checkshot controls are required.")
        if st.button("Validate marker", disabled=len(control)<3, help=help_text("Validate marker")):
            report = validation(bp, control, radius, {"Either": 0, "Peak": 1, "Trough": -1}[polarity])
            st.dataframe(report, hide_index=True)
            st.session_state.v8_horizon_evidence["cross_validation"] = report.to_dict("records")
            export("Export held-out well QC", report, "angsi_heldout_wells.csv")
            manifest("angsi_heldout_wells.json", input=file_stamp(bp), marker=marker, radius=radius, polarity=polarity)


def surface_diagnostics(surface, search_ms):
    """Detect abrupt relief and quantify tracking-window/smoothing sensitivity."""
    base = surface["base"]
    step_il = np.abs(np.diff(base, axis=0, prepend=np.nan))
    step_xl = np.abs(np.diff(base, axis=1, prepend=np.nan))
    relief = np.fmax(step_il, step_xl)
    ambiguous = surface.get("sensitivity", np.full_like(base, np.nan))
    st.caption(f"Review flags use >{search_ms:g} ms between sampled neighbors. The envelope is the "
               "range across tested search/smoothing settings, in TWT.")
    frame = pd.DataFrame({"IL": surface["IL"].ravel(), "XL": surface["XL"].ravel(),
                          "Baseline_TWT_ms": base.ravel(), "Maximum_neighbor_step_ms": relief.ravel(),
                          "Review_discontinuity": (relief>search_ms).ravel(),
                          "Envelope_min_ms": surface.get("envelope_low", base).ravel(),
                          "Envelope_max_ms": surface.get("envelope_high", base).ravel(),
                          "Sensitivity_ms": ambiguous.ravel()})
    frame = frame[frame.Baseline_TWT_ms.notna()]
    st.dataframe(frame.nlargest(20, "Maximum_neighbor_step_ms"), hide_index=True)
    return frame

