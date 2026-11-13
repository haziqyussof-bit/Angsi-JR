"""Two-panel, QC-only seismic-synthetic tie display for ANGSI JR V10."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def normalized(v,mask):
    values=np.asarray(v,float)
    good=np.asarray(mask,bool)&np.isfinite(values)
    out=np.full_like(values,np.nan)
    if good.sum()>1:
        scale=float(np.sqrt(np.mean(values[good]**2)))
        if scale>0: out[good]=values[good]/scale
    return out


def shifted_model(t,model,shift_ms):
    t=np.asarray(t,float)
    mod=np.asarray(model,float)
    good=np.isfinite(t)&np.isfinite(mod)
    out=np.full_like(mod,np.nan)
    if good.sum()>=3:
        out=np.interp(t-float(shift_ms),t[good],mod[good],left=np.nan,right=np.nan)
    return out


def qc_trace_figure(time,trace,model,mask,marker_times=None):
    good=np.asarray(mask,bool)&np.isfinite(trace)&np.isfinite(model)
    baseline=normalized(trace,good); synth=normalized(model,good)
    fig=go.Figure()
    fig.add_trace(go.Scattergl(x=baseline,y=time,mode='lines',name='1995 Baseline',
                               line=dict(color='#637afc',width=1.3)))
    fig.add_trace(go.Scattergl(x=synth,y=time,mode='lines',name='Ricker Synthetic',
                               line=dict(color='#ff5948',width=1.5)))
    if marker_times is not None:
        for name,t in marker_times:
            if np.isfinite(t):
                fig.add_hline(y=float(t),line=dict(color='#e6b84f',width=1,dash='dot'),
                              annotation_text=str(name),annotation_font_color='#f7e5ab')
    fig.update_layout(template='plotly_dark',height=680,
        paper_bgcolor='#061e31',plot_bgcolor='#061e31',
        legend=dict(orientation='h',x=.5,y=1.12,xanchor='center',yanchor='bottom',
                    bgcolor='rgba(6,30,49,.9)',font=dict(color='white',size=10)),
        margin=dict(l=46,r=20,t=105,b=55),
        title=dict(text='Panel 1 · Actual trace vs synthetic',x=.5,font=dict(size=14)),
        xaxis_title='Normalized amplitude',
        yaxis=dict(title='TWT (ms)',autorange='reversed'),
        dragmode='pan')
    return fig


def seismic_tie_figure(time,xl,section,well_xl,model,mask,wiggle_scale=5.,polarity=1,
                       marker_times=None):
    """Displays actual inline seismic and synthetic wiggle at the well trace.

    Synthetic has its own wiggle scale in crossline units and no shift of seismic XY.
    """
    time=np.asarray(time,float)
    xl=np.asarray(xl,float)
    z=np.asarray(section,float)
    if z.ndim != 2 or z.shape != (len(time),len(xl)):
        raise ValueError('Seismic section must have shape [sample,crossline]')
    if not len(xl):raise ValueError('Empty inline window')
    finite=np.abs(z[np.isfinite(z)])
    lim=float(np.nanpercentile(finite,98)) if len(finite) else 1.
    lim=max(lim,1e-9)
    plot=go.Figure(go.Heatmap(x=xl,y=time,z=z,zmin=-lim,zmax=lim,
            colorscale='RdBu_r',connectgaps=False,showscale=False,
            hovertemplate='XL %{x}<br>TWT %{y:.1f} ms<br>Amp %{z:.2g}<extra></extra>',name='1995 Seismic'))
    valid=np.asarray(mask,bool)&np.isfinite(model)
    wave=normalized(model,valid)
    wave=np.clip(wave,-2.5,2.5)*float(wiggle_scale)*int(polarity)
    well=float(well_xl)
    y=time[valid]
    a=wave[valid]
    if len(y)>3:
        # Positive lobe variable-area fill to the RIGHT of the well trajectory
        peak=np.clip(a,0,None)
        plot.add_trace(go.Scatter(x=np.r_[np.full(len(y),well), (well+peak)[::-1]],
              y=np.r_[y,y[::-1]],fill='toself',fillcolor='rgba(239,65,51,.36)',
              line=dict(width=0),name='Positive synthetic variable area',showlegend=False,hoverinfo='skip'))
        plot.add_trace(go.Scatter(x=well+a,y=y,mode='lines',name='Synthetic wiggle',
              line=dict(color='#ff5c49',width=2)))
    plot.add_vline(x=well,line_color='#ffdc47',line_dash='dot',line_width=1.3)
    for name,t in (marker_times or []):
        if np.isfinite(t):
            plot.add_hline(y=float(t),line_color='#ffe29d',line_dash='dot',line_width=.6,
                          annotation_text=str(name),annotation_font_color='#ffe29d')
    span = max(40.,float(np.ptp(xl)))
    plot.update_layout(template='plotly_dark',height=680,
        paper_bgcolor='#061e31',plot_bgcolor='#061e31',
        title=dict(text='Panel 2 · Synthetic wiggle on 1995 seismic',x=.5,font=dict(size=14)),
        legend=dict(orientation='h',x=.5,y=1.12,xanchor='center',yanchor='bottom',
                    bgcolor='rgba(6,30,49,.9)',font=dict(color='white',size=10)),
        xaxis=dict(title='Crossline',range=[min(xl.min(),well)-span*.04,max(xl.max(),well)+span*.04]),
        yaxis=dict(title='TWT (ms)',autorange='reversed'),
        margin=dict(l=48,r=25,t=105,b=55),dragmode='pan')
    return plot
