"""ANGSI JR V10 presentation and seismic geometry helpers.

No seismic data is invented. All categorical displays use the existing sampled
IL/XL keys; gaps are never interpolated or filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

DIM = '#879099'          # dim / shale-prone hypothesis
INTERMEDIATE = '#28ab66' # uncertain intermediate (green)
BRIGHT = '#f6d349'       # bright / sand-prone hypothesis (yellow)
OIL = '#1aa15d'
GAS = '#e34242'


def facies_colorscale():
    # Codes 0=dim, 1=intermediate, 2=bright, zmin=-0.5, zmax=2.5
    return [[0.0, DIM], [1/3 - 1e-5, DIM],
            [1/3, INTERMEDIATE], [2/3 - 1e-5, INTERMEDIATE],
            [2/3, BRIGHT], [1.0, BRIGHT]]


def fluid_colorscale():
    # Categories 0=Other,1=Water,2=HC,3=Oil,4=Gas. All but oil/gas transparent.
    # Numeric category raster is only used where fluid == Oil/Gas.
    return [[0, OIL], [.499999, OIL], [.5, GAS], [1., GAS]]


def fluid_only_grid(depth, fluid, top, bottom, rows=650, columns=12):
    """Transparent NaN background; only explicitly classified oil/gas samples fill."""
    out = np.full((rows, columns), np.nan, float)
    zz = np.asarray(depth, float)
    fl = np.asarray(fluid, object)
    if len(zz) < 2 or len(fl) != len(zz):
        return out
    order = np.argsort(zz)
    zz, fl = zz[order], fl[order]
    yy = np.linspace(top, bottom, rows)
    ii = np.searchsorted(zz, yy).clip(0, len(zz) - 1)
    prev = (ii - 1).clip(0, len(zz) - 1)
    dd = np.minimum(np.abs(yy-zz[ii]), np.abs(yy-zz[prev]))
    valid = (yy >= zz[0]) & (yy <= zz[-1]) & (dd <= max((bottom-top)/rows*1.6,2.))
    # closest side of depth sample, not always the higher index
    closest = np.where(np.abs(yy-zz[prev]) < np.abs(yy-zz[ii]), prev, ii)
    for y in np.flatnonzero(valid):
        value = str(fl[closest[y]])
        if value == 'Oil-model': out[y, :] = 0.0
        if value == 'Gas-model': out[y, :] = 1.0
    return out


def extend_route_xy(points_xy, extension_m=1250.0):
    """Extend a 2D well polyline beyond BOTH endpoints, return new route + original stations.

    Exact original well positions are preserved; only route endpoints change.
    """
    xy = np.asarray(points_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError('At least two well XY positions are required')
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    valid = np.flatnonzero(segment > 1e-8)
    if not len(valid):
        raise ValueError('All well locations coincide')
    dist = float(extension_m)
    if dist < 0 or not np.isfinite(dist):
        raise ValueError('Extension must be nonnegative in metres')
    first = np.diff(xy, axis=0)[valid[0]] / segment[valid[0]]
    last = np.diff(xy, axis=0)[valid[-1]] / segment[valid[-1]]
    extended = np.vstack([xy[0]-dist*first, xy, xy[-1]+dist*last])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(extended, axis=0), axis=1))]
    return extended, cum[1:-1] / 1000.


def affine_survey_xy(base_grid, max_points=25000):
    """Fit XY = A @ (IL,XL,1), verify residual vs bin dimensions.

    This is a geometry check, NOT a declaration of coordinate reference system.
    """
    sub = base_grid[['IL','XL','X','Y']].replace([np.inf,-np.inf],np.nan).dropna()
    sub = sub[(sub.X.abs()>10000) & (sub.Y.abs()>10000)]
    if len(sub) < 25:
        return {'valid':False,'reason':'Not enough physically valid survey XY headers'}
    if len(sub) > max_points:
        sub = sub.iloc[np.linspace(0,len(sub)-1,max_points).astype(int)]
    A = np.column_stack([sub.IL.to_numpy(float), sub.XL.to_numpy(float),
                         np.ones(len(sub))])
    xy = sub[['X','Y']].to_numpy(float)
    coeff, _, rank, _ = np.linalg.lstsq(A,xy,rcond=None)
    if rank < 3:
        return {'valid':False,'reason':'Degenerate survey IL/XL geometry'}
    residuals = np.linalg.norm(A @ coeff - xy,axis=1)
    rms = float(np.sqrt(np.mean(residuals**2)))
    J = coeff[:2, :]
    area = abs(float(np.linalg.det(J)))
    bin_size = min(float(np.linalg.norm(J[0])),float(np.linalg.norm(J[1])))
    trusted = bool(np.isfinite(area) and area > 0 and rms < max(20., 2.0*bin_size))
    return {'valid':trusted,'reason':'Survey XY affine consistency check' if trusted else 'Spatial geometry residual too large',
            'xy_per_grid':coeff,'rmse_m':rms,'cell_area_m2':area,
            'nominal_bin_m':bin_size,'sampled_headers':len(sub)}


def candidate_areas_km2(regions, base_grid, spacing):
    """Grid-cell footprint areas; never assume anomaly polygon equals hydrocarbon closure."""
    fit = affine_survey_xy(base_grid)
    areas={r['ID']: float(r['Cells'])*float(spacing)**2 *fit['cell_area_m2']/1e6
           for r in regions} if fit.get('valid') else {}
    return areas, fit


def map_bubble_layer(fig, ranked, regions, *, max_labels=10, areas=None, scale_by='Area'):
    """Add candidate centroids with size controlled by physical or sampled area."""
    by_id = {r['ID']:r for r in regions}
    df = ranked.head(max_labels).copy()
    if df.empty:return fig
    coords=[]
    for r in df.itertuples():
        geom = by_id[r.ID]['Geometry']
        p = geom.representative_point()
        area = (areas or {}).get(r.ID,np.nan)
        metric = area if scale_by=='Area' and np.isfinite(area) else float(r.Cells)
        coords.append((r.ID, p.y, p.x, max(metric,1e-9),r.Category))
    v=np.asarray([c[3] for c in coords]); v=np.sqrt(v/np.nanmax(v))
    sizes = 13+31*v
    fig.add_trace(go.Scatter(
        x=[c[1] for c in coords],y=[c[2] for c in coords],mode='markers+text',
        text=[c[0] for c in coords],textposition='middle center',
        textfont=dict(size=9,color='#ffffff'),
        marker=dict(size=sizes,color=['#f8b952' if c[4].startswith('Higher') else '#36c6ac' for c in coords],
                    opacity=.30,line=dict(width=2,color='#f7faff')),
        customdata=np.array([[c[4],c[3]] for c in coords],dtype=object),
        hovertemplate='%{text}<br>%{customdata[0]}<br>Relative size %{customdata[1]:.3g}<extra></extra>',
        name='Top candidate bubbles'))
    return fig
