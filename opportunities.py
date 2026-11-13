"""Screen connected survey cells and score available independent evidence."""
from hashlib import sha256
import numpy as np
import pandas as pd
from scipy.ndimage import label
from shapely.geometry import box, mapping
from shapely import transform
from shapely.ops import unary_union

COMPONENTS = ("Reservoir", "Structure", "Seismic", "Proximity", "Horizon", "Production")


def regions(frame, stride, low=.7, high=1.6, minimum=4):
    if frame.empty:
        return []
    il = np.arange(frame.IL.min(), frame.IL.max()+1, stride)
    xl = np.arange(frame.XL.min(), frame.XL.max()+1, stride)
    grid = frame.set_index(["IL", "XL"]).reindex(pd.MultiIndex.from_product([il, xl], names=["IL","XL"]))
    eligible = ((grid.status == "PASS") & (grid.raw_valid_fraction >= .8)
                & (grid.reference_valid_fraction >= .8))
    contrast = grid.local_repeatability_contrast
    candidates = []
    for category, selected in (("Low change", contrast <= low), ("High contrast", contrast >= high)):
        groups, n = label((eligible & selected).to_numpy().reshape(len(il), len(xl)))
        for group in range(1, n+1):
            iy, ix = np.where(groups == group)
            if len(iy) < minimum:
                continue
            cells = list(zip(il[iy], xl[ix]))
            subset = grid.loc[cells].reset_index()
            subset.attrs["stride"] = stride
            signature = f"{category}:{stride}:"+repr([(int(y), int(x)) for y,x in cells])
            identifier = "C-"+sha256(signature.encode()).hexdigest()[:10].upper()
            half = stride/2
            polygon = unary_union([box(x-half, y-half, x+half, y+half) for y,x in cells])
            candidates.append((identifier, category, subset, polygon))
    return candidates


def projected_geometry(polygon, geometry, epsg, tolerance=5):
    """Fit a verified affine index transform and check residuals before reporting area."""
    from pyproj import CRS, Transformer
    crs = CRS.from_epsg(epsg)
    if not crs.is_projected or any(abs(a.unit_conversion_factor-1) > 1e-9 for a in crs.axis_info[:2]):
        raise ValueError("Area requires a projected metre CRS")
    xy = geometry[["X", "Y"]].to_numpy(float)
    valid = np.isfinite(xy).all(axis=1) & (np.abs(xy).sum(axis=1) > 0)
    samples = geometry.loc[valid, ["XL", "IL"]].to_numpy(float)
    design = np.c_[samples, np.ones(len(samples))]
    fit, _, rank, _ = np.linalg.lstsq(design, xy[valid], rcond=None)
    if rank < 3 or abs(np.linalg.det(fit[:2])) < 1e-9:
        raise ValueError("Geometry lacks 2D affine support")
    if np.max(np.linalg.norm(design @ fit-xy[valid], axis=1)) > tolerance:
        raise ValueError("XY residual exceeds affine tolerance")
    projected = transform(polygon, lambda coords: coords @ fit[:2] + fit[2])
    lonlat = transform(projected, Transformer.from_crs(crs, 4326, always_xy=True).transform,
                       interleaved=False)
    if lonlat.is_empty or not np.isfinite(lonlat.bounds).all():
        raise ValueError("Coordinate transform failed")
    from shapely.geometry.polygon import orient
    return projected.area/1e6, mapping(orient(lonlat, sign=1))


def rank(table, weights):
    weights = np.asarray(weights, float)
    if weights.shape != (len(COMPONENTS),) or np.any(weights < 0) or not np.isclose(weights.sum(), 100):
        raise ValueError("Six nonnegative weights must sum to 100%")
    scores = table.loc[:, COMPONENTS].to_numpy(float)
    have = np.isfinite(scores)
    if np.any(have & ((scores < 0) | (scores > 1))):
        raise ValueError("Component scores must be in [0,1]")
    denominator = have @ weights
    total = np.where(have, scores, 0) @ weights
    return pd.DataFrame({"Score": np.divide(100*total, denominator,
                        out=np.full(len(scores), np.nan), where=denominator > 0),
                         "Evidence_weight_pct": denominator}, index=table.index)

