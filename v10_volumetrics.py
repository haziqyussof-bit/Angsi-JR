"""ANGSI JR V10 prospective in-place volumetrics — scenario ONLY.

Not a petroleum reserves calculator; requires explicit assumptions/closure validation.
All gas quantities use Bg in reservoir m^3 / standard m^3 and BOE 6 Mscf/BOE default.
"""
from __future__ import annotations

import numpy as np

STB_PER_M3 = 6.289810770432105
SCF_PER_SM3 = 35.3146667215


def validate_inputs(area_km2, thickness_m, ntg, porosity, sw, bo, bg, oil_fraction=1.):
    values = (area_km2, thickness_m, ntg, porosity, sw, bo, bg, oil_fraction)
    if not np.isfinite(values).all():raise ValueError('Volumetric inputs must be finite')
    if area_km2 <= 0 or thickness_m <= 0 or bo <= 0 or bg <= 0:
        raise ValueError('Area, thickness, Bo and Bg must be strictly positive')
    if not all(0 <= x <= 1 for x in (ntg, porosity, sw, oil_fraction)):
        raise ValueError('NTG, PHIE, Sw and oil fraction must be in [0,1]')


def in_place(area_km2, thickness_m, ntg, porosity, sw,
             bo=1.25, bg=.005, oil_fraction=1., scf_per_boe=6000.):
    """Returns HCPV reservoir m³, STOIIP MMstb, GIIP Bscf, in-place MMboe.

    The oil and gas fractions partition HCPV; never double count the same pore space.
    """
    validate_inputs(area_km2, thickness_m, ntg, porosity, sw, bo, bg, oil_fraction)
    if scf_per_boe <= 0:raise ValueError('SCF/BOE must be positive')
    grv = float(area_km2)*1e6*float(thickness_m)
    hcpv = grv*ntg*porosity*(1-sw)
    oil_m3 = hcpv*oil_fraction
    gas_m3 = hcpv*(1-oil_fraction)
    stoiip = oil_m3/bo*STB_PER_M3/1e6
    giip = gas_m3/bg*SCF_PER_SM3/1e9
    boe = stoiip + giip*1000/scf_per_boe # Bscf -> MMboe: 1e9/6e3 / 1e6 = 1e3/6e3
    return dict(GRV_m3=grv,HCPV_rm3=hcpv,STOIIP_MMstb=stoiip,GIIP_Bscf=giip,
                Inplace_MMboe=boe)


def sensitivity_scenarios(*, area_km2, thickness_m, ntg, porosity, sw, bo, bg,
                          oil_fraction=1., scf_per_boe=6000., n=3000, seed=2026,
                          uncertainty=0.20):
    """Illustrative quantiles under independent TRIANGULAR input assumptions.

    P90=lower 10th percentile, P50=median, P10=upper 90th percentile.
    These are scenario sensitivities, NOT calibrated petroleum reserves distributions.
    """
    validate_inputs(area_km2, thickness_m, ntg, porosity, sw, bo, bg, oil_fraction)
    if n < 50 or not 0 <= uncertainty < .95:
        raise ValueError('Need >=50 simulations and a valid fractional uncertainty')
    rng = np.random.default_rng(seed)
    base = np.array([area_km2,thickness_m,ntg,porosity,sw,bo,bg],float)
    draws = np.vstack([rng.triangular(x*(1-uncertainty),x,x*(1+uncertainty),n)
                       for x in base]).T
    draws[:,2:5] = np.clip(draws[:,2:5],0,1)
    values = [in_place(*row,oil_fraction=oil_fraction,scf_per_boe=scf_per_boe) for row in draws]
    return {k:{'P90':float(np.percentile([v[k] for v in values],10)),
               'P50':float(np.percentile([v[k] for v in values],50)),
               'P10':float(np.percentile([v[k] for v in values],90))}
            for k in values[0]}
