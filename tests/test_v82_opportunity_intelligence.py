"""Synthetic fixtures — not Angsi measured field data."""
import numpy as np
import pandas as pd
import pytest

from opportunity_intelligence import (
    blank_inputs, example_hypothetical_inputs, evidence_gaps,
    hcpv_scenarios, label_and_rank_candidates,
)


def sample_table():
    return pd.DataFrame({
        "ID": ["C-99", "C-07", "C-01"],
        "Score": [80., 65., 75.],
        "Area_km2": [1.0, 3.0, 2.0],
        "Cells": [10, 30, 20],
    }).set_index("ID")


def test_friendly_names_and_independent_area_ranks():
    got = label_and_rank_candidates(sample_table())
    assert got.loc["C-99", "Opportunity"] == "OPP-01"
    assert got.loc["C-99", "Overall_rank"] == 1
    assert got.loc["C-07", "Area_rank"] == 1
    assert got.loc["C-07", "Cell_count_rank"] == 1
    assert got.loc["C-07", "Opportunity"] == "OPP-03"
    assert got.loc["C-07", "Area_km2"] == 3.
    assert "C-99" in got.index  # persistent IDs remain for audit


def test_missing_area_or_assumptions_produces_no_volume():
    t = sample_table().copy()
    t.loc["C-99", "Area_km2"] = np.nan
    a = example_hypothetical_inputs(t.index)
    got = hcpv_scenarios(t, a, draws=400)
    assert np.isnan(got.loc["C-99", "HCPV_P50_Mm3"])
    assert "Physical area unavailable" in got.loc["C-99", "Volume_assessment"]
    got2 = hcpv_scenarios(t, blank_inputs(t.index), draws=400)
    assert got2.HCPV_P50_Mm3.isna().all()


def test_volume_sensitivity_deterministic_and_monotone_area():
    t = sample_table()
    inputs = example_hypothetical_inputs(t.index, net=(12, 15, 18),
                                         phi=(.19, .2, .21), sw=(.35, .4, .45))
    x = hcpv_scenarios(t, inputs, draws=2500, seed=42)
    y = hcpv_scenarios(t, inputs, draws=2500, seed=42)
    pd.testing.assert_frame_equal(x, y)
    assert (x.HCPV_P90_Mm3 <= x.HCPV_P50_Mm3).all()
    assert (x.HCPV_P50_Mm3 <= x.HCPV_P10_Mm3).all()
    assert x.HCPV_P50_Mm3.idxmax() == "C-07"  # largest physical area
    assert x.Volume_scenario_top_rank_share_pct.sum() == pytest.approx(100., abs=.1)


def test_candidate_specific_net_pay_can_change_volume_rank():
    t = sample_table()
    a = example_hypothetical_inputs(t.index, net=(10, 12, 15))
    # Smaller area, much more net pay => might beat largest area.
    a.loc[a.ID == "C-99", ["NetPay_low_m", "NetPay_mode_m", "NetPay_high_m"]] = [55, 60, 65]
    got = hcpv_scenarios(t, a, seed=2026)
    ranked = label_and_rank_candidates(t.join(got))
    assert ranked.loc["C-99", "Volume_rank"] == 1
    assert ranked.loc["C-07", "Area_rank"] == 1


def test_invalid_saturation_and_draws():
    t = sample_table()
    a = example_hypothetical_inputs(t.index)
    a.loc[a.ID == "C-99", "Sw_high"] = 1.5
    got = hcpv_scenarios(t, a, draws=250)
    assert np.isnan(got.loc["C-99", "HCPV_P50_Mm3"])
    assert "Water saturation" in got.loc["C-99", "Volume_assessment"]
    with pytest.raises(ValueError):
        hcpv_scenarios(t, a, draws=100)


def test_missing_geological_evidence_caveats():
    data = {
        "Reservoir": np.nan, "Horizon": np.nan,
        "Production": np.nan, "Structure": np.nan,
        "Area_km2": np.nan, "Support_wells": 0,
    }
    msg = evidence_gaps(data)
    assert "production/drainage" in msg
    assert "verified surface area" in msg
    assert "local well control" in msg


def test_unparseable_editor_value_returns_message_instead_of_crash():
    t = sample_table()
    inputs = example_hypothetical_inputs(t.index)
    inputs["Phi_low"] = inputs["Phi_low"].astype(object)
    inputs.loc[inputs.ID == "C-07", "Phi_low"] = "unavailable"
    got = hcpv_scenarios(t, inputs, draws=300)
    assert np.isnan(got.loc["C-07", "HCPV_P50_Mm3"])
    assert "numeric" in got.loc["C-07", "Volume_assessment"]


def test_fixed_parameter_scenario_is_supported():
    t = sample_table()
    assumed = example_hypothetical_inputs(t.index, net=(10., 10., 10.),
                                          phi=(.2, .2, .2), sw=(.5, .5, .5))
    result = hcpv_scenarios(t, assumed, draws=300)
    assert result.loc["C-07", "HCPV_P50_Mm3"] == pytest.approx(3.0)
    assert result.loc["C-07", "HCPV_range_Mm3"] == pytest.approx(0)
