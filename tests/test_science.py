import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import angsi_core as core
from fence import figure
from interpretation import convert, fence_seed, reflector, synthetic
from opportunities import COMPONENTS, projected_geometry, rank, regions
from petrophysics import interval_support
from seismic_qc import analyze_trace, metrics, register, sample_mask, window
from uncertainty import sensitivity


def test_registration():
    base = pd.DataFrame({"IL": [1, 1, 2], "XL": [2, 3, 3], "idx": [0, 1, 2]})
    monitor = base.iloc[[2, 0]].copy().assign(idx=[44, 99])
    pairs = register(base, monitor)
    assert pairs.set_index("bi").loc[0, "mi"] == 99
    assert (pairs["_merge"] == "left_only").sum() == 1
    with pytest.raises(pd.errors.MergeError):
        register(base, pd.concat([monitor, monitor]))


def test_mute_and_valid_zero():
    a = np.r_[np.zeros(8), 2, 0, 4, np.nan]
    valid = sample_mask(a)
    assert not valid[:8].any()
    assert valid[9]
    assert not valid[-1]
    assert sample_mask(a, 0)[:8].all()


def test_nrms_and_predictability():
    a = np.arange(1., 12.)
    assert metrics(a, a)["nrms"] == pytest.approx(0)
    assert metrics(a, -a)["nrms"] == pytest.approx(200)
    assert metrics(a, -a)["predictability"] == pytest.approx(100)
    assert metrics(a, 2*a)["nrms"] == pytest.approx(200/3)
    assert np.isnan(metrics(a*0, a*0)["nrms"])


def test_window_and_out_of_window_data():
    t = np.arange(360.)
    rng = np.random.default_rng(84)
    a = rng.normal(size=len(t))
    b = np.r_[np.nan, .8*a[:-1]]
    b[221:250] *= 1.3
    params = dict(reference=(10, 170), reservoir=(215, 260), max_lag=4)
    result, _, _, corrected = analyze_trace(t, a, b, **params)
    assert result["status"] == "PASS"
    assert result["shift_ms"] == 1
    assert result["heldout_after_nrms"] < 1e-10
    assert not np.allclose(corrected[225:246], a[225:246])
    assert window(t, (5, 8)).sum() == 4
    with pytest.raises(ValueError):
        analyze_trace(t, a, b, (10, 230), (215, 260))
    bad, _, _, values = analyze_trace(t, a, b, (10, 13), (215, 260))
    assert bad["status"] == "REVIEW" and np.isnan(values).all()


def test_spectral_equalization_and_rejected_polarity():
    rng = np.random.default_rng(36)
    t = np.arange(400.)
    a = rng.normal(size=len(t))
    b = .9*a
    good, *_ = analyze_trace(t, a, b, (10, 260), (290, 350), max_lag=0, spectral=True)
    assert good["status"] == "PASS" and good["spectral_applied"]
    bad, _, _, corrected = analyze_trace(t, a, -a, (10, 260), (290, 350), max_lag=0)
    assert bad["status"] == "FAIL" and np.isnan(corrected).all()


def test_time_depth_and_marker_parser(tmp_path):
    z, t = [100, 200, 300], [75, 170, 250]
    assert convert([150], z, t) == pytest.approx([122.5])
    assert np.isnan(convert([0], z, t)).all()
    assert np.isnan(convert([150], z, [75, 170, 160])).all()
    (tmp_path/"markers").mkdir()
    (tmp_path/"markers"/"Well-1_Markers.txt").write_text("# MDKB TVDSS\nTop 135 103\n")
    table = core.marker_file(tmp_path, "Well-1", 30)
    assert table.Marker.tolist() == ["Top"]
    assert table.Datum_Diff_m.iloc[0] == 2
    assert core.marker_file(tmp_path, "Well-2", 30).empty


def test_reflector_polarity_gaps_and_seed_bounds():
    t = np.arange(100.)
    section = np.zeros((100, 6))
    section[28, :] = 1
    section[32, :] = -2
    section[:, 2] = np.nan
    picks, _, _, _ = reflector(t, section, np.full(6, 30), 6, polarity=1)
    assert picks[[0, 1, 3, 4, 5]] == pytest.approx(np.full(5, 28))
    assert np.isnan(picks[2])
    seed = fence_seed(np.arange(5), [1, 3], [25, 35])
    assert np.isnan(seed[[0, 4]]).all()
    assert seed[2] == 30


def test_synthetic_and_petrophysics():
    time = np.arange(0., 500., 2)
    depth = np.arange(25., 450.)
    trace = synthetic(time, depth, np.where(depth < 240, 95, 75), np.full(len(depth), 2.3),
                      [0, 500], [0, 500], 35)
    assert time[np.nanargmax(trace)] == pytest.approx(240, abs=2)
    log = pd.DataFrame({"MD": depth, "PHIE": .14, "VSHALE": .25})
    assert interval_support(log, 0, 100, 200)["quality"] == 1
    assert np.isnan(interval_support(log.drop(columns="VSHALE"), 0, 100, 200)["quality"])


def test_candidate_geometry_and_verified_projection():
    frame = pd.DataFrame({"IL": [0, 5, 5, 0], "XL": [0, 0, 5, 5],
                          "status": ["PASS"]*3+["FAIL"], "raw_valid_fraction": [1.]*4,
                          "reference_valid_fraction": [1.]*4, "local_repeatability_contrast": [.3]*4})
    out = regions(frame, 5, minimum=3)
    assert len(out) == 1
    identifier, _, cells, shape = out[0]
    assert cells[["IL", "XL"]].shape == (3, 2)
    assert identifier == regions(frame.sample(frac=1, random_state=2), 5, minimum=3)[0][0]
    assert shape.is_valid and shape.area == 75 and shape.area < box(*shape.bounds).area
    coordinates = frame.assign(X=lambda f: 500000+25*f.XL, Y=lambda f: 200000+25*f.IL)
    area, geometry = projected_geometry(shape, coordinates, 32648)
    assert area == pytest.approx(75*25**2/1e6)
    assert geometry["type"] == "Polygon"
    with pytest.raises(ValueError):
        projected_geometry(shape, coordinates, 4326)


def test_ranking_missing_evidence_and_weight_stability():
    frame = pd.DataFrame([[.8, np.nan, .6, np.nan, np.nan, np.nan],
                          [.7, .7, .6, np.nan, np.nan, np.nan]], columns=COMPONENTS, index=["A","B"])
    weights = [20, 20, 30, 10, 10, 10]
    result = rank(frame, weights)
    assert result.Score["A"] == pytest.approx(68)
    assert result.Evidence_weight_pct["A"] == 50
    with pytest.raises(ValueError):
        rank(frame, [1]*6)
    uncertainty = sensitivity(frame, weights, {"Reservoir": pd.Series([.2, .2], index=frame.index)})
    assert uncertainty.equals(sensitivity(frame, weights, {"Reservoir": pd.Series([.2, .2], index=frame.index)}))
    assert uncertainty.Reservoir_score_span["A"] == pytest.approx(8)
    assert (uncertainty.Best_rank <= uncertainty.Worst_rank).all()


def test_plotly_fence_layout_and_depth_axis():
    b = np.arange(30.).reshape(10, 3)
    fig = figure([0, .1, .2], np.arange(10), b, b*.8, b*-.2, ["Greys"]*3, "TVDSS")
    assert [trace.yaxis for trace in fig.data] == ["y", "y2", "y3"]
    assert fig.data[0].zmax == fig.data[1].zmax
    assert fig.layout.yaxis3.title.text == "TVDSS (m)"

