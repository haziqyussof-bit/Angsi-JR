"""Synthetic in-memory trace validation for time-window / exclusion accounting."""
import numpy as np
import pandas as pd
from survey_analysis import evaluate
import survey_analysis


class FakeVolume:
    def __init__(self, traces):
        self.samples = np.arange(len(traces[0]), dtype=float)
        self.trace = traces

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_time_window_reports_valid_and_invalid_cells(monkeypatch):
    rng = np.random.default_rng(42)
    a = rng.normal(size=1900)
    baseline = FakeVolume([a, a])
    monitor = FakeVolume([np.r_[a[1:], 0], np.full_like(a, np.nan)])
    vols = {"baseline": baseline, "monitor": monitor}
    monkeypatch.setattr(survey_analysis.segyio, "open", lambda path, *args, **kwargs: vols[str(path)])
    pairs = pd.DataFrame({"IL": [100, 100], "XL": [200, 210], "X": [1., 2.], "Y": [3., 4.],
                          "bi": [0, 1], "mi": [0, 1]})
    params = dict(reference=(700., 950.), reservoir=(1500., 1700.), max_lag=4,
                  zero_run=8, reference_limit=200., spectral=False, mute_before_ms=None)
    frame, qc = evaluate("baseline", "monitor", pairs, params, return_qc=True)
    assert len(frame) == len(qc) == 2
    assert qc.reason.iloc[0] == "valid_for_screening"
    assert qc.reason.iloc[1] == "insufficient_valid_target_samples"
    assert qc.included.sum() == 1


def test_missing_horizon_is_counted_not_silently_dropped(monkeypatch):
    a = np.ones(1900)
    vols = {"baseline": FakeVolume([a, a]), "monitor": FakeVolume([a, a])}
    monkeypatch.setattr(survey_analysis.segyio, "open", lambda path, *args, **kwargs: vols[str(path)])
    pairs = pd.DataFrame({"IL": [100, 100], "XL": [200, 210], "X": [1., 2.], "Y": [3., 4.],
                          "bi": [0, 1], "mi": [0, 1]})
    params = dict(reference=(700., 950.), reservoir=(1500., 1700.), max_lag=4,
                  zero_run=8, reference_limit=200., spectral=False, mute_before_ms=None)
    _, qc = evaluate("baseline", "monitor", pairs, params, top={(100, 200): 1500},
                     base={(100, 200): 1700}, return_qc=True)
    assert (qc.reason == "missing_or_unreliable_horizon").sum() == 1
