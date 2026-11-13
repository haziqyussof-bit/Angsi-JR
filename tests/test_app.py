from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import segyio
from streamlit.testing.v1 import AppTest

import angsi_core as C

APP = Path(__file__).resolve().parents[1]/"angsi_jr_v8.py"


@pytest.fixture
def synthetic_project(tmp_path, monkeypatch):
    """Test-only small survey pair with V7 directory and source-file formats."""
    vols = tmp_path/"volumes"
    vols.mkdir()
    wd = tmp_path/"well_data"
    for sub in ("las", "checkshots", "markers"):
        (wd/sub).mkdir(parents=True)
    keys = [(il, xl) for il in range(100, 111, 5) for xl in range(200, 216, 5)]
    time = np.arange(0, 2200, 2)
    random = np.random.default_rng(8)
    baseline = random.normal(0, .1, (len(keys), len(time)))
    for center in (1510, 1690):
        baseline += 2*np.exp(-((time-center)/5)**2)
    for vintage in ("baseline", "monitor"):
        spec = segyio.spec()
        spec.format, spec.samples, spec.tracecount = 5, time, len(keys)
        path = vols/f"{vintage}.sgy"
        with segyio.create(str(path), spec) as file:
            for i, (il, xl) in enumerate(keys):
                file.header[i] = {segyio.TraceField.INLINE_3D: il,
                                  segyio.TraceField.CROSSLINE_3D: xl,
                                  segyio.TraceField.CDP_X: 500000+20*(xl-200),
                                  segyio.TraceField.CDP_Y: 200000+20*(il-100),
                                  segyio.TraceField.SourceGroupScalar: 1}
                data = baseline[i]
                if vintage == "monitor":
                    data = np.where((time>=1490)&(time<=1710), data, data*.9)
                file.trace[i] = data.astype(np.float32)
    wells = pd.DataFrame({"Well Name": ["Well-1","Well-2","Well-3"], "X": [500000,500100,500300],
                          "Y": [200000]*3, "RT": [20]*3, "TD": [2400]*3})
    wells.to_excel(wd/"WellList.xlsx", index=False, startrow=2)
    for well in wells["Well Name"]:
        (wd/"checkshots"/f"{well}_Checkshot.txt").write_text("0 0\n2500 2500\n")
        (wd/"markers"/f"{well}_Markers.txt").write_text("# MDKB TVDSS\nI-25 1530 1510\nI-35 1710 1690\n")
        log = """~Version Information
 VERS. 2.0 : Version
 WRAP. NO  : One line per depth
~Well Information
 STRT.M 1000 :
 STOP.M 2300 :
 STEP.M 5 :
 NULL. -999.25 :
 WELL. Fixture :
~Curve Information
 DEPT.M :
 GR_P.API :
 PHIE.V/V :
 VSHALE.V/V :
 DENS_P.G/C3 :
 DT.US/F :
~ASCII
"""
        log += "\n".join(f"{depth} 50 0.15 0.25 {2.1 if depth < 1530 else 2.3} 85"
                          for depth in range(1000, 2301, 5))
        (wd/"las"/f"{well}.las").write_text(log)
    monkeypatch.setenv("ANGSI_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(C, "find_project", lambda root=tmp_path: original_find(root))
    return tmp_path


original_find = C.find_project


def _button(at, label):
    return next(item for item in at.button if label in item.label)


def _radio(at, label):
    return next(item for item in at.radio if item.label == label)


def assert_clean(at):
    assert not at.exception, [exc.message for exc in at.exception]


def test_startup_missing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "find_project", lambda: original_find(tmp_path/"unavailable"))
    app = AppTest.from_file(str(APP)).run()
    assert_clean(app)
    assert "Project files unavailable" in app.error[0].value


def test_streamlit_pages_and_calculations(synthetic_project):
    app = AppTest.from_file(str(APP), default_timeout=40).run()
    assert_clean(app)
    assert not app.error
    _radio(app, "Seismic orientation").set_value("East → West well fence").run()
    assert_clean(app)
    _radio(app, "Vertical axis").set_value("TVDSS").run()
    assert_clean(app)
    _button(app, "Seismic QC").click().run()
    assert_clean(app)
    next(item for item in app.selectbox if item.label == "QC module").select("4.9 Before and after").run()
    assert_clean(app)
    _button(app, "Horizon 2D & 3D").click().run()
    assert_clean(app)
    _button(app, "Build / refresh horizon previews").click().run()
    assert_clean(app)
    _button(app, "Well Correlation").click().run()
    assert_clean(app)
    next(item for item in app.toggle if item.label == "Show East–West seismic fence").set_value(True).run()
    assert_clean(app)
    _button(app, "Opportunities").click().run()
    _radio(app, "Sample spacing in IL/XL").set_value(5).run()
    _button(app, "Calculate survey diagnostics").click().run()
    assert_clean(app)
    assert any(item.label == "Attribute map" for item in app.selectbox)
    assert any(item.label == "Candidate card" for item in app.selectbox)
    next(item for item in app.checkbox if item.label == "Use two interpreted marker surfaces").check().run()
    _button(app, "Calculate survey diagnostics").click().run()
    assert_clean(app)
    assert any(item.label == "Attribute map" and "depth_conversion_span_m" in item.options for item in app.selectbox)
    _button(app, "Use this trace").click().run()
    _button(app, "Overview").click().run()
    assert_clean(app)


def test_gr_fill_to_right(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "find_project", lambda: original_find(tmp_path/"unavailable"))
    from angsi_jr_v8 import gr_category_grid
    grid = gr_category_grid([0, 1, 2, 3], [15, 45, 75, 105], 0, 3, rows=4, columns=10)
    assert np.isnan(grid[:, 0]).all()
    assert grid[:, -1].tolist() == [0, 1, 2, 3]

