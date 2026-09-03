import numpy as np, pytest
from camsim import config, track

CSV = "examples/example_waypoints.csv"

@pytest.fixture(scope="module")
def trk():
    return track.from_csv(CSV, config.load())

def test_resample_is_uniform_and_closed(trk):
    d = np.hypot(*np.diff(trk.center, axis=0).T)
    assert np.allclose(d, 0.05, atol=2e-3)
    assert np.hypot(*(trk.center[0] - trk.center[-1])) > 0.03     # no duplicated closing point
    assert trk.length == pytest.approx(156.36, abs=0.2)
    assert trk.s[0] == 0.0 and trk.s[-1] < trk.length

def test_heading_matches_direction(trk):
    d = trk.center[1] - trk.center[0]
    assert np.arctan2(d[1], d[0]) == pytest.approx(trk.heading[0], abs=0.05)

def test_quads_sit_on_both_sides(trk):
    n = len(trk.center)
    assert trk.quads.shape == (2 * n, 4, 2)
    i = 100
    nrm = np.array([-np.sin(trk.heading[i]), np.cos(trk.heading[i])])
    left = (trk.quads[i].mean(0) - trk.center[i]) @ nrm
    right = (trk.quads[n + i].mean(0) - trk.center[i]) @ nrm
    assert left == pytest.approx(0.4, abs=0.02)
    assert right == pytest.approx(-0.4, abs=0.02)

def test_quad_width_is_tape_width(trk):
    q = trk.quads[100]
    assert np.hypot(*(q[3] - q[0])) == pytest.approx(0.05, abs=1e-3)

def test_save_load_roundtrip(trk, tmp_path):
    p = tmp_path / "t.npz"
    track.save(trk, p)
    t2 = track.load(p)
    assert np.allclose(t2.quads, trk.quads) and t2.length == trk.length
