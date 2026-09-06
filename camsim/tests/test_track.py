import numpy as np, pytest
from camsim import config, track

CSV = "examples/example_waypoints.csv"

@pytest.fixture(scope="module")
def ctx():
    cfg = config.load()
    return cfg, track.from_csv(CSV, cfg)


@pytest.fixture(scope="module")
def ctx_racing():
    """CSV 의 레이싱 라인을 기준 경로로 (CSV 고유 속성을 검사할 때)."""
    cfg = config.load()
    cfg.waypoints.line = "racing"
    return cfg, track.from_csv(CSV, cfg)


@pytest.fixture(scope="module")
def ctx_fixed():
    """벽을 무시하고 중심선에서 일정 폭 (실차 테이프 트랙 방식)."""
    cfg = config.load()
    cfg.lane.follow_walls = False
    return cfg, track.from_csv(CSV, cfg)

def test_resample_is_uniform_and_closed(ctx, ctx_racing):
    for cfg, trk in (ctx, ctx_racing):
        d = np.hypot(*np.diff(trk.center, axis=0).T)
        assert np.allclose(d, cfg.lane.segment_len_m, atol=2e-3)
        assert np.hypot(*(trk.center[0] - trk.center[-1])) > 0.03   # no duplicated closing point
        assert trk.s[0] == 0.0 and trk.s[-1] < trk.length
    assert ctx_racing[1].length == pytest.approx(156.36, abs=0.2)   # a property of the CSV
    assert ctx[1].length > ctx_racing[1].length                     # 중간선이 레이싱 라인보다 길다

def test_heading_matches_direction(ctx):
    """heading 은 중앙차분 접선이므로, 곡률이 작은 구간에서 전방 세그먼트 방향과 거의 같아야 한다."""
    cfg, trk = ctx
    d = np.roll(trk.center, -1, axis=0) - trk.center
    seg = np.arctan2(d[:, 1], d[:, 0])
    err = np.abs(np.angle(np.exp(1j * (seg - trk.heading))))
    assert np.median(err) < 0.02 and np.percentile(err, 95) < 0.1

def test_quads_sit_on_both_sides(ctx_fixed):
    cfg, trk = ctx_fixed
    n = len(trk.center)
    assert trk.quads.shape == (2 * n, 4, 2)
    i = 100
    half = cfg.lane.track_width_m / 2.0
    nrm = np.array([-np.sin(trk.heading[i]), np.cos(trk.heading[i])])
    left = (trk.quads[i].mean(0) - trk.center[i]) @ nrm
    right = (trk.quads[n + i].mean(0) - trk.center[i]) @ nrm
    assert left == pytest.approx(half, abs=0.02)
    assert right == pytest.approx(-half, abs=0.02)

def test_quad_width_is_tape_width(ctx):
    cfg, trk = ctx
    q = trk.quads[100]
    assert np.hypot(*(q[3] - q[0])) == pytest.approx(cfg.lane.tape_width_m, abs=1e-3)

def test_save_load_roundtrip(ctx, tmp_path):
    cfg, trk = ctx
    p = tmp_path / "t.npz"
    track.save(trk, p)
    t2 = track.load(p)
    assert np.allclose(t2.quads, trk.quads) and t2.length == trk.length
