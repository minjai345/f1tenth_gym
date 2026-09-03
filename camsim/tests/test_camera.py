import copy
import numpy as np, pytest
from camsim import config, camera

@pytest.fixture
def cfg():
    return config.load()

def test_focal_from_hfov(cfg):
    expected = (cfg.camera.image_width / 2.0) / np.tan(np.deg2rad(cfg.camera.hfov_deg) / 2.0)
    assert camera.focal_px(cfg) == pytest.approx(expected)

def test_horizon_at_center_for_pitch_zero(cfg):
    H_g2i, _ = camera.build(cfg, pitch_deg=0.0)
    uv = camera.project(H_g2i, np.array([[1000.0, 0.0]]))
    assert uv[0, 0] == pytest.approx(cfg.camera.image_width / 2.0, abs=0.5)
    assert uv[0, 1] == pytest.approx(cfg.camera.image_height / 2.0, abs=1.0)

def test_left_is_left_and_near_is_low(cfg):
    H_g2i, _ = camera.build(cfg)
    cy = cfg.camera.image_height / 2.0
    cx = cfg.camera.image_width / 2.0
    near = camera.project(H_g2i, np.array([[1.0, 0.0]]))[0]
    far = camera.project(H_g2i, np.array([[3.0, 0.0]]))[0]
    left = camera.project(H_g2i, np.array([[2.0, 0.5]]))[0]
    assert near[1] > far[1] > cy             # nearer -> lower in image, both below horizon
    assert left[0] < cx                      # vehicle +y (left) -> smaller u

def test_nearest_visible_ground(cfg):
    # bottom row v=image_height with h=height_m, f=focal_px -> x = h*f/(v-cy)
    _, H_i2g = camera.build(cfg, pitch_deg=0.0)
    v = float(cfg.camera.image_height)
    u = cfg.camera.image_width / 2.0
    cy = cfg.camera.image_height / 2.0
    f = camera.focal_px(cfg)
    expected_x = cfg.camera.height_m * f / (v - cy)
    g = camera.project(H_i2g, np.array([[u, v]]))[0]
    assert g[0] == pytest.approx(expected_x, abs=1e-3)
    assert g[1] == pytest.approx(0.0, abs=1e-6)

def test_round_trip(cfg):
    H_g2i, H_i2g = camera.build(cfg)
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.5, 8, 200), rng.uniform(-2, 2, 200)])
    back = camera.project(H_i2g, camera.project(H_g2i, pts))
    assert np.abs(back - pts).max() < 0.01

def test_pitch_down_raises_horizon(cfg):
    H0, _ = camera.build(cfg, pitch_deg=0.0)
    H10, _ = camera.build(cfg, pitch_deg=10.0)
    v0 = camera.project(H0, np.array([[1000.0, 0.0]]))[0, 1]
    v10 = camera.project(H10, np.array([[1000.0, 0.0]]))[0, 1]
    assert v10 < v0   # pitching down moves the horizon up in the image

def test_h_file_overrides(cfg, tmp_path):
    H_g2i, H_i2g = camera.build(cfg)
    p = tmp_path / "h.npy"; np.save(p, H_i2g * 1.0)
    cfg.camera.h_i2g_file = str(p)
    cfg.camera.height_m = 99.0   # would change H if not overridden
    H2, _ = camera.build(cfg)
    # NOTE: brief used `H2[2, 2]` / `H_g2i[2, 2]` to compare up to scale, but for the
    # default config (offset_x_m=0, pitch_deg=0) that element is exactly 0.0 (measured:
    # H_g2i[2, 2] == 0.0), so dividing by it is a division-by-zero bug independent of
    # camera.build()'s own normalization. Compare up to scale via the Frobenius norm
    # instead, which is always nonzero for an invertible homography.
    assert np.allclose(H2 / np.linalg.norm(H2), H_g2i / np.linalg.norm(H_g2i))

def test_h_file_pitch_jitter_matches_assumed_delta(cfg, tmp_path):
    """A measured h_i2g_file must not silently disable a pitch_deg override: the horizon
    should move by (about) the same amount as it would for the assumed camera model."""
    H_g2i_assumed0, H_i2g_assumed0 = camera.build(cfg)   # cfg.camera.pitch_deg == 0.0 by default
    p = tmp_path / "h.npy"; np.save(p, H_i2g_assumed0)
    cfg_file = copy.deepcopy(cfg)
    cfg_file.camera.h_i2g_file = str(p)

    far = np.array([[1000.0, 0.0]])
    v_file_base = camera.project(camera.build(cfg_file)[0], far)[0, 1]
    v_file_5 = camera.project(camera.build(cfg_file, pitch_deg=5.0)[0], far)[0, 1]
    v_assumed_base = camera.project(camera.build(cfg)[0], far)[0, 1]
    v_assumed_5 = camera.project(camera.build(cfg, pitch_deg=5.0)[0], far)[0, 1]

    delta_file = v_file_5 - v_file_base
    delta_assumed = v_assumed_5 - v_assumed_base
    assert delta_file == pytest.approx(delta_assumed, abs=0.5)

    # build(cfg_file) with no pitch override still equals the file's inverse up to scale.
    H2, _ = camera.build(cfg_file)
    assert np.allclose(H2 / np.linalg.norm(H2), H_g2i_assumed0 / np.linalg.norm(H_g2i_assumed0))

def test_h_file_is_cached(cfg, tmp_path, monkeypatch):
    H_g2i0, H_i2g0 = camera.build(cfg)
    p = tmp_path / "h.npy"; np.save(p, H_i2g0)
    cfg2 = copy.deepcopy(cfg)
    cfg2.camera.h_i2g_file = str(p)
    camera.build(cfg2)   # primes the module-level cache

    calls = {"n": 0}
    orig_load = np.load
    def counting_load(*a, **k):
        calls["n"] += 1
        return orig_load(*a, **k)
    monkeypatch.setattr(np, "load", counting_load)

    camera.build(cfg2)
    camera.build(cfg2)
    assert calls["n"] == 0
