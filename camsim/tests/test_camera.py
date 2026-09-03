import numpy as np, pytest
from camsim import config, camera

@pytest.fixture
def cfg():
    return config.load()

def test_focal_from_hfov(cfg):
    assert camera.focal_px(cfg) == pytest.approx(320.0)   # 320 / tan(45deg)

def test_horizon_at_center_for_pitch_zero(cfg):
    H_g2i, _ = camera.build(cfg, pitch_deg=0.0)
    uv = camera.project(H_g2i, np.array([[1000.0, 0.0]]))
    assert uv[0, 0] == pytest.approx(320.0, abs=0.5)
    assert uv[0, 1] == pytest.approx(200.0, abs=1.0)

def test_left_is_left_and_near_is_low(cfg):
    H_g2i, _ = camera.build(cfg)
    near = camera.project(H_g2i, np.array([[1.0, 0.0]]))[0]
    far = camera.project(H_g2i, np.array([[3.0, 0.0]]))[0]
    left = camera.project(H_g2i, np.array([[2.0, 0.5]]))[0]
    assert near[1] > far[1] > 200.0          # nearer -> lower in image, both below horizon
    assert left[0] < 320.0                   # vehicle +y (left) -> smaller u

def test_nearest_visible_ground(cfg):
    # bottom row v=400 with h=0.2, f=320 -> x = h*f/(v-cy) = 0.32 m
    _, H_i2g = camera.build(cfg, pitch_deg=0.0)
    g = camera.project(H_i2g, np.array([[320.0, 400.0]]))[0]
    assert g[0] == pytest.approx(0.32, abs=1e-3)
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
