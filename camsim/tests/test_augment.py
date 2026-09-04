import numpy as np, pytest
from camsim import config, camera, augment

@pytest.fixture
def cfg():
    return config.load()

def test_jitter_pitch_changes_horizon(cfg):
    cfg.augment.pitch_jitter_deg = 2.0
    rng = np.random.default_rng(0)
    horizon = cfg.camera.image_height / 2.0
    vs = {round(camera.project(augment.jitter_pitch(cfg, rng), np.array([[1000., 0.]]))[0, 1], 1)
          for _ in range(20)}
    assert len(vs) > 5 and all(abs(v - horizon) < 15 for v in vs)

def test_jitter_zero_when_disabled(cfg):
    cfg.augment.pitch_jitter_deg = 0.0
    H = augment.jitter_pitch(cfg, np.random.default_rng(0))
    assert np.allclose(H, camera.build(cfg)[0])

def test_dropout_removes_contiguous_run(cfg):
    q = np.zeros((1000, 4, 2)); q[:, :, 0] = np.arange(1000)[:, None]
    cfg.augment.tape_dropout_prob = 1.0
    out = augment.dropout_quads(q, cfg, np.random.default_rng(3))
    removed = 1000 - len(out)
    assert 5 <= removed <= 20
    ids = out[:, 0, 0].astype(int)
    gaps = np.where(np.diff(ids) > 1)[0]
    assert len(gaps) == 1

def test_dropout_noop_when_prob_zero(cfg):
    q = np.zeros((10, 4, 2)); cfg.augment.tape_dropout_prob = 0.0
    assert len(augment.dropout_quads(q, cfg, np.random.default_rng(0))) == 10

def test_image_ops_keep_shape_dtype(cfg):
    img = np.full((400, 640, 3), 200, np.uint8)
    img[200:, 300:340] = (30, 30, 220)
    cfg.augment.blur_max_px = 3; cfg.augment.glare_prob = 1.0
    rng = np.random.default_rng(0)
    out = augment.augment_image(img, cfg, rng)
    assert out.shape == img.shape and out.dtype == np.uint8

def test_glare_brightens(cfg):
    img = np.full((400, 640, 3), 100, np.uint8)
    cfg.augment.glare_prob = 1.0
    out = augment.glare(img, cfg, np.random.default_rng(0))
    assert out.mean() > img.mean()

def test_glare_alpha_from_config(cfg):
    """glare_alpha=[1.0, 1.0] forces a full-strength blend, proving the config key is read."""
    img = np.full((400, 640, 3), 100, np.uint8)
    cfg.augment.glare_prob = 1.0
    cfg.augment.glare_alpha = [1.0, 1.0]
    out = augment.glare(img, cfg, np.random.default_rng(0))
    assert out.max() == 255


def test_jitter_bev_warps_far_more_than_near(cfg):
    """pitch 지터 BEV 워프: 먼 곳이 더 많이 움직이고, 지터 0이면 항등."""
    from camsim import render
    import cv2
    h, w = render.bev_size(cfg)
    bev = np.full((h, w, 3), 128, np.uint8)
    for x in (1.0, 3.5):
        u, v = render.bev_pixels(np.array([[x, 0.0]]), cfg)[0]
        cv2.circle(bev, (int(u), int(v)), 3, (255, 255, 255), -1)
    cfg.augment.pitch_jitter_deg = 0.0
    assert np.array_equal(augment.jitter_bev(bev, cfg, np.random.default_rng(0)), bev)
    cfg.augment.pitch_jitter_deg = 3.0
    out = augment.jitter_bev(bev, cfg, np.random.default_rng(1))
    assert out.shape == bev.shape and not np.array_equal(out, bev)
    def row_of(img, lo, hi):
        rows = np.where(np.any(np.all(img == 255, axis=-1)[lo:hi], axis=1))[0]
        return rows.mean() + lo if len(rows) else np.nan
    near_v = render.bev_pixels(np.array([[1.0, 0.0]]), cfg)[0][1]
    far_v = render.bev_pixels(np.array([[3.5, 0.0]]), cfg)[0][1]
    d_near = abs(row_of(out, int(near_v) - 60, int(near_v) + 60) - near_v)
    d_far = abs(row_of(out, 0, int(far_v) + 80) - far_v)
    assert d_far > d_near


def test_defaults_are_plain(cfg):
    """기본 config 는 증강 전부 off: example_augment 가 항등이어야 한다."""
    from camsim import render
    bev = np.random.default_rng(0).integers(0, 255, (*render.bev_size(cfg), 3), dtype=np.uint8)
    assert np.array_equal(augment.example_augment(bev, cfg, np.random.default_rng(1)), bev)


def test_erase_patches_uses_floor_color(cfg):
    from camsim import render
    bev = np.full((*render.bev_size(cfg), 3), 255, np.uint8)
    cfg.augment.tape_dropout_prob = 1.0
    out = augment.erase_patches(bev, cfg, np.random.default_rng(5), n_max=3)
    erased = np.all(out == cfg.lane.color_floor, axis=-1)
    assert out.shape == bev.shape and erased.sum() >= 100
