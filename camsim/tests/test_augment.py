import numpy as np, pytest
from camsim import config, camera, augment

@pytest.fixture
def cfg():
    return config.load()

def test_jitter_pitch_changes_horizon(cfg):
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
