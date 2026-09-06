import copy

import cv2
import numpy as np
import pytest

from camsim import augment, config, render


@pytest.fixture
def cfg():
    return config.load()


@pytest.fixture
def bev(cfg):
    """가운데에 노란 테이프 두 줄이 있는 회색 BEV."""
    h, w = render.bev_size(cfg)
    img = np.full((h, w, 3), cfg.lane.color_floor, np.uint8)
    img[:, w // 2 - 60:w // 2 - 50] = cfg.lane.color_tape
    img[:, w // 2 + 50:w // 2 + 60] = cfg.lane.color_tape
    return img


def rng():
    return np.random.default_rng(0)


ALL = [augment.jitter_bev, augment.ipm_blur, augment.erase_patches, augment.illumination,
       augment.shadow, augment.brightness_contrast, augment.gamma, augment.hsv_shift,
       augment.blur, augment.noise, augment.jpeg]


def test_defaults_are_identity(cfg, bev):
    """기본 config 에서는 모든 증강이 항등이어야 한다 (plain 모델)."""
    for fn in ALL:
        assert np.array_equal(fn(bev, cfg, rng()), bev), fn.__name__
    assert np.array_equal(augment.example_augment(bev, cfg, rng()), bev)


def test_all_keep_shape_and_dtype(cfg, bev):
    """세기를 켜도 크기·타입은 그대로 (모델 입력 규격 유지)."""
    a = cfg.augment
    a.pitch_jitter_deg, a.ipm_blur_max_px, a.tape_dropout_prob = 2.0, 9, 1.0
    a.brightness_delta, a.contrast_range, a.gamma_range = 40.0, [0.7, 1.3], [0.5, 2.0]
    a.hue_shift_deg, a.sat_scale, a.illum_strength = 20.0, [0.5, 1.5], 0.4
    a.shadow_prob, a.blur_max_px, a.noise_sigma, a.jpeg_quality = 1.0, 7, 8.0, [30, 60]
    for fn in ALL + [augment.example_augment]:
        out = fn(bev, cfg, rng())
        assert out.shape == bev.shape and out.dtype == np.uint8, fn.__name__


def test_brightness_and_gamma_change_mean(cfg, bev):
    cfg.augment.brightness_delta = 40.0
    outs = [augment.brightness_contrast(bev, cfg, np.random.default_rng(s)).mean() for s in range(6)]
    assert max(outs) > bev.mean() > min(outs)
    cfg.augment.brightness_delta = 0.0
    cfg.augment.gamma_range = [2.0, 2.0]
    assert augment.gamma(bev, cfg, rng()).mean() > bev.mean()      # gamma > 1 -> 밝아짐


def test_hsv_shift_moves_hue_not_geometry(cfg, bev):
    cfg.augment.hue_shift_deg = 60.0
    out = augment.hsv_shift(bev, cfg, rng())
    tape = np.all(bev == cfg.lane.color_tape, axis=-1)
    h_in = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)[..., 0][tape].mean()
    h_out = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)[..., 0][tape].mean()
    assert abs(h_in - h_out) > 3                                    # 색상은 바뀌고
    changed = np.any(out != bev, axis=-1)
    assert changed[tape].mean() > 0.9                               # 테이프 위치는 그대로


def test_illumination_is_non_uniform(cfg, bev):
    cfg.augment.illum_strength = 0.5
    out = augment.illumination(bev, cfg, rng()).astype(float)
    b = bev.astype(float)
    ratio = (out + 1) / (b + 1)
    assert ratio.max() - ratio.min() > 0.3                          # 화면 위치마다 배율이 다르다


def test_shadow_darkens_a_region(cfg, bev):
    cfg.augment.shadow_prob = 1.0
    cfg.augment.shadow_darkness = [0.3, 0.3]
    out = augment.shadow(bev, cfg, rng())
    darker = (out.astype(int) < bev.astype(int) - 10).any(-1)
    assert 0.02 < darker.mean() < 0.98                              # 일부만 어두워진다


def test_noise_and_jpeg_perturb(cfg, bev):
    cfg.augment.noise_sigma = 10.0
    assert not np.array_equal(augment.noise(bev, cfg, rng()), bev)
    cfg.augment.noise_sigma = 0.0
    cfg.augment.jpeg_quality = [20, 20]
    out = augment.jpeg(bev, cfg, rng())
    assert not np.array_equal(out, bev) and out.shape == bev.shape


def test_ipm_blur_is_stronger_far_away(cfg):
    """먼 곳(위쪽 행)이 가까운 곳(아래쪽 행)보다 더 뭉개져야 한다."""
    h, w = render.bev_size(cfg)
    img = np.full((h, w, 3), 0, np.uint8)
    img[:, ::20] = 255                                              # 세로 줄무늬
    cfg.augment.ipm_blur_max_px = 15
    out = augment.ipm_blur(img, cfg, rng())

    def contrast(rows):
        return out[rows].astype(float).std()

    assert contrast(slice(0, h // 6)) < contrast(slice(-h // 6, None))


def test_jitter_bev_moves_far_more_than_near(cfg):
    h, w = render.bev_size(cfg)
    img = np.full((h, w, 3), 128, np.uint8)
    for x in (1.0, 3.5):
        u, v = render.bev_pixels(np.array([[x, 0.0]]), cfg)[0]
        cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)
    cfg.augment.pitch_jitter_deg = 3.0
    out = augment.jitter_bev(img, cfg, np.random.default_rng(1))
    assert not np.array_equal(out, img)

    def row_of(lo, hi):
        rows = np.where(np.any(np.all(out == 255, axis=-1)[lo:hi], axis=1))[0]
        return rows.mean() + lo if len(rows) else np.nan

    near_v = render.bev_pixels(np.array([[1.0, 0.0]]), cfg)[0][1]
    far_v = render.bev_pixels(np.array([[3.5, 0.0]]), cfg)[0][1]
    d_near = abs(row_of(int(near_v) - 60, int(near_v) + 60) - near_v)
    d_far = abs(row_of(0, int(far_v) + 80) - far_v)
    assert d_far > d_near


def test_erase_patches_uses_floor_color(cfg, bev):
    cfg.augment.tape_dropout_prob = 1.0
    out = augment.erase_patches(bev, cfg, np.random.default_rng(5), n_max=3)
    erased = np.all(out == cfg.lane.color_floor, axis=-1)
    assert erased.sum() >= 100


def test_dropout_quads_removes_contiguous_run(cfg):
    q = np.zeros((1000, 4, 2))
    q[:, :, 0] = np.arange(1000)[:, None]
    cfg.augment.tape_dropout_prob = 1.0
    out = augment.dropout_quads(q, cfg, np.random.default_rng(3))
    removed = 1000 - len(out)
    assert 5 <= removed <= 20
    ids = out[:, 0, 0].astype(int)
    assert len(np.where(np.diff(ids) > 1)[0]) == 1


def test_dropout_quads_noop_when_prob_zero(cfg):
    q = np.zeros((10, 4, 2))
    assert len(augment.dropout_quads(q, cfg, rng())) == 10


def test_jitter_pitch_changes_horizon(cfg):
    from camsim import camera
    cfg.augment.pitch_jitter_deg = 2.0
    vs = {round(camera.project(augment.jitter_pitch(cfg, rng()), np.array([[1000.0, 0.0]]))[0, 1], 1)}
    r = np.random.default_rng(0)
    vs |= {round(camera.project(augment.jitter_pitch(cfg, r), np.array([[1000.0, 0.0]]))[0, 1], 1)
           for _ in range(20)}
    assert len(vs) > 5 and all(abs(v - cfg.camera.image_height / 2) < 15 for v in vs)


def test_jitter_pitch_zero_when_disabled(cfg):
    from camsim import camera
    cfg.augment.pitch_jitter_deg = 0.0
    assert np.allclose(augment.jitter_pitch(cfg, rng()), camera.build(cfg)[0])
