"""시뮬 -> 실차 격차를 줄이는 열화. pitch 지터는 H를, 테이프 결손은 quad를, 나머지는 이미지를 바꾼다."""
import cv2
import numpy as np
from .config import Config
from .camera import build


def jitter_pitch(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    j = cfg.augment.pitch_jitter_deg
    d = rng.uniform(-j, j) if j > 0 else 0.0
    return build(cfg, pitch_deg=cfg.camera.pitch_deg + d)[0]


def dropout_quads(quads: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if rng.uniform() >= cfg.augment.tape_dropout_prob or len(quads) == 0:
        return quads
    lo, hi = cfg.augment.tape_dropout_len
    n = int(rng.integers(lo, hi + 1))
    start = int(rng.integers(0, max(1, len(quads) - n)))
    mask = np.ones(len(quads), bool)
    mask[start:start + n] = False
    return quads[mask]


def motion_blur(img: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(1, cfg.augment.blur_max_px + 1))
    if k <= 1:
        return img
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k                 # horizontal streak
    return cv2.filter2D(img, -1, kernel)


def glare(img: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if rng.uniform() >= cfg.augment.glare_prob:
        return img
    h, w = img.shape[:2]
    overlay = img.copy()
    center = (int(rng.integers(0, w)), int(rng.integers(h // 3, h)))
    axes = (int(rng.integers(w // 10, w // 3)), int(rng.integers(h // 12, h // 4)))
    cv2.ellipse(overlay, center, axes, float(rng.uniform(0, 180)), 0, 360, (255, 255, 255), -1)
    alpha = float(rng.uniform(0.3, 0.7))
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def augment_image(img: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return glare(motion_blur(img, cfg, rng), cfg, rng)
