"""시뮬 -> 실차 격차를 줄이는 열화. pitch 지터는 H를, 테이프 결손은 quad를, 나머지는 이미지를 바꾼다."""
import cv2
import numpy as np
from .config import Config
from .camera import build
from .render import ground_to_bev_matrix

# Internal glare ellipse shape parameters (not student tunables -- these fix the
# *shape* of the glare blob relative to image size, unlike glare_prob/glare_alpha
# in config.yaml which control *whether/how strongly* it appears).
_GLARE_AXES_FRAC_W = (1 / 10, 1 / 3)    # ellipse semi-axis (x) as a fraction of image width
_GLARE_AXES_FRAC_H = (1 / 12, 1 / 4)    # ellipse semi-axis (y) as a fraction of image height
_GLARE_Y_MIN_FRAC = 1 / 3               # ellipse center is restricted to the lower 2/3 of the image


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
    center = (int(rng.integers(0, w)), int(rng.integers(int(h * _GLARE_Y_MIN_FRAC), h)))
    axes = (int(rng.integers(w * _GLARE_AXES_FRAC_W[0], w * _GLARE_AXES_FRAC_W[1])),
            int(rng.integers(h * _GLARE_AXES_FRAC_H[0], h * _GLARE_AXES_FRAC_H[1])))
    cv2.ellipse(overlay, center, axes, float(rng.uniform(0, 180)), 0, 360, (255, 255, 255), -1)
    alpha = float(rng.uniform(*cfg.augment.glare_alpha))
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def augment_image(img: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return glare(motion_blur(img, cfg, rng), cfg, rng)


def jitter_bev(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """pitch가 δ만큼 틀어진 카메라 영상을 '공칭' H_i2g로 IPM하면 BEV가 어떻게 휘는지를 BEV 워프로 재현한다.

    참 지면점 -> (pitch+δ 카메라) 이미지 -> (공칭 H_i2g) 지면 -> BEV 픽셀.  실차에서 가감속 시 pitch가
    변해 IPM 평면 가정이 깨지는 효과(원거리 BEV 휘어짐)가 학습 데이터에 들어간다. 원근 렌더 없이 BEV만으로 적용.
    """
    j = cfg.augment.pitch_jitter_deg
    if j <= 0:
        return bev
    d = rng.uniform(-j, j)
    if cfg.camera.h_i2g_file:
        H_g2i_true, H_i2g_nom = build(cfg, pitch_deg=cfg.camera.pitch_deg + d)[0], build(cfg)[1]
    else:
        H_g2i_true = build(cfg, pitch_deg=cfg.camera.pitch_deg + d)[0]
        H_i2g_nom = build(cfg)[1]
    A = ground_to_bev_matrix(cfg)
    M = A @ H_i2g_nom @ H_g2i_true @ np.linalg.inv(A)
    h, w = bev.shape[:2]
    floor = tuple(int(c) for c in cfg.lane.color_floor)
    return cv2.warpPerspective(bev, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=floor)


def erase_patches(bev: np.ndarray, cfg: Config, rng: np.random.Generator, n_max: int = 3,
                  size_px=(10, 60)) -> np.ndarray:
    """BEV 위 임의 사각형을 바닥색으로 지운다 (테이프 마모·가림의 이미지 수준 근사). 학생 증강 예시용."""
    if rng.uniform() >= cfg.augment.tape_dropout_prob:        # 적용 확률은 tape_dropout_prob 를 공유
        return bev
    out = bev.copy()
    h, w = out.shape[:2]
    floor = np.array(cfg.lane.color_floor, np.uint8)
    for _ in range(int(rng.integers(1, n_max + 1))):
        ph, pw = rng.integers(size_px[0], size_px[1] + 1, 2)
        y, x = int(rng.integers(0, max(1, h - ph))), int(rng.integers(0, max(1, w - pw)))
        out[y:y + ph, x:x + pw] = floor
    return out


def example_augment(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """참고용 조합: pitch 워프 -> 패치 지우기 -> 블러 -> 글레어. 세기는 cfg.augment 로 조절.
    노트북에서 학생이 만드는 my_augment(bev, rng) 의 출발점."""
    return glare(motion_blur(erase_patches(jitter_bev(bev, cfg, rng), cfg, rng), cfg, rng), cfg, rng)
