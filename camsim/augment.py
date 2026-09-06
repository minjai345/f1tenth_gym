"""BEV 이미지 증강 (sim-to-real 갭 줄이기).

모델 입력이 BEV 이므로 증강도 BEV 에 직접 넣는다. 라벨(waypoint)은 지오메트리 참값이라 증강으로 바뀌지 않는다.
따라서 **정답을 옮기는 변형(이동·회전·스케일)은 넣으면 안 되고**, "같은 장면의 다른 관측"만 만들어야 한다.
예외는 `jitter_bev` 로, 실차에서 pitch 가 변하면 IPM 결과 자체가 휘므로 관측 변화가 맞다.

함수는 모두 `f(bev_bgr, cfg, rng) -> bev_bgr` 이고, 세기는 `config.yaml` 의 `augment:` 섹션에서 온다.
기본값은 전부 "변화 없음"이라, 학생이 노트북에서 값을 켜야 효과가 난다.

  기하   : jitter_bev(pitch 변화), ipm_blur(원거리 해상도 저하), erase_patches(테이프 마모)
  조명   : brightness_contrast, gamma, hsv_shift, illumination, shadow
  센서   : blur, noise, jpeg
  조합   : example_augment (위를 순서대로 적용하는 참고용 체인)
"""
import cv2
import numpy as np
from .config import Config
from .camera import build
from .render import ground_to_bev_matrix


# ---- 기하 -------------------------------------------------------------------

def jitter_pitch(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """pitch 를 ±pitch_jitter_deg 안에서 흔든 H_g2i. (원근 렌더용. BEV 학습에는 jitter_bev 를 쓴다.)"""
    j = cfg.augment.pitch_jitter_deg
    d = rng.uniform(-j, j) if j > 0 else 0.0
    return build(cfg, pitch_deg=cfg.camera.pitch_deg + d)[0]


def jitter_bev(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """pitch 가 δ만큼 틀어진 카메라 영상을 '공칭' H_i2g 로 IPM 했을 때의 BEV 왜곡을 재현한다.

    참 지면점 -> (pitch+δ 카메라) 이미지 -> (공칭 H_i2g) 지면 -> BEV 픽셀. 가감속 시 서스펜션이 물러
    IPM 평면 가정이 깨지는 현상이고, 먼 곳일수록 크게 휜다. 기하학적으로 정확한 유일한 증강이다.
    """
    j = cfg.augment.pitch_jitter_deg
    if j <= 0:
        return bev
    d = rng.uniform(-j, j)
    H_g2i_true = build(cfg, pitch_deg=cfg.camera.pitch_deg + d)[0]
    H_i2g_nom = build(cfg)[1]
    A = ground_to_bev_matrix(cfg)
    M = A @ H_i2g_nom @ H_g2i_true @ np.linalg.inv(A)
    h, w = bev.shape[:2]
    return cv2.warpPerspective(bev, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=tuple(int(c) for c in cfg.lane.color_floor))


def ipm_blur(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """전방 거리에 비례해 커지는 블러. 실차 IPM 은 먼 곳일수록 카메라 픽셀 하나가 넓은 바닥을 덮어 뭉개진다.

    BEV 위쪽(먼 곳)일수록 강한 GaussianBlur 를 준 여러 장을 만들어 행 구간별로 이어 붙인다.
    세기: `ipm_blur_max_px` (BEV 맨 위 행에서의 커널 크기. 0 이면 사용 안 함).
    """
    kmax = int(cfg.augment.ipm_blur_max_px)
    if kmax < 3:
        return bev
    h = bev.shape[0]
    levels = list(range(3, kmax + 1, 2))                  # 홀수 커널만
    if not levels:
        return bev
    out = bev.copy()
    edges = np.linspace(0, h, len(levels) + 1).astype(int)   # 위(먼 곳)부터 강한 블러
    for k, lvl in enumerate(reversed(levels)):
        y0, y1 = edges[k], edges[k + 1]
        pad = lvl
        a, b = max(0, y0 - pad), min(h, y1 + pad)
        blurred = cv2.GaussianBlur(bev[a:b], (lvl, lvl), 0)
        out[y0:y1] = blurred[y0 - a:y1 - a]
    return out


def dropout_quads(quads: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """렌더 전에 테이프 사각형을 연속 구간으로 제거한다 (테이프가 실제로 뜯긴 상태)."""
    if rng.uniform() >= cfg.augment.tape_dropout_prob or len(quads) == 0:
        return quads
    lo, hi = cfg.augment.tape_dropout_len
    n = int(rng.integers(lo, hi + 1))
    start = int(rng.integers(0, max(1, len(quads) - n)))
    mask = np.ones(len(quads), bool)
    mask[start:start + n] = False
    return quads[mask]


def erase_patches(bev: np.ndarray, cfg: Config, rng: np.random.Generator, n_max: int = 3,
                  size_px=(10, 60)) -> np.ndarray:
    """BEV 위 임의 사각형을 바닥색으로 지운다 (테이프 마모·다른 차의 가림을 이미지 수준에서 근사)."""
    if rng.uniform() >= cfg.augment.tape_dropout_prob:
        return bev
    out = bev.copy()
    h, w = out.shape[:2]
    floor = np.array(cfg.lane.color_floor, np.uint8)
    for _ in range(int(rng.integers(1, n_max + 1))):
        ph, pw = rng.integers(size_px[0], size_px[1] + 1, 2)
        y, x = int(rng.integers(0, max(1, h - ph))), int(rng.integers(0, max(1, w - pw)))
        out[y:y + ph, x:x + pw] = floor
    return out


# ---- 조명 (OpenCV 표준 기법) --------------------------------------------------

def brightness_contrast(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """cv2.convertScaleAbs 로 밝기(beta)·대비(alpha) 변경. 노출·조도 변화에 해당."""
    a = cfg.augment.contrast_range
    b = cfg.augment.brightness_delta
    alpha = rng.uniform(a[0], a[1])
    beta = rng.uniform(-b, b) if b > 0 else 0.0
    if alpha == 1.0 and beta == 0.0:
        return bev
    return cv2.convertScaleAbs(bev, alpha=alpha, beta=beta)


def gamma(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """LUT 감마 보정. 카메라 감마·톤커브 차이, 어두운 곳의 디테일 변화."""
    lo, hi = cfg.augment.gamma_range
    if lo == 1.0 and hi == 1.0:
        return bev
    g = rng.uniform(lo, hi)
    lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / g)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(bev, lut)


def hsv_shift(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """HSV 색상·채도 변경. 색온도(백열/형광/자연광)에 따라 테이프 색이 달라 보이는 것."""
    dh = cfg.augment.hue_shift_deg
    slo, shi = cfg.augment.sat_scale
    if dh == 0 and slo == 1.0 and shi == 1.0:
        return bev
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV).astype(np.int16)
    if dh:                                        # OpenCV hue 는 0..179 (도의 절반)
        hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-dh, dh) / 2)) % 180
    if not (slo == 1.0 and shi == 1.0):
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(slo, shi), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def illumination(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """불균일 조명: 부드러운 밝기 기울기를 곱한다. 천장 조명이 한쪽만 밝은 실습실 상황.

    세기 s(`illum_strength`) 이면 배율이 1-s ~ 1+s 사이에서 화면을 가로지르며 완만히 변한다.
    """
    s = float(cfg.augment.illum_strength)
    if s <= 0:
        return bev
    h, w = bev.shape[:2]
    th = rng.uniform(0, 2 * np.pi)                       # 밝아지는 방향
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (np.cos(th) * xs / w + np.sin(th) * ys / h)      # -1..1 범위의 선형 기울기
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-6) * 2 - 1
    field = (1.0 + s * t)[..., None]
    return np.clip(bev.astype(np.float32) * field, 0, 255).astype(np.uint8)


def shadow(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """임의의 볼록 다각형 영역을 어둡게 (구조물·사람 그림자). 경계는 블러로 부드럽게."""
    if rng.uniform() >= cfg.augment.shadow_prob:
        return bev
    h, w = bev.shape[:2]
    n = int(rng.integers(3, 6))
    pts = np.column_stack([rng.integers(-w // 4, w + w // 4, n), rng.integers(-h // 4, h + h // 4, n)])
    mask = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 1.0, cv2.LINE_AA)
    k = max(3, (min(h, w) // 20) | 1)
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
    dark = rng.uniform(cfg.augment.shadow_darkness[0], cfg.augment.shadow_darkness[1])
    return np.clip(bev.astype(np.float32) * (1 - mask * (1 - dark)), 0, 255).astype(np.uint8)


# ---- 센서 --------------------------------------------------------------------

def blur(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """GaussianBlur. 초점 흐림·모션 블러의 단순 근사 (글로벌 셔터라 실제 모션 블러는 작다)."""
    kmax = int(cfg.augment.blur_max_px)
    if kmax < 3:
        return bev
    k = int(rng.integers(1, (kmax + 1) // 2 + 1)) * 2 - 1     # 홀수
    return bev if k < 3 else cv2.GaussianBlur(bev, (k, k), 0)


def noise(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """가우시안 센서 노이즈 (게인을 올렸을 때)."""
    s = float(cfg.augment.noise_sigma)
    if s <= 0:
        return bev
    return np.clip(bev.astype(np.float32) + rng.normal(0, s, bev.shape), 0, 255).astype(np.uint8)


def jpeg(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """JPEG 압축 아티팩트. ROS image_transport compressed 를 쓰면 실차 입력에 이게 섞인다."""
    lo, hi = cfg.augment.jpeg_quality
    if lo >= 100 and hi >= 100:
        return bev
    q = int(rng.integers(lo, hi + 1))
    ok, enc = cv2.imencode(".jpg", bev, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else bev


# ---- 조합 --------------------------------------------------------------------

def example_augment(bev: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """참고용 체인: 기하 -> 조명 -> 센서 순서. 실제 촬영 과정과 같은 순서다.

    config 의 세기가 전부 기본값(0/1.0)이면 항등 함수다. 노트북 과제에서 my_augment 의 출발점으로 쓴다.
    """
    for fn in (jitter_bev, ipm_blur, erase_patches,
               illumination, shadow, brightness_contrast, gamma, hsv_shift,
               blur, noise, jpeg):
        bev = fn(bev, cfg, rng)
    return bev
