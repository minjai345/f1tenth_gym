"""pose + 테이프 quad + (선택) LiDAR scan -> 합성 전방 카메라 이미지."""
import cv2
import numpy as np
from .config import Config
from .camera import project

_SHIFT = 4
_SCALE = 1 << _SHIFT


def to_vehicle(pose, pts_world: np.ndarray) -> np.ndarray:
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    A = np.array([[c, -s], [s, c]])          # rows of R(th); (p - o) @ A == R(-th) (p - o)
    return (np.asarray(pts_world, dtype=np.float64) - [x, y]) @ A


def visible_quads(qv: np.ndarray, scan, cfg: Config) -> np.ndarray:
    """qv: (M,4,2) vehicle-frame quads. Returns bool mask of quads to draw.

    Near/far culling must be relative to the camera, not the vehicle origin: with
    camera.offset_x_m > 0 the camera sits ahead of the rear axle, so a quad at
    vehicle-frame x in (0, offset_x_m) is actually *behind* the camera and would
    otherwise pass a vehicle-relative near check, get projected with a flipped
    (negative-depth) homogeneous coordinate, and appear mirrored above the horizon.
    LiDAR range/bearing stay vehicle-relative: the scan comes from gym at the
    vehicle pose, not the (assumed) camera pose.
    """
    ctr = qv.mean(1)
    off = cfg.camera.offset_x_m
    xc = qv[:, :, 0] - off                            # camera-frame forward coordinate
    rng = np.hypot(ctr[:, 0], ctr[:, 1])               # vehicle-origin range, for LiDAR
    cam_rng = np.hypot(ctr[:, 0] - off, ctr[:, 1])     # camera-origin range, for the far cut
    keep = (xc.min(1) > cfg.render.near_m) & (cam_rng < cfg.render.far_m)
    if scan is not None:
        scan = np.asarray(scan)
        fov = cfg.render.lidar_fov_rad
        brg = np.arctan2(ctr[:, 1], ctr[:, 0])
        # gym lays out beam i at angle -fov/2 + i*fov/(n-1), i.e. n-1 steps span fov.
        idx = np.rint((brg + fov / 2.0) / fov * (len(scan) - 1)).astype(int).clip(0, len(scan) - 1)
        keep &= rng < scan[idx]
    return keep


def render(pose, quads_world: np.ndarray, scan, H_g2i: np.ndarray, cfg: Config) -> np.ndarray:
    W, Hh = cfg.camera.image_width, cfg.camera.image_height
    img = np.empty((Hh, W, 3), np.uint8)
    img[:] = cfg.lane.color_floor
    qv = to_vehicle(pose, quads_world)
    keep = visible_quads(qv, scan, cfg)
    if keep.any():
        uv = project(H_g2i, qv[keep])                       # (K,4,2)
        polys = np.round(uv * _SCALE).astype(np.int32)
        cv2.fillPoly(img, list(polys), tuple(int(c) for c in cfg.lane.color_tape),
                     lineType=cv2.LINE_AA, shift=_SHIFT)
    return img


def draw_points(img, pts_vehicle, H_g2i, color=(0, 255, 0), radius=4):
    uv = project(H_g2i, np.asarray(pts_vehicle, float).reshape(-1, 2))
    for u, v in uv:
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img, (int(round(u)), int(round(v))), radius, color, -1)
    return img


# ---- BEV (top-down) ----------------------------------------------------------
# BEV 픽셀 규약: 위 = 전방(+x), 왼쪽 = 차량 좌측(+y). 범위·해상도는 config의 bev 섹션.

def bev_size(cfg: Config):
    """(height, width) in pixels of the BEV image."""
    b = cfg.bev
    return (int(round((b.x_range_m[1] - b.x_range_m[0]) / b.resolution_m)),
            int(round((b.y_range_m[1] - b.y_range_m[0]) / b.resolution_m)))


def bev_pixels(pts_vehicle: np.ndarray, cfg: Config) -> np.ndarray:
    """Vehicle-frame ground points (...,2) m -> BEV pixel coords (...,2) (u right, v down)."""
    b = cfg.bev
    p = np.asarray(pts_vehicle, dtype=np.float64)
    u = (b.y_range_m[1] - p[..., 1]) / b.resolution_m
    v = (b.x_range_m[1] - p[..., 0]) / b.resolution_m
    return np.stack([u, v], axis=-1)


def ground_to_bev_matrix(cfg: Config) -> np.ndarray:
    """3x3 affine mapping ground (x, y, 1) -> BEV pixel (u, v, 1). Same convention as bev_pixels."""
    b = cfg.bev
    r = b.resolution_m
    return np.array([[0.0, -1.0 / r, b.y_range_m[1] / r],
                     [-1.0 / r, 0.0, b.x_range_m[1] / r],
                     [0.0, 0.0, 1.0]])


def render_bev(pose, quads_world: np.ndarray, cfg: Config) -> np.ndarray:
    """정답 BEV: 테이프 quad를 지오메트리에서 직접 top-down으로 그린다 (카메라·IPM 무관)."""
    h, w = bev_size(cfg)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = cfg.lane.color_floor
    qv = to_vehicle(pose, quads_world)
    b = cfg.bev
    ctr = qv.mean(1)
    keep = ((ctr[:, 0] > b.x_range_m[0] - 0.5) & (ctr[:, 0] < b.x_range_m[1] + 0.5) &
            (ctr[:, 1] > b.y_range_m[0] - 0.5) & (ctr[:, 1] < b.y_range_m[1] + 0.5))
    if keep.any():
        polys = np.round(bev_pixels(qv[keep], cfg) * _SCALE).astype(np.int32)
        cv2.fillPoly(img, list(polys), tuple(int(c) for c in cfg.lane.color_tape),
                     lineType=cv2.LINE_AA, shift=_SHIFT)
    return img


def ipm_bev(img_perspective: np.ndarray, H_i2g: np.ndarray, cfg: Config) -> np.ndarray:
    """실차 2주차 IPM과 같은 연산: 원근 영상을 H_i2g로 지면에 펴서 BEV 규격으로 warp한다."""
    h, w = bev_size(cfg)
    H_img2bev = ground_to_bev_matrix(cfg) @ H_i2g
    floor = tuple(int(c) for c in cfg.lane.color_floor)
    return cv2.warpPerspective(img_perspective, H_img2bev, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=floor)
