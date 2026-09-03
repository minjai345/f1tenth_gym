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
