"""가정 카메라(화각, 높이, pitch) 또는 실측 H_i2g 파일에서 지면<->이미지 homography를 만든다.

frames
  vehicle : x forward, y left, z up, origin at rear axle on the ground
  camera  : x right, y down, z forward (OpenCV)
  image   : u right, v down (pixels)
H_g2i maps ground (x, y, 1) -> image (u, v, w).  H_i2g = inv(H_g2i).
"""
import os
import numpy as np
from .config import Config

# vehicle -> camera axes, pitch 0:  x_c = -y_v, y_c = -z_v, z_c = x_v
_R_VC = np.array([[0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [1.0, 0.0, 0.0]])

# Cache of loaded h_i2g_file matrices, keyed by absolute path, so build() does not
# hit the disk on every call (it may be called once per rendered frame).
_H_FILE_CACHE: dict = {}


def focal_px(cfg: Config) -> float:
    return (cfg.camera.image_width / 2.0) / np.tan(np.deg2rad(cfg.camera.hfov_deg) / 2.0)


def intrinsics(cfg: Config) -> np.ndarray:
    f = focal_px(cfg)
    cx, cy = cfg.camera.image_width / 2.0, cfg.camera.image_height / 2.0
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def extrinsics(cfg: Config, pitch_deg: float) -> np.ndarray:
    """Return 3x4 [R | t] with p_c = R p_v + t."""
    th = np.deg2rad(pitch_deg)
    # rotate about camera x axis: positive pitch tilts the optical axis toward +y_c (down)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(th), -np.sin(th)],
                   [0.0, np.sin(th), np.cos(th)]])
    R = Rx @ _R_VC
    C = np.array([cfg.camera.offset_x_m, 0.0, cfg.camera.height_m])  # camera center in vehicle frame
    t = -R @ C
    return np.hstack([R, t[:, None]])


def _load_h_i2g_file(path: str) -> np.ndarray:
    key = os.path.abspath(path)
    H = _H_FILE_CACHE.get(key)
    if H is None:
        H = np.load(key).astype(np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"h_i2g_file must be 3x3, got {H.shape}")
        _H_FILE_CACHE[key] = H
    return H


def _assumed_h_g2i(cfg: Config, pitch_deg: float) -> np.ndarray:
    """H_g2i from the assumed camera model (hfov/height/pitch/offset), not the measured file."""
    Rt = extrinsics(cfg, pitch_deg)
    H_g2i = intrinsics(cfg) @ Rt[:, [0, 1, 3]]   # ground plane z_v = 0
    # Normalize by the Frobenius norm rather than H_g2i[2, 2]: for offset_x_m=0 at
    # pitch_deg=0 (the default config), H_g2i[2, 2] is exactly 0 (it is the depth of
    # ground point (0, 0), which sits at zero forward distance from the camera), so
    # dividing by it produces NaN/Inf. The norm is always nonzero for an invertible H
    # and project() is scale-invariant, so this does not change any projected result.
    H_g2i /= np.linalg.norm(H_g2i)
    return H_g2i


def build(cfg: Config, pitch_deg=None):
    """Return (H_g2i, H_i2g). A measured H_i2g file, if configured, wins over the assumed camera.

    If a pitch_deg override is requested that differs from cfg.camera.pitch_deg (e.g. augment's
    pitch jitter), the measured H is corrected by the *delta* between the assumed camera at the
    configured pitch and at the requested pitch, rather than being silently ignored:
    H_g2i = H_file @ inv(H_assumed(cfg.pitch_deg)) @ H_assumed(pitch_deg).
    """
    if cfg.camera.h_i2g_file:
        H_i2g_file = _load_h_i2g_file(cfg.camera.h_i2g_file)
        H_g2i_file = np.linalg.inv(H_i2g_file)
        if pitch_deg is not None and pitch_deg != cfg.camera.pitch_deg:
            H_assumed_base = _assumed_h_g2i(cfg, cfg.camera.pitch_deg)
            H_assumed_new = _assumed_h_g2i(cfg, pitch_deg)
            H_g2i = H_g2i_file @ np.linalg.inv(H_assumed_base) @ H_assumed_new
            H_g2i /= np.linalg.norm(H_g2i)
            return H_g2i, np.linalg.inv(H_g2i)
        return H_g2i_file, H_i2g_file
    if pitch_deg is None:
        pitch_deg = cfg.camera.pitch_deg
    H_g2i = _assumed_h_g2i(cfg, pitch_deg)
    return H_g2i, np.linalg.inv(H_g2i)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    hom = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1) @ H.T
    return hom[..., :2] / hom[..., 2:3]
