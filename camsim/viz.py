"""지도 위 시각화: 전체 맵에서 pose 위치, 차 주변 확대 top-down에 카메라 화각과 GT waypoint.

세 좌표계를 오간다.
  world  : gym 맵 좌표 (m)
  map px : examples/*.png 픽셀. col = (x - ox)/res, row = H-1 - (y - oy)/res  (gym은 이미지를 상하 반전해 씀)
  canvas : 확대 뷰 픽셀. 차를 중심에 두고 위가 world +y 가 아니라 **차량 전방**이 되도록 회전한다.
"""
import os
import cv2
import numpy as np
import yaml
from .config import Config
from .track import Track
from .render import to_vehicle, bev_pixels

CAR_LEN_M, CAR_WID_M = 0.58, 0.31          # gym 기본 차체 크기 (표시용)
COL_WP, COL_CAR, COL_FOV, COL_CENTER = (0, 255, 0), (255, 80, 0), (255, 200, 0), (90, 90, 90)


def to_world(pose, pts_vehicle: np.ndarray) -> np.ndarray:
    """Inverse of render.to_vehicle."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return np.asarray(pts_vehicle, dtype=np.float64) @ R.T + [x, y]


class MapImage:
    """맵 PNG + yaml(origin, resolution). world <-> pixel 변환을 담당."""

    def __init__(self, yaml_path: str):
        meta = yaml.safe_load(open(yaml_path))
        img_path = os.path.join(os.path.dirname(yaml_path), meta["image"])
        self.gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if self.gray is None:
            raise FileNotFoundError(img_path)
        self.res = float(meta["resolution"])
        self.ox, self.oy = float(meta["origin"][0]), float(meta["origin"][1])
        self.h, self.w = self.gray.shape

    def world_to_px(self, pts_world: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_world, dtype=np.float64)
        col = (p[..., 0] - self.ox) / self.res
        row = (self.h - 1) - (p[..., 1] - self.oy) / self.res
        return np.stack([col, row], axis=-1)


def _poly(pts, shift=4):
    return np.round(np.asarray(pts) * (1 << shift)).astype(np.int32)


def draw_track_on_map(mapimg: MapImage, track: Track, cfg: Config, crop_margin_m: float = 2.0):
    """맵 PNG 위에 테이프를 그리고 트랙 주변만 잘라 반환. (img_bgr, offset_px) — offset은 잘라낸 좌상단."""
    img = cv2.cvtColor(mapimg.gray, cv2.COLOR_GRAY2BGR)
    tape = tuple(int(c) for c in cfg.lane.color_tape)
    px = mapimg.world_to_px(track.quads)                     # (M,4,2)
    cv2.fillPoly(img, list(_poly(px)), tape, lineType=cv2.LINE_AA, shift=4)
    lo = np.floor(px.reshape(-1, 2).min(0) - crop_margin_m / mapimg.res).astype(int).clip(0)
    hi = np.ceil(px.reshape(-1, 2).max(0) + crop_margin_m / mapimg.res).astype(int)
    hi = np.minimum(hi, [mapimg.w, mapimg.h])
    return img[lo[1]:hi[1], lo[0]:hi[0]].copy(), lo


def mark_poses_on_map(map_bgr: np.ndarray, offset_px, mapimg: MapImage, poses, labels=None, radius=10):
    """잘라낸 맵 위에 pose들을 번호 붙은 원과 헤딩 선으로 표시."""
    for k, pose in enumerate(poses):
        c = mapimg.world_to_px(pose[:2]) - offset_px
        tip = mapimg.world_to_px(to_world(pose, [[3.0, 0.0]])[0]) - offset_px
        c, tip = tuple(np.round(c).astype(int)), tuple(np.round(tip).astype(int))
        cv2.line(map_bgr, c, tip, COL_CAR, 2, cv2.LINE_AA)
        cv2.circle(map_bgr, c, radius, COL_CAR, -1, cv2.LINE_AA)
        cv2.putText(map_bgr, str(labels[k] if labels else k), (c[0] + radius + 2, c[1] + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return map_bgr


def local_view(pose, track: Track, wp_vehicle, cfg: Config, mapimg: MapImage = None,
               ahead_m: float = 5.0, behind_m: float = 1.5, half_width_m: float = 3.0,
               res_m: float = 0.01) -> np.ndarray:
    """차량 중심 확대 top-down. 위 = 차량 전방. 벽(맵), 테이프, 차, 카메라 화각, BEV 범위, GT waypoint."""
    h, w = int(round((ahead_m + behind_m) / res_m)), int(round(2 * half_width_m / res_m))

    def vp(pts_v):   # vehicle m -> canvas px
        p = np.asarray(pts_v, dtype=np.float64)
        return np.stack([(half_width_m - p[..., 1]) / res_m, (ahead_m - p[..., 0]) / res_m], axis=-1)

    img = np.full((h, w, 3), 235, np.uint8)
    if mapimg is not None:                                    # 맵의 벽을 캔버스 좌표로 warp
        src = mapimg.world_to_px(to_world(pose, [[0, 0], [1, 0], [0, 1]])).astype(np.float32)
        dst = vp([[0, 0], [1, 0], [0, 1]]).astype(np.float32)
        A = cv2.getAffineTransform(src, dst)
        occ = cv2.warpAffine(mapimg.gray, A, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        img[occ < 128] = (40, 40, 40)
    qv = to_vehicle(pose, track.quads)
    ctr = qv.mean(1)
    keep = (ctr[:, 0] > -behind_m - 1) & (ctr[:, 0] < ahead_m + 1) & (np.abs(ctr[:, 1]) < half_width_m + 1)
    tape = tuple(int(c) for c in cfg.lane.color_tape)
    cv2.fillPoly(img, list(_poly(vp(qv[keep]))), tape, lineType=cv2.LINE_AA, shift=4)
    cv = to_vehicle(pose, track.center)
    kc = (cv[:, 0] > -behind_m - 1) & (cv[:, 0] < ahead_m + 1) & (np.abs(cv[:, 1]) < half_width_m + 1)
    for p in vp(cv[kc]):
        cv2.circle(img, tuple(np.round(p).astype(int)), 1, COL_CENTER, -1)
    # BEV 범위(점선 사각형)와 카메라 화각(반투명 부채꼴)
    b = cfg.bev
    rect = vp([[b.x_range_m[0], b.y_range_m[1]], [b.x_range_m[1], b.y_range_m[1]],
               [b.x_range_m[1], b.y_range_m[0]], [b.x_range_m[0], b.y_range_m[0]]])
    cv2.polylines(img, [_poly(rect)], True, (200, 120, 200), 1, cv2.LINE_AA, shift=4)
    half = np.deg2rad(cfg.camera.hfov_deg) / 2
    ox = cfg.camera.offset_x_m
    far = cfg.render.far_m
    wedge = vp([[ox, 0], [ox + far * np.cos(half), far * np.sin(half)], [ox + far * np.cos(half), -far * np.sin(half)]])
    overlay = img.copy()
    cv2.fillPoly(overlay, [_poly(wedge)], COL_FOV, lineType=cv2.LINE_AA, shift=4)
    img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)
    # 차체(후륜축 원점, 앞으로 뻗은 사각형)와 heading 화살표
    car = vp([[-0.1, CAR_WID_M / 2], [CAR_LEN_M - 0.1, CAR_WID_M / 2],
              [CAR_LEN_M - 0.1, -CAR_WID_M / 2], [-0.1, -CAR_WID_M / 2]])
    cv2.fillPoly(img, [_poly(car)], COL_CAR, lineType=cv2.LINE_AA, shift=4)
    a0, a1 = vp([[0, 0], [1.0, 0]])
    cv2.arrowedLine(img, tuple(np.round(a0).astype(int)), tuple(np.round(a1).astype(int)), COL_CAR, 2, cv2.LINE_AA, tipLength=0.25)
    # GT waypoint
    for k, p in enumerate(vp(np.asarray(wp_vehicle))):
        c = tuple(np.round(p).astype(int))
        cv2.circle(img, c, 6, COL_WP, -1, cv2.LINE_AA)
        cv2.putText(img, f"{cfg.waypoints.ahead_m[k]:g}", (c[0] + 8, c[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 0), 1, cv2.LINE_AA)
    return img


def side_by_side(*imgs, gap=10, bg=255) -> np.ndarray:
    """높이를 맞춰 가로로 붙인다."""
    hmax = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        s = hmax / i.shape[0]
        r = cv2.resize(i, (int(round(i.shape[1] * s)), hmax), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out += [r, np.full((hmax, gap, 3), bg, np.uint8)]
    return np.hstack(out[:-1])
