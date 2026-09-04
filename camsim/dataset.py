"""학습 데이터 두 경로. 모델 입력은 BEV(top-down) 이미지다.

  시뮬 : pose -> render_bev (지오메트리에서 직접 top-down) -> 카메라 가시 마스크 -> (증강)  => BEV
  실차 : 카메라 -> undistort -> IPM(render.ipm_bev)                                    => BEV
  두 BEV는 config의 bev 섹션(범위·해상도)을 공유하므로 같은 모델에 그대로 들어간다.

  SynthDataset : 디스크 없이 매 샘플 생성 (무한 IterableDataset)
  DiskDataset  : generate_dataset() 이 저장한 폴더(images/*.png = BEV, labels.csv)를 읽는다.
                 증강은 augment_fn(bev, rng) 하나로 로딩 때 적용한다 (기본 None = plain).
                 실차 IPM 출력을 같은 포맷으로 저장하면 그대로 학습된다.
"""
import csv
import os
import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .config import Config
from .track import Track
from . import gt, render, augment


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1).float().div_(255.0)


def make_sample(track: Track, cfg: Config, rng: np.random.Generator, augment_fn=None,
                mask: np.ndarray = None, with_camera: bool = False):
    """Return (bev_bgr, waypoints (K,2) m, pose) — with_camera=True 면 (bev, wp, pose, cam_img).

    augment_fn : callable(bev_bgr, rng) -> bev_bgr. None 이면 증강 없음(plain).
                 라벨은 증강과 무관한 참값이므로, 증강은 "같은 정답에 대한 다른 관측"을 만드는 것이어야 한다.
    mask       : render.bev_visibility_mask(H_g2i, cfg). None이면 계산한다(느림 — 반복 호출 시 미리 만들어 넘길 것).
    with_camera: 시각화용 원근 카메라 뷰도 함께 렌더 (학습에는 쓰지 않음)
    """
    from .camera import build
    H_g2i = build(cfg)[0]
    if mask is None:
        mask = render.bev_visibility_mask(H_g2i, cfg)
    pose = gt.sample_pose(track, cfg, rng)
    bev = render.render_bev(pose, track.quads, cfg, mask)
    if augment_fn is not None:
        bev = augment_fn(bev, rng)
    wp = gt.waypoints_ahead(pose, track, cfg)
    if with_camera:
        return bev, wp, pose, render.render(pose, track.quads, None, H_g2i, cfg)
    return bev, wp, pose


class SynthDataset(IterableDataset):
    def __init__(self, track: Track, cfg: Config, seed: int = 0, augment_fn=None):
        from .camera import build
        self.track, self.cfg, self.seed, self.augment_fn = track, cfg, seed, augment_fn
        self.mask = render.bev_visibility_mask(build(cfg)[0], cfg)

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        norm = self.cfg.waypoints.norm_m
        while True:
            img, wp, _ = make_sample(self.track, self.cfg, rng, self.augment_fn, mask=self.mask)
            yield to_tensor(img), torch.from_numpy(wp.reshape(-1) / norm).float()


# ---- 디스크 데이터셋 -------------------------------------------------------------

LABELS_CSV = "labels.csv"
IMAGES_DIR = "images"


def _label_header(cfg: Config):
    return ["file", "x", "y", "theta"] + [f"wp{k}_{a}" for k in range(len(cfg.waypoints.ahead_m)) for a in ("x", "y")]


def generate_dataset(track: Track, cfg: Config, n: int, out_dir: str, seed: int = 0,
                     augment_fn=None, log_every: int = 2000) -> str:
    """n장을 out_dir/images/NNNNNN.png (BEV, 모델 입력) + out_dir/labels.csv 로 저장. labels.csv 경로를 반환.
    기본은 증강 없는 plain 데이터. 증강은 보통 로딩 시(DiskDataset(augment_fn=...))에 넣는다."""
    from .camera import build
    img_dir = os.path.join(out_dir, IMAGES_DIR)
    os.makedirs(img_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    mask = render.bev_visibility_mask(build(cfg)[0], cfg)
    path = os.path.join(out_dir, LABELS_CSV)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_label_header(cfg))
        for i in range(n):
            img, wp, pose = make_sample(track, cfg, rng, augment_fn, mask=mask)
            name = f"{i:06d}.png"
            cv2.imwrite(os.path.join(img_dir, name), img)
            w.writerow([name, *np.round(pose, 6), *np.round(wp.reshape(-1), 4)])
            if log_every and (i + 1) % log_every == 0:
                print(f"{i + 1}/{n}", flush=True)
    return path


def read_labels(out_dir: str):
    """labels.csv -> (files list, poses (N,3), waypoints (N,K,2))."""
    with open(os.path.join(out_dir, LABELS_CSV), newline="") as f:
        rows = list(csv.reader(f))
    hdr, rows = rows[0], rows[1:]
    files = [r[0] for r in rows]
    arr = np.array([[float(v) for v in r[1:]] for r in rows], dtype=np.float64).reshape(len(rows), -1)
    poses, wps = arr[:, :3], arr[:, 3:].reshape(len(rows), -1, 2)
    return files, poses, wps


def split_indices(n: int, split: str, val_frac: float = 0.1, seed: int = 0) -> np.ndarray:
    """결정적 train/val 분리. split in {"train", "val", "all"}."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_frac))
    if split == "all":
        return np.arange(n)
    if split == "val":
        return np.sort(perm[:n_val])
    if split == "train":
        return np.sort(perm[n_val:])
    raise ValueError(f"split must be train/val/all, got {split!r}")


class DiskDataset(torch.utils.data.Dataset):
    """generate_dataset() 폴더를 읽는 map-style Dataset. __getitem__ -> (tensor (3,h,w), target (2K,))."""

    def __init__(self, root: str, cfg: Config, split: str = "train", val_frac: float = 0.1,
                 seed: int = 0, augment_fn=None):
        """augment_fn: callable(bev_bgr, rng) -> bev_bgr, 로딩 때 적용. None 이면 저장된 이미지 그대로."""
        self.root, self.cfg, self.augment_fn = root, cfg, augment_fn
        self.files, self.poses, self.wps = read_labels(root)
        self.idx = split_indices(len(self.files), split, val_frac, seed)
        self.seed = seed
        if self.wps.shape[1] != len(cfg.waypoints.ahead_m):
            raise ValueError(f"labels have {self.wps.shape[1]} waypoints but config has {len(cfg.waypoints.ahead_m)}")

    def __len__(self):
        return len(self.idx)

    def load_image(self, i: int) -> np.ndarray:
        """저장된 BEV BGR 이미지. i는 이 split 안의 인덱스."""
        img = cv2.imread(os.path.join(self.root, IMAGES_DIR, self.files[self.idx[i]]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.files[self.idx[i]])
        return img

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.augment_fn is not None:
            rng = np.random.default_rng([self.seed, int(self.idx[i]), int(torch.randint(0, 2**31 - 1, (1,)))])
            img = self.augment_fn(img, rng)
        wp = self.wps[self.idx[i]]
        return to_tensor(img), torch.from_numpy(wp.reshape(-1) / self.cfg.waypoints.norm_m).float()
