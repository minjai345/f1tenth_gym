"""디스크 없이 매 샘플 pose -> 렌더 -> 증강 -> 크롭. 무한 IterableDataset."""
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .config import Config
from .track import Track
from . import gt, render, augment


def crop(img: np.ndarray, cfg: Config) -> np.ndarray:
    return img[cfg.camera.image_height // 2:]


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_bgr)).permute(2, 0, 1).float().div_(255.0)


def make_sample(track: Track, cfg: Config, rng: np.random.Generator, do_augment: bool = True):
    """Return (img_crop_bgr, waypoints (K,2) m, pose). Pure numpy; shared by dataset and evaluation."""
    pose = gt.sample_pose(track, cfg, rng)
    if do_augment:
        H = augment.jitter_pitch(cfg, rng)
        quads = augment.dropout_quads(track.quads, cfg, rng)
    else:
        from .camera import build
        H = build(cfg)[0]
        quads = track.quads
    img = render.render(pose, quads, None, H, cfg)
    if do_augment:
        img = augment.augment_image(img, cfg, rng)
    return crop(img, cfg), gt.waypoints_ahead(pose, track, cfg), pose


class SynthDataset(IterableDataset):
    def __init__(self, track: Track, cfg: Config, seed: int = 0, augment: bool = True):
        self.track, self.cfg, self.seed, self.augment = track, cfg, seed, augment

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        norm = self.cfg.waypoints.norm_m
        while True:
            img, wp, _ = make_sample(self.track, self.cfg, rng, self.augment)
            yield to_tensor(img), torch.from_numpy(wp.reshape(-1) / norm).float()
