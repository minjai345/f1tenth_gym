"""학습 루프(Huber)와 오프라인 지표(waypoint별 오차 m)."""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .track import Track
from .dataset import SynthDataset, make_sample
from . import model as M


def evaluate(predictor, track: Track, cfg: Config, n: int = 200, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        img, wp, pose = make_sample(track, cfg, rng, do_augment=False)
        if hasattr(predictor, "set_pose"):
            predictor.set_pose(pose)
        full = np.empty((cfg.camera.image_height, cfg.camera.image_width, 3), np.uint8)
        full[:] = cfg.lane.color_floor
        full[cfg.camera.image_height // 2:] = img
        pred = predictor.predict(full)
        errs.append(np.hypot(*(pred - wp).T))
    errs = np.array(errs)
    per = errs.mean(0)
    return {"per_waypoint_m": per, "mean_m": float(per.mean()), "max_m": float(errs.max())}


def train(track: Track, cfg: Config, steps: int, batch_size: int = 32, lr: float = 1e-3,
          device: str = "cpu", out_path=None, num_workers: int = 0, log_every: int = 50, seed: int = 0):
    torch.manual_seed(seed)
    net = M.WaypointNet(n_out=2 * len(cfg.waypoints.ahead_m)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = torch.nn.SmoothL1Loss()
    dl = DataLoader(SynthDataset(track, cfg, seed=seed), batch_size=batch_size, num_workers=num_workers)
    history, run, t0 = [], 0.0, time.time()
    net.train()
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if step % log_every == 0:
            history.append({"step": step, "loss": run / log_every, "sec": time.time() - t0})
            print(f"step {step:6d}  loss {run / log_every:.4f}  {time.time() - t0:6.0f}s", flush=True)
            run = 0.0
        if step >= steps:
            break
    net.eval()
    if out_path is not None:
        M.save(net, out_path)
    return net, history
