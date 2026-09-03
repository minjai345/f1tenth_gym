"""작은 CNN과, 폐루프/ROS2 노드가 공유하는 predict(image) -> (K,2) m 래퍼."""
import numpy as np
import torch
import torch.nn as nn
from .config import Config
from .track import Track
from . import gt
from .dataset import crop, to_tensor


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class WaypointNet(nn.Module):
    def __init__(self, n_out: int = 12):
        super().__init__()
        self.features = nn.Sequential(_block(3, 16), _block(16, 32), _block(32, 64),
                                      _block(64, 128), _block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(),
                                  nn.Linear(128 * 8, 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, n_out))

    def forward(self, x):
        return self.head(self.features(x))


class Predictor:
    def __init__(self, net: nn.Module, cfg: Config, device: str = "cpu"):
        self.net, self.cfg, self.device = net.to(device).eval(), cfg, device
        self.k = len(cfg.waypoints.ahead_m)

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        x = to_tensor(crop(img_bgr, self.cfg))[None].to(self.device)
        y = self.net(x)[0].cpu().numpy() * self.cfg.waypoints.norm_m
        return y.reshape(self.k, 2)


class OraclePredictor:
    """GT waypoints (+ optional gaussian noise). Lets the closed loop run before any model exists."""
    def __init__(self, track: Track, cfg: Config, noise_sigma: float = 0.0, rng=None):
        self.track, self.cfg, self.sigma = track, cfg, noise_sigma
        self.rng = rng or np.random.default_rng(0)
        self.pose = None

    def set_pose(self, pose):
        self.pose = np.asarray(pose, float)

    def predict(self, img_bgr) -> np.ndarray:
        wp = gt.waypoints_ahead(self.pose, self.track, self.cfg)
        if self.sigma > 0:
            wp = wp + self.rng.normal(0, self.sigma, wp.shape)
        return wp


def save(net: nn.Module, path) -> None:
    torch.save({"state_dict": net.state_dict(), "n_out": net.head[-1].out_features}, path)


def load(path, cfg: Config) -> WaypointNet:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = WaypointNet(n_out=ck["n_out"])
    net.load_state_dict(ck["state_dict"])
    return net
