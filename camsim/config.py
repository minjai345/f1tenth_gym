"""config.yaml -> typed Config. 키 누락/오타는 ConfigError로 어떤 키인지 알려준다."""
from dataclasses import dataclass, fields
from typing import List, Optional
import os, yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


class ConfigError(Exception):
    pass


@dataclass
class Camera:
    image_width: int
    image_height: int
    sensor_width_mm: float
    sensor_height_mm: float
    hfov_deg: float
    height_m: float
    pitch_deg: float
    offset_x_m: float
    h_i2g_file: Optional[str]


@dataclass
class Lane:
    track_width_m: float
    tape_width_m: float
    segment_len_m: float
    color_floor: List[int]
    color_tape: List[int]


@dataclass
class Render:
    near_m: float
    far_m: float
    lidar_fov_rad: float


@dataclass
class Waypoints:
    ahead_m: List[float]
    norm_m: float


@dataclass
class Sampling:
    lateral_frac: float
    heading_deg: float


@dataclass
class Augment:
    pitch_jitter_deg: float
    blur_max_px: int
    tape_dropout_prob: float
    tape_dropout_len: List[int]
    glare_prob: float


@dataclass
class ClosedLoop:
    map_yaml: str
    centerline_csv: str
    control_hz: int
    lookahead_m: float
    speed_mps: float
    latency_steps: int
    max_steps: int
    offtrack_m: float
    wheelbase_m: float
    steer_max_rad: float


@dataclass
class Config:
    camera: Camera
    lane: Lane
    render: Render
    waypoints: Waypoints
    sampling: Sampling
    augment: Augment
    closed_loop: ClosedLoop
    path: str = DEFAULT_PATH


def _build(cls, section: str, data):
    if not isinstance(data, dict):
        raise ConfigError(f"section '{section}' must be a mapping")
    names = {f.name for f in fields(cls)}
    missing = [f"{section}.{n}" for n in names if n not in data]
    extra = [f"{section}.{k}" for k in data if k not in names]
    if missing or extra:
        raise ConfigError(f"missing keys: {missing}; unknown keys: {extra}")
    return cls(**data)


def load(path: Optional[str] = None) -> Config:
    path = path or DEFAULT_PATH
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    sections = {f.name: f.type for f in fields(Config) if f.name != "path"}
    unknown = [k for k in raw if k not in sections]
    if unknown:
        raise ConfigError(f"unknown top-level keys: {unknown}")
    built = {}
    for name, cls in sections.items():
        if name not in raw:
            raise ConfigError(f"missing section '{name}'")
        built[name] = _build(cls, name, raw[name])
    return Config(path=path, **built)
