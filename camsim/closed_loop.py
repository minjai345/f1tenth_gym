"""f110_gym 안에서 렌더 -> 추론 -> pure pursuit -> step. ROS 없음. 지연은 제어 틱 단위 버퍼로 주입.

종료 조건 (Result.reason): "lap" 한 바퀴 완주 / "tape_crossed" 차체 모서리가 테이프 안쪽 선을 넘음(실차 실격 규칙,
gt.crosses_tape) / "collision" gym 벽 충돌 / "max_steps". 실차 트랙은 벽 없이 테이프만 있으므로 실격 판정은 벽이 아니라 테이프 기준이다.
"""
from collections import deque
from dataclasses import dataclass
import os
import warnings
import cv2
import numpy as np
from .config import Config
from .track import Track
from . import gt, render, viz
from .pure_pursuit import pure_pursuit
from .model import OraclePredictor


def make_env(cfg: Config):
    import gym
    from f110_gym.envs.base_classes import Integrator
    base = os.path.splitext(cfg.closed_loop.map_yaml)[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Chosen integrator is RK4.*")
        return gym.make("f110_gym:f110-v0", map=base, map_ext=".png", num_agents=1,
                        timestep=0.01, integrator=Integrator.RK4)


def pose_of(obs) -> np.ndarray:
    return np.array([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]])


@dataclass
class Result:
    finished: bool
    reason: str
    steps: int
    mean_lateral_m: float
    max_lateral_m: float
    progress_m: float
    control_hz_eff: float
    time_s: float = 0.0                # 주행 시간 (초). 랩타임 비교용
    lateral_trace: np.ndarray = None   # 제어 틱마다 중심선 대비 횡오차 (m). 그래프용
    pose_trace: np.ndarray = None      # (steps,3) 제어 틱마다 world pose. 맵 위 경로 그리기용


def _unwrap_progress(track: Track, s_prev: float, s_now: float) -> float:
    ds = s_now - s_prev
    if ds < -track.length / 2: ds += track.length
    if ds > track.length / 2: ds -= track.length
    return ds


def run(env, predictor, track: Track, cfg: Config, H_g2i: np.ndarray, start_index: int = 0,
        latency_steps=None, video_path=None, seed: int = 0) -> Result:
    """Run one closed-loop episode: render -> predict -> pure pursuit -> env.step.

    seed is reserved for future stochastic reset/noise injection and is currently unused.
    """
    cl = cfg.closed_loop
    if latency_steps is None:
        latency_steps = cl.latency_steps
    k = len(cfg.waypoints.ahead_m)
    physics_per_tick = max(1, int(round(1.0 / (env.timestep * cl.control_hz))))
    # The actual control rate, which only equals cl.control_hz when control_hz evenly
    # divides gym's 1/env.timestep physics rate (physics_per_tick is rounded to an int).
    hz_eff = 1.0 / (env.timestep * physics_per_tick)
    if abs(hz_eff - cl.control_hz) / cl.control_hz > 0.02:
        warnings.warn(
            f"camsim: control_hz={cl.control_hz} is not an integer divisor of the gym physics "
            f"rate (1/{env.timestep}); effective control rate is {hz_eff:.3f} Hz instead",
            UserWarning,
        )

    p0 = track.center[start_index]
    obs, _, done, _ = env.reset(np.array([[p0[0], p0[1], track.heading[start_index]]]))
    buf = deque([np.column_stack([np.asarray(cfg.waypoints.ahead_m), np.zeros(k)])] * latency_steps)

    mask = render.bev_visibility_mask(H_g2i, cfg)
    writer = None            # 영상 프레임 = [카메라 뷰 | 모델 입력 BEV], 첫 프레임 크기로 열림

    lats, traveled, reason = [], 0.0, "max_steps"
    poses = []
    s_prev = track.s[gt.nearest_index(track, p0)]
    steps = 0
    try:
        for steps in range(1, cl.max_steps + 1):
            pose = pose_of(obs)
            bev = render.render_bev(pose, track.quads, cfg, mask)          # 모델 입력 (시뮬 BEV)
            if hasattr(predictor, "set_pose"):
                predictor.set_pose(pose)
            buf.append(predictor.predict(bev))
            wp = buf.popleft()
            steer = pure_pursuit(wp, cl.lookahead_m, cl.wheelbase_m, cl.steer_max_rad)
            if video_path is not None:
                cam = render.draw_points(render.render(pose, track.quads, obs["scans"][0], H_g2i, cfg), wp, H_g2i)
                frame = viz.side_by_side(cam, render.draw_points_bev(bev.copy(), wp, cfg))
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), hz_eff,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            for _ in range(physics_per_tick):
                obs, _, done, _ = env.step(np.array([[steer, cl.speed_mps]]))
                if done:
                    break
            pose = pose_of(obs)
            lat = gt.lateral_error(track, pose[:2])
            lats.append(lat)
            poses.append(pose)
            s_now = track.s[gt.nearest_index(track, pose[:2])]
            traveled += _unwrap_progress(track, s_prev, s_now)
            s_prev = s_now
            if obs["collisions"][0]:
                reason = "collision"; break
            if gt.crosses_tape(pose, track, cfg):          # 실차 규칙: 테이프를 넘으면 실격
                reason = "tape_crossed"; break
            if traveled >= track.length:
                reason = "lap"; break
    finally:
        if writer is not None:
            writer.release()
    lats = np.array(lats) if lats else np.zeros(1)
    return Result(reason == "lap", reason, steps, float(lats.mean()), float(lats.max()), float(traveled),
                  hz_eff, steps / hz_eff, lats, np.array(poses).reshape(-1, 3))


def sweep(env, track: Track, cfg: Config, H_g2i, latency_list, sigma_list, predictor_factory=None):
    rows = []
    for lat in latency_list:
        for sig in sigma_list:
            pred = (predictor_factory(sig) if predictor_factory
                    else OraclePredictor(track, cfg, noise_sigma=sig))
            r = run(env, pred, track, cfg, H_g2i, latency_steps=lat)
            rows.append({"latency_steps": lat, "sigma": sig, "finished": r.finished, "reason": r.reason,
                         "mean_lateral_m": r.mean_lateral_m, "max_lateral_m": r.max_lateral_m,
                         "steps": r.steps, "progress_m": r.progress_m})
            print(f"latency={lat:2d} sigma={sig:.2f} -> {r.reason:9s} lat_mean={r.mean_lateral_m:.3f} "
                  f"lat_max={r.max_lateral_m:.3f} progress={r.progress_m:.1f}m", flush=True)
    return rows
