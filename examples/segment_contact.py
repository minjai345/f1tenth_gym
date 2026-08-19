"""Drive a car into a wall and watch it scrape instead of stopping dead.

``SEGMENT_CONTACT`` removes only the component into the surface; ``LIDAR_SCAN``
zeroes the whole velocity vector. Run both::

    python examples/segment_contact.py --mode contact
    python examples/segment_contact.py --mode halt
    python examples/segment_contact.py --compare        # no window, numbers only
"""

import argparse

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.contact import ContactConfig
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    EnvConfig,
    SimulationConfig,
    TerminationConfig,
)

MODES = {"contact": CollisionCheckMode.SEGMENT_CONTACT, "halt": CollisionCheckMode.LIDAR_SCAN}


def build_config(mode, friction, restitution):
    return EnvConfig(
        map_name="Spielberg",
        simulation_config=SimulationConfig(max_laps=None),
        # terminate_on_collision=False is what makes the response observable at all;
        # with it True the episode ends on the first contact step either way.
        termination_config=TerminationConfig(terminate_on_collision=False),
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.SPEED,
            steering_mode=SteerActionType.STEERING_ANGLE,
        ),
        contact_config=ContactConfig(friction=friction, restitution=restitution),
        collision_check=MODES[mode],
        render_enabled=True,
    )


def run(mode, steer, speed, steps, friction, restitution, render):
    """Spawn on the raceline, hold a steering angle, and drive into the boundary.

    Hand-placing the car near a wall is a trap: the corridor is often narrower than
    it looks, and a 2 m standoff from one wall can put the body 0.23 m inside
    another, so it starts in contact and the run tells you nothing.
    """
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=build_config(mode, friction, restitution),
        render_mode="human" if render else None,
    )
    env.reset(seed=7)
    sim = env.unwrapped.sim
    action = np.array([[steer, speed]], dtype=np.float32)

    entry_speed = None
    contact_point = None
    contact_steps = 0
    for _ in range(steps):
        before = float(sim.state.standard_state[0][3])
        _obs, _reward, _term, _trunc, info = env.step(action)
        if info["collisions"][0]:
            contact_steps += 1
            if entry_speed is None:
                entry_speed = before
                contact_point = sim.state.standard_state[0][:2].astype(np.float64).copy()
        if render:
            env.render()
    final = sim.state.standard_state[0]
    travelled = 0.0 if contact_point is None else float(
        np.hypot(*(final[:2].astype(np.float64) - contact_point))
    )
    env.close()
    return entry_speed, travelled, float(final[3]), contact_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), default="contact")
    parser.add_argument("--steer", type=float, default=0.18,
                        help="held steering angle in radians; larger turns into the wall sooner")
    parser.add_argument("--speed", type=float, default=6.0, help="target speed, m/s")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--compare", action="store_true",
                        help="run both modes over a sweep of angles, no window")
    args = parser.parse_args()

    if args.compare:
        print("how far the car travels after touching the wall: a scrape carries on,")
        print("a halt does not.\n")
        print(f"{'steer':>6} {'mode':>9} {'entry':>7} {'travelled':>10} "
              f"{'final':>7} {'contact steps':>14}")
        for steer in (0.10, 0.18, 0.30):
            for mode in ("halt", "contact"):
                entry, travelled, final, held = run(mode, steer, args.speed, 400,
                                                    args.friction, args.restitution,
                                                    render=False)
                if entry is None:
                    print(f"{steer:6.2f} {mode:>9}   never reached the wall")
                    continue
                print(f"{steer:6.2f} {mode:>9} {entry:7.2f} {travelled:9.2f}m "
                      f"{final:7.2f} {held:14d}")
        return

    entry, travelled, final, held = run(args.mode, args.steer, args.speed, args.steps,
                                        args.friction, args.restitution, render=True)
    if entry is None:
        print("the car never reached the wall -- try a larger --steer")
        return
    print(f"mode={args.mode} steer={args.steer:.2f} rad")
    print(f"  touched the wall at {entry:.2f} m/s, then travelled {travelled:.2f} m")
    print(f"  final {final:.2f} m/s after {held} steps in contact")


if __name__ == "__main__":
    main()
