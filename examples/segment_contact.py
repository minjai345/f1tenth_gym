"""Drive a car into a wall and watch it scrape instead of stopping dead.

``SEGMENT_CONTACT`` removes only the component into the surface; ``LIDAR_SCAN``
zeroes the whole velocity vector. Run both::

    python examples/segment_contact.py --mode contact
    python examples/segment_contact.py --mode halt
    python examples/segment_contact.py --compare        # no window, numbers only
"""

import argparse
import math

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
from f1tenth_gym.envs.track.walls import wall_segments

MODES = {"contact": CollisionCheckMode.SEGMENT_CONTACT, "halt": CollisionCheckMode.LIDAR_SCAN}


def build_config(mode, friction, restitution):
    return EnvConfig(
        map_name="Spielberg",
        simulation_config=SimulationConfig(max_laps=None),
        # terminate_on_collision=False is what makes the response observable at all;
        # with it True the episode ends on the first contact step either way.
        termination_config=TerminationConfig(terminate_on_collision=False),
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.ACCL,
            steering_mode=SteerActionType.STEERING_SPEED,
        ),
        contact_config=ContactConfig(friction=friction, restitution=restitution),
        collision_check=MODES[mode],
        render_enabled=True,
    )


def aim_at_a_wall(env, degrees, speed):
    """Place the car `standoff` metres off the longest wall, pointed at it."""
    walls = wall_segments(env.unwrapped.track)
    k = int(np.argmax(walls.length))
    normal = walls.n[k].astype(np.float64)
    tangent = np.array([-normal[1], normal[0]])
    midpoint = 0.5 * (walls.a[k] + walls.b[k]).astype(np.float64)
    theta = math.radians(degrees)
    heading = -normal * math.cos(theta) + tangent * math.sin(theta)
    start = midpoint + normal * 2.0
    state = np.zeros((1, 7))
    state[0] = [start[0], start[1], 0.0, speed, math.atan2(heading[1], heading[0]), 0.0, 0.0]
    return state


def run(mode, degrees, speed, steps, friction, restitution, render):
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=build_config(mode, friction, restitution),
        render_mode="human" if render else None,
    )
    env.reset(seed=7)
    env.reset(seed=7, options={"states": aim_at_a_wall(env, degrees, speed)})
    sim = env.unwrapped.sim
    coast = np.array([[0.0, 0.0]], dtype=np.float32)

    entry_speed = None
    contact_point = None
    contact_steps = 0
    for _ in range(steps):
        before = float(sim.state.standard_state[0][3])
        _obs, _reward, _term, _trunc, info = env.step(coast)
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
    parser.add_argument("--angle", type=float, default=60.0,
                        help="approach angle off the wall normal; 90 grazes, 0 is head-on")
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--restitution", type=float, default=0.0)
    parser.add_argument("--compare", action="store_true",
                        help="run both modes over a sweep of angles, no window")
    args = parser.parse_args()

    if args.compare:
        print("how far the car travels after touching the wall: a scrape carries on,")
        print("a halt does not.\n")
        print(f"{'angle':>6} {'mode':>9} {'entry':>7} {'travelled':>10} "
              f"{'final':>7} {'contact steps':>14}")
        for degrees in (30.0, 60.0, 75.0):
            for mode in ("halt", "contact"):
                entry, travelled, final, held = run(mode, degrees, args.speed, 250,
                                                    args.friction, args.restitution,
                                                    render=False)
                if entry is None:
                    print(f"{degrees:6.0f} {mode:>9}   never reached the wall")
                    continue
                print(f"{degrees:6.0f} {mode:>9} {entry:7.2f} {travelled:9.2f}m "
                      f"{final:7.2f} {held:14d}")
        return

    entry, travelled, final, held = run(args.mode, args.angle, args.speed, args.steps,
                                        args.friction, args.restitution, render=True)
    if entry is None:
        print("the car never reached the wall -- try a smaller --angle")
        return
    print(f"mode={args.mode} angle={args.angle:.0f} deg")
    print(f"  touched the wall at {entry:.2f} m/s, then travelled {travelled:.2f} m")
    print(f"  final {final:.2f} m/s after {held} steps in contact")


if __name__ == "__main__":
    main()
