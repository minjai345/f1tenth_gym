"""Crash two cars into each other and watch the impulses resolve.

``SEGMENT_CONTACT`` gives both bodies a manifold and a two-body impulse, so a
head-on stops both cars and a rear-end shares the momentum. ``BOUNDING_BOX``
detects the same overlap and does nothing about it, so the cars drive through
each other at unchanged speed. Run::

    python examples/two_agent_contact.py --scenario head-on
    python examples/two_agent_contact.py --scenario rear-end --restitution 0.5
    python examples/two_agent_contact.py --compare        # no window, numbers only

The track is ``Spielberg_blank``, which has no obstacles at all, so nothing here
is contaminated by wall contact.
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

MODES = {"contact": CollisionCheckMode.SEGMENT_CONTACT,
         "passthrough": CollisionCheckMode.BOUNDING_BOX}


def _state(x, y, yaw, speed):
    """A full ST state row: [x, y, delta, v, yaw, yaw_rate, beta]."""
    return [x, y, 0.0, speed, yaw, 0.0, 0.0]


# Each pair is placed so the two bodies arrive together; coasting from there means
# the only thing that acts on them is the contact impulse.
SCENARIOS = {
    "head-on": (_state(0.0, 0.0, 0.0, 3.0), _state(6.0, 0.0, math.pi, 3.0)),
    "rear-end": (_state(0.0, 0.0, 0.0, 6.0), _state(3.0, 0.0, 0.0, 1.0)),
    "side-swipe": (_state(0.0, 0.0, 0.0, 5.0), _state(0.0, 0.42, -0.10, 5.0)),
    "t-bone": (_state(0.0, 0.0, 0.0, 4.0), _state(3.0, -3.0, math.pi / 2, 4.0)),
}


def build_config(mode, friction, restitution):
    return EnvConfig(
        map_name="Spielberg_blank",
        num_agents=2,
        simulation_config=SimulationConfig(max_laps=None),
        # Without this the episode ends on the first contact step and the response
        # is never visible.
        termination_config=TerminationConfig(terminate_on_collision=False),
        # Coast: no throttle, no steering rate, so contact is the only force.
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.ACCL,
            steering_mode=SteerActionType.STEERING_SPEED,
        ),
        contact_config=ContactConfig(friction=friction, restitution=restitution),
        collision_check=MODES[mode],
        render_enabled=True,
    )


def run(scenario, mode, steps, friction, restitution, render):
    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=build_config(mode, friction, restitution),
        render_mode="human" if render else None,
    )
    start = np.array(SCENARIOS[scenario])
    env.reset(seed=1, options={"states": start})
    sim = env.unwrapped.sim
    coast = np.zeros((2, 2), dtype=np.float32)

    contact_steps = 0
    peak_spin = np.zeros(2)
    for _ in range(steps):
        _obs, _reward, _term, _trunc, info = env.step(coast)
        if np.any(info["collisions"]):
            contact_steps += 1
        peak_spin = np.maximum(peak_spin, np.abs(sim.state.standard_state[:, 5]))
        if render:
            env.render()
    exit_speed = sim.state.standard_state[:, 3].astype(np.float64).copy()
    env.close()
    return start[:, 3], exit_speed, contact_steps, peak_spin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="head-on")
    parser.add_argument("--mode", choices=sorted(MODES), default="contact")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--restitution", type=float, default=0.0,
                        help="0 is a dead stop, 1 a perfect bounce")
    parser.add_argument("--compare", action="store_true",
                        help="every scenario in both modes, no window")
    args = parser.parse_args()

    if args.compare:
        print("entry and exit speeds, m/s. BOUNDING_BOX flags the overlap and lets")
        print("the cars pass through it; SEGMENT_CONTACT resolves it.\n")
        print(f"{'scenario':<12} {'mode':<12} {'e':>4} {'entry':>13} {'exit':>17} "
              f"{'contact':>8} {'peak yaw rate':>15}")
        for scenario in sorted(SCENARIOS):
            for mode, restitution in (("passthrough", 0.0), ("contact", 0.0),
                                      ("contact", 0.5)):
                entry, exit_, held, spin = run(scenario, mode, args.steps,
                                               args.friction, restitution, render=False)
                print(f"{scenario:<12} {mode:<12} {restitution:>4.1f} "
                      f"{np.array2string(entry, precision=2):>13} "
                      f"{np.array2string(exit_, precision=2):>17} {held:>8} "
                      f"{np.array2string(spin, precision=1):>15}")
        return

    entry, exit_, held, spin = run(args.scenario, args.mode, args.steps,
                                   args.friction, args.restitution, render=True)
    print(f"{args.scenario} / {args.mode} / restitution {args.restitution}")
    print(f"  entry {np.array2string(entry, precision=2)} m/s"
          f" -> exit {np.array2string(exit_, precision=2)} m/s")
    print(f"  {held} steps in contact, peak yaw rate "
          f"{np.array2string(spin, precision=2)} rad/s")


if __name__ == "__main__":
    main()
