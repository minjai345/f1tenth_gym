Quickstart
==========

An episode ends when the simulator decides it does — on a crash, or once the lap
target is met — not when your loop runs out of iterations. Driving one to that
end with no step budget of your own is the fastest way to see which of the two
conditions fired, and to read enough of the car's state to do something about it.

Drive one episode
-----------------

A default :class:`~f1tenth_gym.envs.env_config.EnvConfig` places a single car on
the Spielberg track under single-track dynamics, with speed and steering-angle
commands and the ``DEFAULT`` observation preset. The loop below holds one
constant command and stops only when the environment says to:

>>> import gymnasium as gym
>>> import numpy as np
>>> from f1tenth_gym.envs.env_config import EnvConfig
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> obs, info = env.reset(seed=42)
>>> action = np.array([[0.0, 2.0]], dtype=np.float32)   # [[steering, speed]]
>>> terminated = truncated = False
>>> steps = 0
>>> while not (terminated or truncated):
...     obs, reward, terminated, truncated, info = env.step(action)
...     steps += 1
>>> print(steps, round(info["sim_time"], 2))
1787 17.87

The ``f1tenth_gym:`` prefix on the id is an instruction to import the package,
whose side effect registers ``f1tenth-v0``; after an explicit
``import f1tenth_gym`` the bare id works too. ``seed=42`` pins the spawn
waypoint, so the run above replays step for step — :doc:`reproducibility` covers
what a seed reaches, and how ``EnvConfig(seed=...)`` covers the first ``reset``
you leave unseeded.

Find out why it ended
---------------------

Two conditions raise ``terminated``: contact, since ``terminate_on_collision``
is on by default, and the ego reaching ``max_laps``. ``truncated`` is separate
and comes only from ``max_episode_steps``, which is ``None`` unless you set it.
The final ``info`` names the one that fired — it carries a per-agent array for
each:

>>> terminated, truncated
(True, False)
>>> info["collisions"], info["lap_counts"]
(array([1.], dtype=float32), array([0.]))

The car hit a wall after 17.87 simulated seconds without completing a lap, so
the default ``max_laps=1`` was never in play. That figure is also the episode
return: the default reward mode pays the physics timestep once per step, making
the return pure survival time (:doc:`rl`). ``info`` from ``reset`` is smaller
than ``info`` from ``step`` — ``collisions`` and ``progress`` appear only after
a step (:doc:`observations`).

Every config object is frozen, so changing the lap limit means rebuilding that
branch of the tree with ``with_updates`` rather than assigning to it:

>>> base = EnvConfig(render_enabled=False)
>>> cfg = base.with_updates(
...     simulation_config=base.simulation_config.with_updates(max_laps=None)
... )
>>> cfg.simulation_config.max_laps is None
True

With the lap exit removed, a step limit is the exit you control. It ends the
episode through ``truncated``, which keeps it distinguishable from a crash —
bootstrapping RL algorithms depend on that distinction:

>>> from f1tenth_gym.envs.env_config import TerminationConfig
>>> cfg = cfg.with_updates(
...     termination_config=TerminationConfig(max_episode_steps=500)
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> obs, info = env.reset(seed=42)
>>> terminated = truncated = False
>>> steps = 0
>>> while not (terminated or truncated):
...     obs, reward, terminated, truncated, info = env.step(action)
...     steps += 1
>>> steps, terminated, truncated
(500, False, True)
>>> env.close()

Read the car's state
--------------------

Observations nest by agent id under ``agent_0``, ``agent_1``, …, and the default
preset hands back eight fields per agent:

>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> obs, info = env.reset(seed=42)
>>> print(*sorted(obs["agent_0"]))
collision frenet_pose lap_count lap_time scan sim_time state std_state

Position, heading and speed ride inside the seven-element ``std_state``, laid
out ``[X, Y, delta, speed, yaw, yaw_rate, beta]``:

>>> std = obs["agent_0"]["std_state"]
>>> print(f"x={std[0]:.4f} y={std[1]:.4f} yaw={std[4]:.4f}")
x=-0.0441 y=-0.8492 yaw=-2.8798

Scalars arrive as 0-d ``float32`` arrays rather than Python floats, so wrap a
read in ``float(...)`` where a number is wanted. The first observation of an
episode is already a real measurement rather than a placeholder: ``reset`` runs
one LiDAR sweep, noise included, and deliberately does not adjudicate collisions
at the spawn pose.

>>> scan = obs["agent_0"]["scan"]
>>> print(scan.shape, round(float(scan.min()), 3), float(scan.max()))
(1080,) 0.284 30.0
>>> env.close()

``pose_x``, ``pose_y`` and ``pose_theta`` are derived fields the default preset
leaves out, and reading one raises ``KeyError``. Switching to
``KINEMATIC_STATE`` supplies them as named scalars at the cost of everything
else — five fields, no ``scan``, no ``frenet_pose``:

>>> from f1tenth_gym.envs.env_config import ObservationConfig
>>> from f1tenth_gym.envs.observation import ObservationType
>>> cfg = EnvConfig(
...     observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
...     render_enabled=False,
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> obs, info = env.reset(seed=42)
>>> print(*sorted(obs["agent_0"]))
delta linear_vel_x pose_theta pose_x pose_y
>>> env.close()

Every preset, the shape and dtype of each field, and the bounds the space
declares are listed in :doc:`observations`.

Command the car
---------------

Both action columns are setpoints an actuator chases, not values written into
the physics. One step of a left-hand command at 2 m/s, starting from rest,
moves the steering angle 0.032 rad and the speed to 0.019 m/s:

>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> obs, info = env.reset(seed=42)
>>> obs, reward, terminated, truncated, info = env.step(
...     np.array([[0.2, 2.0]], dtype=np.float32)
... )
>>> std = obs["agent_0"]["std_state"]
>>> print(f"delta={std[2]:.4f} speed={std[3]:.4f}")
delta=0.0320 speed=0.0190
>>> env.close()

That 0.032 rad is exactly ``sv_max * timestep``: the steering rate saturates at
3.2 rad/s and one step lasts 0.01 s. The first four steps run at that limit,
after which the proportional controller eases in and the 0.2 rad target is met
to within a milliradian by step 13. Plan for a command to take several steps to
be realised, and read column 0 as steering and column 1 as the longitudinal
command — both are float32 with overlapping ranges, so a swapped pair is a
perfectly valid action that the simulator executes as given (:doc:`actions`).

Next: :doc:`examples` walks the bundled pure-pursuit follower, which is this
loop with a controller in place of the constant action.
