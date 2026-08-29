How to train an RL agent against this environment
=================================================

The environment is natively multi-agent: ``reset`` hands back
``dict[agent_id -> dict[field -> ndarray]]`` and ``step`` expects a
``(num_agents, 2)`` action array. Two wrappers and one reward mode collapse
that into the flat, finite ``Box`` pair that Stable-Baselines3, CleanRL, and
``gymnasium.utils.env_checker.check_env`` all assume.

Build the flat single-agent env
-------------------------------

Compose the stack and drive it for 200 steps:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from gymnasium.wrappers import FlattenObservation

   from f1tenth_gym.envs.env_config import (
       EnvConfig, ObservationConfig, RewardConfig, RewardMode, SimulationConfig,
   )
   from f1tenth_gym.envs.observation import ObservationType
   from f1tenth_gym.envs.wrappers import SingleAgentWrapper

   cfg = EnvConfig(
       observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
       simulation_config=SimulationConfig(max_laps=None),  # default ends after 1 lap
       reward_config=RewardConfig(
           mode=RewardMode.PROGRESS,
           progress_weight=1.0,
           velocity_weight=0.1,
           collision_penalty=10.0,
       ),
       render_enabled=False,
   )

   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   env = SingleAgentWrapper(env)
   env = FlattenObservation(env)

   obs, info = env.reset(seed=42)
   print("obs:", obs.shape, obs.dtype)
   total = 0.0
   for _ in range(200):
       action = np.array([0.0, 2.0], dtype=np.float32)  # flat (2,): steer, speed
       obs, reward, terminated, truncated, info = env.step(action)
       total += reward
       if terminated or truncated:
           break

   print("return:", round(total, 4))
   print("progress:", round(float(info["progress"][0]), 5))
   env.close()

This prints ``obs: (5,) float32``, ``return: 39.4076`` and
``progress: 0.02`` — the five ``KINEMATIC_STATE`` fields arrive as one
float32 vector, and at 2 m/s the car gains 0.02 m of track per 10 ms step.
:class:`~f1tenth_gym.envs.wrappers.SingleAgentWrapper` requires
``num_agents == 1`` (``ValueError`` otherwise), unwraps ``obs["agent_0"]``,
and reshapes the flat ``(2,)`` action — steering first — back to the native
``(1, 2)``. It never clips: a command outside the ``Box`` is executed as-is
(:doc:`actions`). ``FlattenObservation`` can promise a finite ``Box`` because
every per-field bound is derived from the vehicle and track
(:doc:`observations`), and the composed stack passes
``check_env(env, skip_render_check=True)``.

Choose what the policy sees
---------------------------

The flattened dimension is set by the observation preset, and the scan
dominates it:

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - ``ObservationType``
     - Flat shape
     - Contents
   * - ``KINEMATIC_STATE``
     - ``(5,)``
     - planar pose, steering angle, forward speed
   * - ``DYNAMIC_STATE``
     - ``(7,)``
     - adds yaw rate and slip angle; speed becomes a magnitude
   * - ``FRENET_DYNAMIC_STATE``
     - ``(8,)``
     - splits speed into body-frame vx/vy (no Frenet field despite the name)
   * - ``DEFAULT``
     - ``(1101,)``
     - 1080 scan beams plus 21 state scalars

Training on ``DEFAULT`` means an 1101-dimensional input that is 98% LiDAR —
pick a state preset unless the policy genuinely consumes the scan. The field
vocabulary, dtypes and bounds live in :doc:`observations`; two keys are
conditional — ``scan`` is dropped when ``lidar_config.enabled=False`` and
``frenet_pose`` when ``compute_frenet_frame=False`` — so the ``DEFAULT`` flat
size shrinks with them.

Flat indices follow the observation *space*, which gymnasium's ``Dict`` sorts
lexicographically — not the preset declaration order:

>>> import gymnasium as gym
>>> from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig
>>> from f1tenth_gym.envs.observation import ObservationType
>>> from f1tenth_gym.envs.wrappers import SingleAgentWrapper
>>> cfg = EnvConfig(
...     observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
...     render_enabled=False,
... )
>>> env = SingleAgentWrapper(gym.make("f1tenth_gym:f1tenth-v0", config=cfg))
>>> list(env.observation_space.keys())
['delta', 'linear_vel_x', 'pose_theta', 'pose_x', 'pose_y']
>>> env.close()

Index a flattened observation by ``sorted(fields)``, never by the preset
tuple. ``ObservationType.DIRECT`` changed meaning in v1.0.0 and warns when
selected: it returns raw agent-batched arrays keyed by name (``scans``,
``standard_state``, ...) with no per-agent level, so ``SingleAgentWrapper``
raises ``KeyError`` on it — use ``DEFAULT`` for the packaged per-agent dict.

Shape the built-in reward
-------------------------

``RewardMode.SURVIVAL``, the default, pays ``timestep`` (0.01 simulated
seconds) per surviving step — episode return is time alive. The other two
modes live on the same :class:`~f1tenth_gym.envs.env_config.RewardConfig`:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - ``RewardMode``
     - Reward returned by ``step``
   * - ``SURVIVAL`` (=0, default)
     - ``timestep`` — 0.01 per step at the defaults.
   * - ``PROGRESS`` (=1)
     - Weighted sum of forward arclength, ego speed, a survival bonus and a
       collision penalty.
   * - ``CUSTOM`` (=2)
     - ``reward_fn(obs, action, info, terminated, truncated) -> float`` — you
       supply the callable.

``PROGRESS`` builds the ego reward from four terms:

.. code-block:: text

   reward =  progress_weight  * (forward Frenet Δs this step, metres)
           + velocity_weight  * (ego speed, m/s)
           + timestep_weight  * timestep
           - collision_penalty        (only on steps where the ego is colliding)

Defaults are ``progress_weight=1.0`` and zero for the rest;
``collision_penalty`` must be ``>= 0``. The progress term is the same
per-agent, wrap-corrected Frenet Δs exposed as ``info["progress"]``: seeded
from the spawn arclength at reset (the first step's progress is ~0, not the
whole spawn ``s``) and independent of the lap counter, so it works with
``max_laps=None`` and any lap-counting mode. Arclength is measured along the
centerline, whatever line the car drives (:doc:`tracks`).

Weigh the terms against their measured magnitudes. With the lead program's
weights, steady driving at 3 m/s returns 0.330 per step — 0.030 progress
+ 0.300 velocity — so ``velocity_weight=0.1`` outweighs the progress term
ten to one, and ``collision_penalty=10.0`` cancels about thirty steps of
driving. The penalty recurs on every step the ego stays in contact, not once
per crash, and the velocity term uses the signed speed: reversing pays
negative reward.

.. warning::

   ``PROGRESS`` needs the Frenet frame:
   :class:`~f1tenth_gym.envs.env_config.EnvConfig` raises ``ValueError`` when
   ``reward_config.mode`` is ``PROGRESS`` while
   ``simulation_config.compute_frenet_frame`` is ``False``. The default
   config computes the frame, so this bites only after you turn it off.

Compute the reward yourself
---------------------------

``RewardMode.CUSTOM`` hands the per-step reward to a callable you supply,
invoked after ``info`` is fully populated for the step:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig, RewardConfig, RewardMode

   def my_reward(obs, action, info, terminated, truncated):
       speed = float(obs["agent_0"]["std_state"][3])
       progress = float(info["progress"][0])
       crashed = bool(info["collisions"][0])
       return progress + 0.05 * speed - (100.0 if crashed else 0.0)

   cfg = EnvConfig(
       reward_config=RewardConfig(mode=RewardMode.CUSTOM, reward_fn=my_reward),
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)
   for _ in range(100):
       obs, reward, terminated, truncated, info = env.step(
           np.array([[0.0, 2.0]], dtype=np.float32)
       )
   print("reward:", round(reward, 4))
   env.close()

This prints ``reward: 0.119`` — 0.020 m of progress plus 0.05 times the
1.984 m/s speed. ``reward_fn`` is required when the mode is ``CUSTOM``
(``ValueError`` at construction otherwise) and is ignored under the other
two modes. The raw material is always there: every ``step`` carries
``info["progress"]`` (per-agent forward arclength, metres) and
``info["collisions"]`` (per-agent flags, ``1.0`` = contact) whatever the
reward mode, alongside copies of ``lap_times``, ``lap_counts`` and
``sim_time``, so a stored ``info`` never mutates retroactively. ``reset``'s
info is smaller — just the three lap/time keys — so read ``progress`` and
``collisions`` only after a ``step``. ``obs`` is the native multi-agent dict
of whatever observation type is configured: under the default it carries
``std_state`` (index 3 is speed), while derived scalars such as
``linear_vel_x`` exist only under presets that include them
(:doc:`observations`).

Scale to many environments
--------------------------

Both wrappers record their constructor arguments
(``gym.utils.RecordConstructorArgs``), so they pickle and survive subprocess
workers. ``gym.make_vec`` rebuilds the whole stack once per worker —
continuing with the ``cfg`` from the doctest above:

.. warning::

   ``vectorization_mode="async"`` forks by default, and the exact scanner
   initialises JAX, which is multithreaded. Forking that deadlocks the
   workers. Pass ``vector_kwargs={"context": "spawn"}`` and call it from a script
   guarded by ``if __name__ == "__main__":`` — spawn re-imports the parent module
   in each worker. The example below uses ``"sync"``, which needs neither.

>>> import numpy as np
>>> from gymnasium.wrappers import FlattenObservation
>>> vec = gym.make_vec(
...     "f1tenth_gym:f1tenth-v0",
...     num_envs=4,
...     vectorization_mode="sync",
...     wrappers=[SingleAgentWrapper, FlattenObservation],
...     config=cfg,
... )
>>> obs, infos = vec.reset(seed=42)
>>> obs.shape
(4, 5)
>>> np.round(obs[:, 3], 4)              # pose_x: a distinct spawn per sub-env
array([-0.0441, -0.4304, -0.6235, -0.8167], dtype=float32)
>>> obs, rewards, terminated, truncated, infos = vec.step(
...     np.tile(np.array([0.0, 2.0], np.float32), (4, 1))
... )
>>> rewards
array([0.01, 0.01, 0.01, 0.01])
>>> vec.close()

``vec.reset(seed=42)`` derives a distinct seed per sub-env, which is why the
four spawn abscissae differ. Do not lean on ``EnvConfig(seed=...)`` for this:
one seeded config shared by every worker makes the workers replay identical
episodes — seed through the vector ``reset`` instead (:doc:`reproducibility`).

What this environment does not ship
-----------------------------------

Planners and training loops are deliberately absent. Pure pursuit, MPC and
other controllers live in the separate ``f1tenth_planning`` repository — the
pure-pursuit follower in ``examples/waypoint_follow.py`` is a demo, not a
supported API — and PPO/SAC training code belongs in ``f1tenth_learning``.
What ships here is the reward surface, the observation presets and the two
wrappers (the second,
:class:`~f1tenth_gym.envs.wrappers.ObservationDelayWrapper`, delays only the
observation to model sensing lag — see :doc:`sim2real`), all configured
through the frozen :class:`~f1tenth_gym.envs.env_config.EnvConfig` tree
(:doc:`configuration`).

Next: :doc:`sim2real` corrupts the command, the car and the sensing this
loop trains on; :doc:`reproducibility` explains what one seed pins when you
A/B those settings.
