How to train an RL agent against this environment
=================================================

There are two supported RL paths. For accelerator-resident rollout storage and
optimization, use the native F1TENTH batch API and the PureJaxRL-style PPO
example. For trainer interoperability, use the conventional Gymnasium boundary
or the optional SBX ``VecEnv`` adapter; those paths cross a NumPy collector
boundary. Neither trainer protocol becomes part of the physics core.

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

Use the JAX-backed Gym adapter
------------------------------

The functional transition also has a conventional Gymnasium boundary.  Import
it directly and use the same wrappers:

.. code-block:: python

   import numpy as np
   from gymnasium.wrappers import FlattenObservation

   from f1tenth_gym.envs.wrappers import SingleAgentWrapper
   from f1tenth_gym.envs.jax_env import JaxF110Env

   env = JaxF110Env(cfg)
   env = SingleAgentWrapper(env)
   env = FlattenObservation(env)

   obs, info = env.reset(seed=42)
   obs, reward, terminated, truncated, info = env.step(
       np.array([0.0, 2.0], dtype=np.float32)
   )
   env.close()

``JaxF110Env`` does not replace the registered environment:
``f1tenth_gym:f1tenth-v0`` still constructs the mutable
:class:`~f1tenth_gym.envs.f110_env.F110Env`.  The JAX-backed class is initially
a deep import rather than another Gymnasium id, so construct it as shown above.
It preserves the native multi-agent action, observation, reward, termination,
reset-option and ``info`` contracts for the functional core's supported config
surface.  Winding-angle laps, ``MAP_RANDOM_STATIC``, non-default active-contact
wall tolerance and mixed active LiDAR/contact devices still fail at
construction rather than falling through to different behavior.

For Stable-Baselines3 or SBX, start with ``KINEMATIC_STATE`` and the two-wrapper
stack above.  It exposes a finite float32 ``Box`` with shape ``(5,)`` and a
float32 action ``Box`` with shape ``(2,)``, so an ``MlpPolicy`` does not need to
understand the native nested agent dictionary.  Select ``DEFAULT`` or a
``FEATURES`` view only when the policy really needs LiDAR.  ``DIRECT`` remains
the exception: its raw arrays have no ``agent_0`` level, so it cannot be passed
through ``SingleAgentWrapper``.

For the primary accelerator-native path, run the PureJaxRL-style PPO example:

.. code-block:: bash

   uv run --extra train python examples/jax_ppo_training.py --device gpu

Its environment rollout, policy, timeout-correct GAE, PPO minibatches and
Optax updates are nested ``lax.scan`` programs under one ``jax.jit``. Domain
randomization is sampled independently for every environment row at reset and
remains in the device ``BatchState``. No Gymnasium or trainer protocol is on
that path; see :doc:`examples` for the task and smoke-test commands.

SBX follows the Stable-Baselines3 environment API. The ordinary
single-environment construction is:

.. code-block:: python

   from sbx import SAC

   env = FlattenObservation(SingleAgentWrapper(JaxF110Env(cfg)))
   model = SAC("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=10_000)
   env.close()

Neither ``sbx-rl`` nor ``stable-baselines3`` is a required dependency of this
gym; install the ``sbx`` extra when using that trainer.  The single-environment
adapter above provides compatibility, not an end-to-end device-native training
path.  Each Gym step runs the compiled JAX transition, then transfers only the
selected observation leaves to independent NumPy arrays for Gymnasium.  SBX
subsequently transfers policy inputs to its own JAX program. No throughput
comparison is claimed for that adapter.

The optional device-batch adapter avoids constructing one Python environment
per row:

.. code-block:: python

   from sbx import PPO

   from f1tenth_gym.envs.sbx import F110SBXVecEnv
   from f1tenth_gym.envs.track import Track

   track = Track.from_track_name("Spielberg", 1.0)
   env = F110SBXVecEnv.from_config(
       cfg,
       track,
       num_envs=256,
       device="gpu",
   )
   model = PPO("MlpPolicy", env, n_steps=128, batch_size=4096)
   model.learn(total_timesteps=1_048_576)
   env.close()

``F110SBXVecEnv`` executes one vmapped transition for the complete batch and
keeps simulator state plus domain-randomized parameters on the selected GPU.
It is deliberately an optional protocol adapter: stock SBX still requires
NumPy actions and observations at each collector step and keeps its SB3 rollout
buffer on the host. Use the native PPO example when rollout storage and
optimization must remain on the accelerator too.

The optional ``tests/test_jax_rl_compat.py`` gate covers the Stable-Baselines3
environment checker and the conventional adapter's SBX update.
``tests/test_sbx_vec_env.py`` separately gates the device-batch contract,
domain-randomization resampling, terminal observations, timeout bootstrapping
and a real SBX PPO rollout. They skip when the optional trainer stack is absent.
The ``sbx`` extra pins the supported SBX minor series; the lock file records the
exact SBX, Stable-Baselines3, CPU-only Torch and JAX versions used by repository
checks.

Python callbacks and Gymnasium episode randomization deliberately stay on the
host.  ``RewardMode.CUSTOM`` is called after the packaged observation and final
``info`` dictionary exist, just as it is for ``F110Env``.  ``JaxF110Env`` also
draws one shared ``VehicleParameters`` value from the Gym RNG at reset and
builds that episode's traced core parameters from it.  The device-batched API
uses a different boundary: rewards must be pure JAX callables, and vehicle
randomization is sampled from explicit device keys without host conversion.

``reset(seed=...)`` is deterministic within ``JaxF110Env``, including domain
randomization and sensor noise, but it is not a byte-for-byte random-stream
match for ``F110Env``.  The adapter derives explicit JAX keys from Gymnasium's
host RNG, and the functional reset reserves named pose, bias and scan children.
In particular, choosing a pose or state reset override discards the pose child
without shifting the functional sensor children.

Rendering uses the existing PyQt/OpenGL backend and clocks.  The renderer gets
a separate ``DEFAULT`` observation view even when the policy uses
``KINEMATIC_STATE``.  Human and RGB modes therefore have the same display
requirements and timing semantics described in :doc:`rendering`; direct
``RecordVideo`` use needs ``JaxF110Env(cfg, render_mode="rgb_array")`` and
``render_enabled=True``.  Rendering also packages host arrays and is not part
of a device-native throughput claim.

Run a device-native batch
-------------------------

The native batch entry points add an environment axis without changing the
physics core. The shared-map API below uses one ``CoreConfig`` and one map
table; state and episode parameters have an independent leading row for every
environment.  The indexed variant described afterward selects among stacked
equal-shape tables.  Construct a simulator on the requested device, then close
over its tables, topology and parameters in compiled reset and step functions:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   from f1tenth_gym.envs.batching import (
       PolicyField,
       PolicyLayout,
       policy_observation,
       scale_normalized_actions,
   )
   from f1tenth_gym.envs.jax_simulator import JaxSimulator
   from f1tenth_gym.envs.track import Track

   track = Track.from_track_name("Spielberg", 1.0)
   simulator = JaxSimulator(cfg, track, device="cpu")
   batch_size = 64
   layout = PolicyLayout((PolicyField.KINEMATIC_STATE,))

   @jax.jit
   def reset(keys):
       return simulator.reset_batch(keys)

   @jax.jit
   def step(step_keys, reset_keys, state, actions):
       return simulator.step_batch_autoreset(
           step_keys,
           reset_keys,
           state,
           actions,
       )

   root = jax.random.key(42)
   reset_keys = jax.random.split(jax.random.fold_in(root, 0), batch_size)
   observation, state = reset(reset_keys)
   policy_input = policy_observation(observation, simulator.config, layout)
   assert policy_input.shape == (batch_size, cfg.num_agents, 5)

   normalized = jnp.zeros((batch_size, cfg.num_agents, 2), dtype=jnp.float32)
   actions = scale_normalized_actions(normalized, state, simulator.config)
   step_keys = jax.random.split(jax.random.fold_in(root, 1), batch_size)
   next_reset_keys = jax.random.split(jax.random.fold_in(root, 2), batch_size)
   transition = step(step_keys, next_reset_keys, state, actions)
   policy_input = policy_observation(
       transition.next_observation, simulator.config, layout
   )

Use time-major key/action arrays around ``step`` in ``jax.lax.scan`` for a
compiled rollout.  ``step_batch`` is the raw alternative: it reports terminal
status but never resets or freezes a row.  ``step_batch_autoreset`` resets an
entire environment when that row is terminated or truncated; it never resets
individual agents.  Its two observation trees have distinct purposes:

* ``transition_observation`` is the post-step value used for terminal targets
  and timeout bootstrapping;
* ``next_observation`` is the selectively reset value used as the next policy
  input and matches ``transition.state``.

Rewards remain ``(batch, agents)``.  ``select_ego_rewards`` produces one scalar
per environment, while ``flatten_joint_observation`` explicitly turns a
decentralized ``(batch, agents, features)`` policy view into a centralized
``(batch, agents * features)`` view.  A custom device reward is vmapped once per
environment and receives ``(observation, actions, events, metrics, params)``;
it must return one value per agent using JAX operations only.

Policy distributions usually produce normalized actions.  The helper
``batch_action_bounds(state, simulator.config)`` returns the active
``(batch, agents, 2)`` physical limits, including each environment's current
domain-randomization draw.  ``scale_normalized_actions`` maps ``-1`` and ``1``
to those limits and zero to their midpoint.  It deliberately does not clip:
use a bounded transform such as ``tanh`` in the policy so an invalid output is
not hidden.  Target-angle versus steering-rate control selects ``s_min/s_max``
versus ``sv_min/sv_max``; target-speed versus acceleration selects
``v_min/v_max`` versus ``-a_max/a_max``.

``simulator.randomization`` contains the finite active vehicle bounds.  Each reset
key draws one parameter row shared by every agent in that environment, and
different environment rows can draw different physics without recompilation.
The draw updates correlated dynamics and body geometry together.  A named
folded key keeps pose and sensor reset streams unchanged when randomization is
toggled.  The contact table is already sized for the configured full bound
envelope; substituting wider bounds at runtime is unsupported.

``IndexedJaxSimulator(cfg, tracks)`` accepts one resolved ``Track`` per
environment row.  It preprocesses repeated object identities once, groups
unique complete reset/track/pair tables by exact leaf shape, and returns one
``IndexedCoreBucket`` per compilation shape.  Within a bucket,
``reset_indexed_batch`` and ``step_indexed_batch`` select maps with the supplied
``map_indices`` on device.  Across buckets, slice inputs by ``source_indices``,
compile once per bucket, and stitch same-shaped outputs back into source order.
This keeps small maps out of a globally padded worst-map table.  Direct callers
must pass in-range indices: traced entry points validate shape and integer dtype,
while ``IndexedJaxSimulator`` is the host boundary that validates values.

``device`` is authoritative when provided; without it, the simulator
follows active LiDAR/contact device settings and otherwise selects CPU.  The
Phase 6 migration validated a complete native PPO rollout and update, then
measured synchronized state, LiDAR, contact, full and indexed-map workloads on
CPU and GPU.  Those repository-only harnesses were removed after the cleanup
reran them successfully; see :doc:`jax_performance` for the retained protocol,
measurements and placement guidance.

JaxMARL status
--------------

An official-JaxMARL adapter is planned as the multi-agent compatibility
boundary, but cannot yet be published against a compatible released package.
JaxMARL 0.1.0 constrains JAX to 0.4 and SciPy to 1.12, while this project uses
JAX 0.11 or newer and SciPy 1.13 or newer. Its unreleased 0.2 development line
removes those conflicting upper bounds.

Once a released JaxMARL 0.2+ is verified compatible, the adapter should translate
one dense F1TENTH race into its agent-keyed ``reset``/``step`` protocol and let
JaxMARL apply the outer environment ``vmap``. It must preserve per-episode
domain randomization, map timeout into ``done["__all__"]`` while retaining
separate status in ``info``, and require bounded continuous policy actions.
The `official JaxMARL package <https://github.com/bold-lab-ai/JaxMARL>`_ remains
the dependency; copied base classes or spaces will not be vendored here.

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

What remains outside this repository
------------------------------------

The repository ships one focused native PPO example and optional trainer
protocol adapters, not a general algorithm suite. Production PPO/SAC libraries
and experiment management remain external. Pure pursuit, MPC and other
controllers live in the separate ``f1tenth_planning`` repository — the
pure-pursuit follower in ``examples/waypoint_follow.py`` is a demo, not a
supported API. The reusable RL surface here is the native batch API, reward
and observation configuration, and the Gym wrappers (the second,
:class:`~f1tenth_gym.envs.wrappers.ObservationDelayWrapper`, delays only the
observation to model sensing lag — see :doc:`sim2real`), all configured
through the frozen :class:`~f1tenth_gym.envs.env_config.EnvConfig` tree
(:doc:`configuration`).

Next: :doc:`sim2real` corrupts the command, the car and the sensing this
loop trains on; :doc:`reproducibility` explains what one seed pins when you
A/B those settings.
