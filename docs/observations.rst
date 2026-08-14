Observations
============

An observation is a ``dict[agent_id -> dict[field -> ndarray]]`` — one entry
per agent under the keys ``agent_0``, ``agent_1``, … — and the default preset
carries no ``pose_x``. Position arrives packed inside ``std_state``, so reading
``obs["agent_0"]["pose_x"]`` out of a default env raises ``KeyError`` rather
than returning a number:

>>> import gymnasium as gym
>>> import numpy as np
>>> from f1tenth_gym.envs.env_config import EnvConfig
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> obs, info = env.reset(seed=42)
>>> print(" ".join(sorted(obs["agent_0"])))
collision frenet_pose lap_count lap_time scan sim_time state std_state
>>> obs["agent_0"]["std_state"]   # [X, Y, delta, speed, yaw, yaw_rate, beta]
array([-0.0440806, -0.8491629,  0.       ,  0.       , -2.8797681,
        0.       ,  0.       ], dtype=float32)
>>> int(np.count_nonzero(obs["agent_0"]["scan"]))   # reset sweeps the LiDAR
1080
>>> env.close()

Presets and their fields
------------------------

Six members of :class:`~f1tenth_gym.envs.observation.ObservationType` decide
what a step hands back, selected through ``ObservationConfig(type=...)``
(:doc:`configuration`). Everything except ``DIRECT`` resolves to the same
``FullObservation`` provider and differs only in the tuple exposed.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Type
     - What it returns
   * - ``DEFAULT`` *(default)*
     - The eight base fields: ``scan``, ``std_state``, ``state``,
       ``collision``, ``lap_time``, ``lap_count``, ``sim_time``,
       ``frenet_pose``.
   * - ``DIRECT``
     - Agent-batched arrays read straight off the simulator's
       struct-of-arrays buffers, as one flat dict rather than a dict per
       agent. Selecting it emits a ``UserWarning``: before v1.0.0 the name
       meant the packaged per-agent dict that ``DEFAULT`` now provides.
   * - ``FEATURES``
     - Exactly the tuple passed as
       ``ObservationConfig(type=FEATURES, features=(...))``, drawn from the
       base and derived vocabularies below. An unknown name raises
       ``ValueError: Unknown observation feature: 'speed'``, and ``features``
       may only be set alongside this type.
   * - ``KINEMATIC_STATE``
     - ``pose_x``, ``pose_y``, ``delta``, ``linear_vel_x``, ``pose_theta``.
   * - ``DYNAMIC_STATE``
     - ``pose_x``, ``pose_y``, ``delta``, ``linear_vel_magnitude``,
       ``pose_theta``, ``ang_vel_z``, ``beta``.
   * - ``FRENET_DYNAMIC_STATE``
     - The ``DYNAMIC_STATE`` fields with ``linear_vel_x`` and
       ``linear_vel_y`` in place of the magnitude. It carries no Frenet
       field: the name refers to splitting velocity into body-frame
       components. For ``(s, ey, ephi)`` use ``DEFAULT``, ``FEATURES`` or
       ``DIRECT``.

Under ``DIRECT`` the leading axis of every array is the agent index, ``scans``
and ``frenet`` appear only when the LiDAR and the Frenet frame are on, and
``sim_time`` is shared rather than per-agent:

>>> from f1tenth_gym.envs.env_config import ObservationConfig
>>> from f1tenth_gym.envs.observation import ObservationType
>>> cfg = EnvConfig(
...     observation_config=ObservationConfig(type=ObservationType.DIRECT),
...     render_enabled=False,
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> obs, info = env.reset(seed=42)
>>> for key in sorted(obs):
...     print(key, obs[key].shape)
collisions (1,)
frenet (1, 3)
lap_counts (1,)
lap_times (1,)
scans (1, 1080)
sim_time ()
standard_state (1, 7)
state (1, 7)
>>> env.close()

Two fields are conditional under every preset. ``scan`` is dropped when
``lidar_config.enabled=False`` and ``frenet_pose`` is dropped when
``compute_frenet_frame=False``, so the missing key is a loud ``KeyError``
instead of a shape-``(0,)`` array holding no data. Asking for either one
explicitly through ``FEATURES`` raises at ``gym.make`` — ``scan requested but
the LiDAR is disabled (lidar_config.enabled=False)``, or ``frenet_pose
requested but environment does not compute the Frenet frame``.

Base fields
-----------

Eight names read straight from the simulator buffers, all ``float32``:

.. list-table::
   :header-rows: 1
   :widths: 18 14 68

   * - Field
     - Shape
     - Meaning
   * - ``scan``
     - ``(num_beams,)``
     - LiDAR ranges in metres, ``(1080,)`` under the default LiDAR config.
       ``reset`` runs one sweep, so the first observation of an episode
       already carries real ranges and their noise (:doc:`reproducibility`).
   * - ``std_state``
     - ``(7,)``
     - ``[X, Y, steering_angle, speed, yaw, yaw_rate, beta]``, seven wide and
       anchored at the vehicle's CoG under every dynamics model. Index 6 is
       the slip angle ``beta``, not a lateral velocity. Under KS indices 5
       and 6 are hardcoded ``0.0``.
   * - ``state``
     - ``(state_dim,)``
     - The raw model state in the model's own frame — KS
       ``[x, y, delta, v, yaw]`` (5), ST ``[x, y, delta, v, yaw, yaw_rate,
       beta]`` (7). See :doc:`dynamics`.
   * - ``collision``
     - ``()``
     - ``1.0`` while this agent is in contact this step, else ``0.0``.
   * - ``lap_time``
     - ``()``
     - Elapsed time on the current lap, in seconds.
   * - ``lap_count``
     - ``()``
     - Completed laps, monotonically non-decreasing — driving backwards does
       not decrement it.
   * - ``sim_time``
     - ``()``
     - Simulated seconds since ``reset``, one timestep behind
       ``info["sim_time"]``.
   * - ``frenet_pose``
     - ``(3,)``
     - ``[s, ey, ephi]`` against the centerline: arclength in metres, signed
       lateral deviation (``+ey`` is left of travel) and heading error in
       radians. A fresh reset does not give ``ey == 0`` unless the reset
       strategy uses the centerline as its reference line (:doc:`tracks`).

Derived fields
--------------

Nine named scalars computed from ``std_state`` at observation time, with
``vx = speed·cos(beta)`` and ``vy = speed·sin(beta)``:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Field
     - Meaning
   * - ``pose_x`` / ``pose_y``
     - Position X / Y (``std_state[0]`` / ``[1]``), on the same anchor as
       ``std_state`` whatever the model (:doc:`dynamics`).
   * - ``pose_theta``
     - Yaw (``std_state[4]``), wrapped to ``[-pi, pi)``.
   * - ``linear_vel_x`` / ``linear_vel_y``
     - Longitudinal ``vx`` / lateral ``vy`` velocity. ``vy`` is identically
       ``0.0`` under KS.
   * - ``linear_vel_magnitude``
     - ``hypot(vx, vy)``, a true magnitude — it stays non-negative when the
       car reverses, unlike the signed ``std_state[3]``.
   * - ``ang_vel_z``
     - Yaw rate (``std_state[5]``). Identically ``0.0`` under KS.
   * - ``delta``
     - Steering angle (``std_state[2]``).
   * - ``beta``
     - Slip angle (``std_state[6]``). Identically ``0.0`` under KS.

Fifty steps of a 2 m/s command leave ``linear_vel_x`` short of the target — the
speed command is a setpoint for a P controller, not an assignment
(:doc:`actions`):

>>> cfg = EnvConfig(observation_config=ObservationConfig(
...     type=ObservationType.KINEMATIC_STATE), render_enabled=False)
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> obs, info = env.reset(seed=42)
>>> for _ in range(50):
...     obs, reward, terminated, truncated, info = env.step(
...         np.array([[0.0, 2.0]], dtype=np.float32))
>>> obs["agent_0"]["linear_vel_x"]
array(1.8179682, dtype=float32)
>>> env.close()

Dtype and shape contract
------------------------

Every scalar arrives as a 0-d ``float32`` ndarray, never a Python ``float``
(unwrap one with ``float()``), and every vector as a 1-d ``float32`` array.
That is load-bearing rather than incidental: it keeps the numba kernels in
downstream code — the ``np.dot`` in ``examples/waypoint_follow.py``, for one —
from hitting a dtype promotion on the first call. Each field is also a copy,
not a view: the simulator overwrites its struct-of-arrays buffers in place on
every step, so an aliased ``scan`` stored in a replay buffer would rewrite
itself from under the learner. ``DIRECT`` copies on the same rule.

>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> obs, info = env.reset(seed=42)
>>> lap = obs["agent_0"]["lap_time"]
>>> type(lap), lap.shape, lap.dtype
(<class 'numpy.ndarray'>, (), dtype('float32'))
>>> scan, first = obs["agent_0"]["scan"], float(obs["agent_0"]["scan"][0])
>>> np.shares_memory(scan, env.unwrapped.sim.state.scans)
False
>>> _ = env.step(np.array([[0.0, 2.0]], dtype=np.float32))
>>> float(scan[0]) == first     # the stored scan survived the step
True

The observation space
---------------------

``env.observation_space`` is a nested ``gym.spaces.Dict`` — one ``Dict`` per
agent, one ``Box`` per selected field — and every bound is finite, so the space
passes gymnasium's ``check_env`` and can be normalised. Limits come from the
vehicle parameters and the track, with steering and velocity padded by one
integrator step of actuator overshoot (``sv_max·idt`` and ``a_max·idt``),
because the constraints zero an actuator rate only after the limit has already
been crossed:

>>> space = env.observation_space["agent_0"]
>>> print(space["std_state"].low[2], space["std_state"].high[3])  # delta, speed
-0.4509 20.0951
>>> space["scan"]
Box(0.0, 30.0, (1080,), float32)
>>> env.close()

Pose bounds are the centerline bounding box plus a hardcoded 5 m margin
(``pose_x`` spans ``[-81.088, 28.886]`` on Spielberg); ``ang_vel_z`` is the
kinematic estimate ``1.5·spd_hi·tan(steer_hi)/wheelbase`` on those padded
values; ``frenet_pose`` runs ``[0, -20, -pi]`` to ``[s_frame_max, 20, pi]``,
where the ±20 m lateral limit is a fixed constant rather than the track width;
and ``lap_time``, ``lap_count`` and ``sim_time`` cap at ``1e6``. Domain
randomization does not invalidate any of it — both spaces are built from the
widest parameters the configured ranges allow, and ``update_params`` rebuilds
the observation space to match (:doc:`sim2real`).

Field order differs between the observation and its space. ``observe()`` emits
fields in preset order, while ``gym.spaces.Dict`` sorts its keys
lexicographically and ``FlattenObservation`` follows the space:

>>> from gymnasium.wrappers import FlattenObservation
>>> from f1tenth_gym.envs.wrappers import SingleAgentWrapper
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)   # KINEMATIC_STATE
>>> list(env.observation_space["agent_0"])
['delta', 'linear_vel_x', 'pose_theta', 'pose_x', 'pose_y']
>>> flat = FlattenObservation(SingleAgentWrapper(env))
>>> obs, info = flat.reset(seed=42)
>>> np.round(obs, 4)
array([ 0.    ,  0.    , -2.8798, -0.0441, -0.8492], dtype=float32)
>>> flat.close()

.. warning::

   Index a flattened observation by ``sorted(fields)``, never by the preset
   tuple: ``KINEMATIC_STATE`` is declared ``(pose_x, pose_y, delta,
   linear_vel_x, pose_theta)`` but flattens to the sorted order above. Both
   are length-5 float32 vectors, so the wrong one yields plausible numbers in
   the wrong slots. The same sort reorders agents past ten, where
   ``agent_10`` precedes ``agent_2``.

The info dict
-------------

The fifth element of the step tuple carries what is not a per-agent
observation. ``reset`` returns three keys and ``step`` returns five —
``collisions`` and ``progress`` exist only on ``step``, so code that reads them
uniformly raises on the first frame. Its clock also runs one timestep ahead of
the observation's, because ``observe()`` is called before the env refreshes
``self.sim_time``:

>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
>>> obs, info = env.reset(seed=42)
>>> sorted(info)
['lap_counts', 'lap_times', 'sim_time']
>>> obs, reward, terminated, truncated, info = env.step(
...     np.array([[0.0, 2.0]], dtype=np.float32))
>>> sorted(info)
['collisions', 'lap_counts', 'lap_times', 'progress', 'sim_time']
>>> float(obs["agent_0"]["sim_time"]), info["sim_time"]
(0.0, 0.01)
>>> env.close()

``lap_times``, ``lap_counts``, ``collisions`` and ``progress`` are per-agent
arrays of length ``num_agents``, and each is a copy of the env's live buffer —
a stored info dict does not change retroactively. ``progress`` is the forward
Frenet arclength each agent gained this step, in metres, and it is computed on
every step whatever the reward mode (:doc:`rl`).
