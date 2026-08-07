Actions
=======

Every :meth:`~f1tenth_gym.envs.f110_env.F110Env.step` consumes one
``(num_agents, 2)`` float32 array — steering in column 0, the longitudinal
command in column 1 — and the simulator checks only its shape. Bounds are
advisory: an action outside ``env.action_space`` is executed rather than
rejected or clipped, and an oversized command is limited only by the actuator
constraints inside the dynamics (:doc:`dynamics`). A 500 m/s speed command is
a legal input that accelerates the car at ``a_max``:

>>> import gymnasium as gym
>>> import numpy as np
>>> from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
>>> cfg = EnvConfig(
...     simulation_config=SimulationConfig(max_laps=None), render_enabled=False
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> env.action_space
Box([[-0.4189 -5.    ]], [[ 0.4189 20.    ]], (1, 2), float32)
>>> obs, info = env.reset(seed=42)
>>> action = np.array([[0.0, 2.0]], dtype=np.float32)   # [[steering, speed]]
>>> for _ in range(100):
...     obs, reward, terminated, truncated, info = env.step(action)
>>> print(f"{obs['agent_0']['std_state'][3]:.4f}")      # speed tracks the target
1.9841
>>> big = np.array([[0.0, 500.0]], dtype=np.float32)
>>> env.action_space.contains(big)
False
>>> obs, *_ = env.step(big)                             # executed anyway
>>> print(f"{obs['agent_0']['std_state'][3]:.4f}")      # one step of a_max
2.0792
>>> env.close()

Column meanings by mode
-----------------------

Two fields on :class:`~f1tenth_gym.envs.env_config.ControlConfig` select how
the simulator interprets each column: ``steering_mode`` for column 0 and
``longitudinal_mode`` for column 1 (:doc:`configuration` covers the nested
``with_updates`` pattern for setting them). The four combinations, with the
``Box`` bounds each one induces for the default ``F1TENTH_VEHICLE_PARAMETERS``:

.. list-table::
   :header-rows: 1
   :widths: 10 28 30 32

   * - Column
     - Mode
     - Command
     - Bounds (per row)
   * - 0
     - ``STEERING_ANGLE`` *(default)*
     - target steering angle, rad
     - ``[s_min, s_max]`` = ``[-0.4189, 0.4189]``
   * - 0
     - ``STEERING_SPEED``
     - steering velocity, rad/s
     - ``[sv_min, sv_max]`` = ``[-3.2, 3.2]``
   * - 1
     - ``SPEED`` *(default)*
     - target speed, m/s
     - ``[v_min, v_max]`` = ``[-5.0, 20.0]``
   * - 1
     - ``ACCL``
     - longitudinal acceleration, m/s²
     - ``[-a_max, a_max]`` = ``[-9.51, 9.51]``

Every bound comes from ``EnvConfig.params``, so swapping the vehicle preset
moves them — read them off ``env.action_space`` instead of hardcoding, and use
``env.action_space.sample()`` for a random command of the right shape and
dtype under any mode pair:

>>> from f1tenth_gym.envs.env_config import ControlConfig
>>> from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
>>> cfg = EnvConfig(
...     control_config=ControlConfig(
...         longitudinal_mode=LongitudinalActionType.ACCL,
...         steering_mode=SteerActionType.STEERING_SPEED,
...     ),
...     render_enabled=False,
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> env.action_space
Box([[-3.2  -9.51]], [[3.2  9.51]], (1, 2), float32)
>>> env.close()

.. warning::

   A transposed pair — ``[speed, steer]`` instead of ``[steer, speed]`` — is
   usually still inside the ``Box``: both columns are float32 with overlapping
   valid ranges, and only the shape is checked, so the swapped pair is
   indistinguishable from a deliberate command. The simulator executes it
   faithfully and the trajectory is simply wrong. Check the column order first
   when results look inexplicable.

What each mode does to the command
----------------------------------

Under the default ``SPEED`` mode the commanded speed is turned into an
acceleration by :func:`~f1tenth_gym.envs.dynamic_models.pid_accl`, a
proportional controller with four gain quadrants (forward vs. reverse,
accelerating vs. braking) — there is no I or D term. The quadrant test is
``current_speed > 0.0``, so a car at exactly rest gets the weaker reverse
gains, and every ``reset()`` spawns at ``v = 0`` — the first step of every
episode launches on that branch:

>>> from f1tenth_gym.envs.dynamic_models import pid_accl
>>> from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as p
>>> pid_accl(5.0, 0.0, p.a_max, p.v_max, p.v_min)    # exactly at rest
4.755
>>> pid_accl(5.0, 0.001, p.a_max, p.v_max, p.v_min)  # barely rolling: 5x jump
23.770245

The returned effort may exceed ``a_max``: it is a raw command that
``accl_constraints`` inside the dynamics then limits (:doc:`dynamics`).

Under ``STEERING_ANGLE`` the target angle is realised by
:func:`~f1tenth_gym.envs.dynamic_models.pid_steer`, a saturated
proportional controller: ``sv = clip(kp * error, -sv_max, sv_max)``. The gain
comes from ``ControlConfig.steer_kp`` — ``None`` (the default) derives
``10 * sv_max / (s_max - s_min)`` from the vehicle limits, and any value
``<= 0`` selects the legacy bang-bang relay, which slams the full ``±sv_max``
at any error above ``1e-4`` rad and therefore limit-cycles around the target
by about ``sv_max * timestep`` (0.032 rad at the defaults) instead of
settling.

Under ``ACCL`` and ``STEERING_SPEED`` the command is applied as-is:
:func:`~f1tenth_gym.envs.action.accl_action` and
:func:`~f1tenth_gym.envs.action.steering_speed_action` are identities.

Multi-agent action arrays
-------------------------

With ``num_agents=2`` the array gains a second row; row ``i`` commands agent
``i``:

.. code-block:: python

   action = np.array(
       [
           [0.0, 2.0],   # agent_0: straight at 2 m/s
           [0.2, 2.0],   # agent_1: steer left at 2 m/s
       ],
       dtype=np.float32,
   )

The per-agent ``Box`` is repeated once per row, so the bounds are identical
across agents:

>>> cfg = EnvConfig(num_agents=2, render_enabled=False)
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> env.action_space.shape
(2, 2)
>>> env.action_space.low
array([[-0.4189, -5.    ],
       [-0.4189, -5.    ]], dtype=float32)
>>> env.close()

:class:`~f1tenth_gym.envs.wrappers.SingleAgentWrapper` removes the leading
axis for single-agent training — a flat ``(2,)`` action is reshaped to
``(1, 2)`` — and it does not clip; see :doc:`rl`.
