Choosing a dynamics model
=========================

.. seealso::

   :mod:`f1tenth_gym.envs.dynamic_models` — ``DynamicModel``,
   ``VehicleParameters``, the three presets;
   :class:`~f1tenth_gym.envs.integrators.IntegratorType`.

Three integrable vehicles ship with the simulator, and the choice between them
is a choice about tyres. ``KS`` has none — steering geometry alone sets the
heading. ``ST`` adds linear cornering stiffness, a yaw-rate state and a body
slip angle. ``MB`` adds a sprung chassis, suspension travel and Pacejka tyres,
at a price in both parameters and world scale. Units throughout are SI: metres,
radians, m/s, m/s², and rad/m for curvature.

Slip is the whole difference
----------------------------

``SimulationConfig.dynamics_model`` selects one of three
:class:`~f1tenth_gym.envs.dynamic_models.DynamicModel` members, and the choice
fixes both the derivative function and the width of the raw state vector:

.. list-table::
   :header-rows: 1
   :widths: 18 12 46 24

   * - Member
     - state_dim
     - Raw state layout
     - Native pose
   * - ``KS`` (= 1)
     - 5
     - ``[x, y, delta, v, yaw]``
     - rear axle
   * - ``ST`` (= 2, default)
     - 7
     - ``[x, y, delta, v, yaw, yaw_rate, beta]``
     - CoG
   * - ``MB`` (= 3)
     - 29
     - multi-body chassis + wheel states
     - CoG

``delta`` is the steering angle, ``v`` the longitudinal speed, ``yaw`` the
heading, and ``beta`` the body slip angle. Under ``KS`` there is nothing for a
tyre to slip against: the derived observation fields ``ang_vel_z``, ``beta``
and ``linear_vel_y`` are hardcoded to ``0.0``, and the car's heading is an
exact function of steering angle and travelled distance. ``ST`` integrates
lateral tyre forces from ``C_Sf``/``C_Sr`` above 0.5 m/s and falls back to the
kinematic expressions below it, so the two agree while parking and separate as
soon as there is cornering load.

Three seconds of one held command on a 30 m circle is enough to see it — a
synthetic reference line keeps the run clear of any wall:

>>> import gymnasium as gym
>>> import numpy as np
>>> from f1tenth_gym.envs.dynamic_models import (
...     DynamicModel, F1TENTH_VEHICLE_PARAMETERS)
>>> from f1tenth_gym.envs.env_config import (
...     EnvConfig, ObservationConfig, SimulationConfig)
>>> from f1tenth_gym.envs.integrators import IntegratorType
>>> from f1tenth_gym.envs.observation import ObservationType
>>> from f1tenth_gym.envs.track import Track
>>> theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
>>> circuit = Track.from_refline(
...     x=30.0 * np.cos(theta), y=30.0 * np.sin(theta), velx=np.full(400, 7.0))
>>> def drive(model, params=F1TENTH_VEHICLE_PARAMETERS,
...           integrator=IntegratorType.RK4):
...     cfg = EnvConfig(
...         map_name=circuit, params=params, render_enabled=False,
...         simulation_config=SimulationConfig(
...             dynamics_model=model, integrator=integrator, max_laps=None),
...         observation_config=ObservationConfig(
...             type=ObservationType.DYNAMIC_STATE),
...     )
...     env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
...     obs, _ = env.reset(seed=42)
...     for _ in range(300):
...         obs, *_ = env.step(np.array([[0.2, 7.0]], dtype=np.float32))
...     env.close()
...     return obs["agent_0"]
>>> for model in (DynamicModel.KS, DynamicModel.ST):
...     a = drive(model)
...     print(f"{model.name}  x={float(a['pose_x']):7.3f}"
...           f"  y={float(a['pose_y']):7.3f}"
...           f"  beta={float(a['beta']):7.4f}"
...           f"  yaw_rate={float(a['ang_vel_z']):6.3f}")
KS  x= 28.834  y= -1.576  beta= 0.0000  yaw_rate= 0.000
ST  x= 28.110  y=  2.527  beta=-0.3005  yaw_rate= 2.999

Same seed, same spawn, same 300 actions: the two cars finish **4.17 m** apart,
and ``ST`` is carrying 0.30 rad of slip that ``KS`` reports as an exact zero.
Pick ``KS`` when the controller under test only needs a plausible kinematically
feasible path, and ``ST`` whenever grip is part of the problem.

One asymmetry is worth knowing before trusting a reverse manoeuvre: that
low-speed switch is ``if V < 0.5``, not ``if abs(V) < 0.5``, so reversing runs
the kinematic sub-model at any speed. At ``v = -3`` m/s the ``ST`` yaw-rate
derivative reproduces the kinematic ``V·cos(beta)·tan(delta)/wheelbase`` to
eight significant figures, and the yaw acceleration and the slip-angle rate are
both exactly zero. ``MB`` uses the symmetric ``abs(v) < 0.5``.

Running MB at full scale
------------------------

The multi-body chassis reads slots 20-87 on top of the 18 that ``KS`` and ``ST``
read, and only ``FULLSCALE_VEHICLE_PARAMETERS`` populates those extra 68. The 1/10- and
1/5-scale presets leave all of them at ``nan``, which used to yield a NaN
trajectory rather than an error; ``EnvConfig`` now rejects the combination at
construction, before the map download and the JIT:

>>> from f1tenth_gym.envs.dynamic_models import FULLSCALE_VEHICLE_PARAMETERS
>>> len(F1TENTH_VEHICLE_PARAMETERS.missing_mb_parameters())
68
>>> len(FULLSCALE_VEHICLE_PARAMETERS.missing_mb_parameters())
0

Supplying the parameters is only half of it. That preset describes a 4.298 m by
1.674 m car, and every shipped map is a 1/10-scale racetrack, so the vehicle
has to be given a world it fits in — ``map_scale=10.0`` stretches the map to
match. Forty steps then run finite and collision-free:

>>> cfg = EnvConfig(
...     map_name="Spielberg", map_scale=10.0,
...     params=FULLSCALE_VEHICLE_PARAMETERS, render_enabled=False,
...     simulation_config=SimulationConfig(
...         dynamics_model=DynamicModel.MB, max_laps=None),
... )
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> obs, info = env.reset(seed=42)
>>> for _ in range(40):
...     obs, _, terminated, _, info = env.step(
...         np.array([[0.05, 8.0]], dtype=np.float32))
>>> terminated, float(info["collisions"][0])
(False, 0.0)
>>> env.close()

.. warning::

   Leaving ``map_scale`` at its ``1.0`` default puts a 4.3 m car on a track
   sized for a 0.58 m one. It collides against a wall at the spawn pose and the
   episode terminates on step 1 — the same rollout as above at ``map_scale=1.0``
   returns ``(True, 1.0)``. Scale the world to the vehicle, or scale the vehicle
   to the world.

``ST`` runs unchanged at ``map_scale=10.0`` with the same full-size parameters,
which makes it the cheaper control when something in an ``MB`` rollout looks
wrong.

Where the pose is measured
--------------------------

Every observation is anchored at the centre of gravity, whatever the model
integrates. ``std_state``, the derived ``pose_x``/``pose_y``, ``frenet_pose``
and the renderer all report the CoG, so a logged trajectory means the same
physical point under ``KS`` as under ``ST``. The raw ``state`` field and
``state.poses`` keep the model's native frame — the rear axle for ``KS`` — and
``options={"poses": ...}`` is interpreted natively too, which is where the
frames become visible. Hand ``KS`` the origin and its rear axle goes there,
putting its CoG ``lr`` ahead:

>>> from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as P
>>> P.lr, P.lf + P.lr                         # rear overhang, wheelbase
(0.17145, 0.3302)
>>> kinematic = EnvConfig(render_enabled=False, simulation_config=SimulationConfig(
...     dynamics_model=DynamicModel.KS, max_laps=None))
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=kinematic)
>>> obs, _ = env.reset(options={"poses": np.array([[0.0, 0.0, 0.0]])})
>>> obs["agent_0"]["state"][:2]               # native frame: the rear axle
array([0., 0.], dtype=float32)
>>> obs["agent_0"]["std_state"][:2]           # CoG, lr along the heading
array([0.17145, 0.     ], dtype=float32)
>>> env.close()

Under ``ST`` both arrays read ``[0., 0.]``, because its native frame already is
the CoG. The frame a model uses is declared by its ``pose_reference``
property, not inferred from ``model != DynamicModel.KS``, so a future
rear-axle model inherits the conversion instead of being silently displaced by
``lr``.

Normalising the pose is half of what ``std_state`` does; the other half is
normalising the width. It is ``(N, 7)`` for every model —
``[x, y, steering_angle, speed, yaw, yaw_rate, beta]`` — so downstream code
reads the same columns whether the raw state has 5 entries or 29. Column 6 is
the slip angle, not a lateral velocity: ``vy`` is ``speed * sin(beta)``, which
is how :doc:`observations` derives ``linear_vel_y``.

What the model reads
--------------------

A car is one frozen
:class:`~f1tenth_gym.envs.dynamic_models.VehicleParameters` instance on
``EnvConfig.params``. Three presets ship — ``F1TENTH_VEHICLE_PARAMETERS`` (the
default), ``F1FIFTH_VEHICLE_PARAMETERS`` for a 1/5-scale car, and
``FULLSCALE_VEHICLE_PARAMETERS``, the only one that fills the multi-body block.
The 18 fields ``KS`` and ``ST`` actually read, grouped by what they change:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Group
     - Fields at their F1TENTH values
   * - Mass and inertia
     - ``m`` 3.74 kg, ``I`` 0.04712 kg·m² about the yaw axis
   * - Geometry
     - ``lf`` 0.15875 m, ``lr`` 0.17145 m (CoG to front / rear axle), ``h``
       0.074 m (CoG height), ``width`` 0.31 m and ``length`` 0.58 m (the
       collision body)
   * - Tyres
     - ``mu`` 1.0489 (friction coefficient), ``C_Sf`` 4.718 and ``C_Sr``
       5.4562 (cornering stiffness, read by ``ST`` only)
   * - Steering limits
     - ``s_min``/``s_max`` ∓0.4189 rad, ``sv_min``/``sv_max`` ∓3.2 rad/s
   * - Longitudinal limits
     - ``v_min`` -5.0 m/s, ``v_max`` 20.0 m/s, ``a_max`` 9.51 m/s²,
       ``v_switch`` 7.319 m/s (above it ``a_max`` is derated)

The wheelbase is ``lf + lr``, 0.3302 m for F1TENTH. Field names are terse and
**not** what you might guess: mass is ``m`` (not ``mass``), CoG height is ``h``
(not ``h_cg``), yaw inertia is ``I``. The steering and speed limits are
``s_min``/``s_max`` and ``v_min``/``v_max``, and those two rows also set the
action- and observation-space bounds (:doc:`actions`, :doc:`observations`).

The dataclass is frozen, so ``with_updates()`` returns a new instance rather
than mutating in place. A heavier car on a lower-grip surface, driven through
the same corner as before:

>>> heavy = F1TENTH_VEHICLE_PARAMETERS.with_updates(m=4.0, mu=0.9)
>>> a = drive(DynamicModel.ST, heavy)
>>> print(f"x={float(a['pose_x']):.3f}  y={float(a['pose_y']):.3f}"
...       f"  beta={float(a['beta']):.4f}")
x=28.508  y=1.960  beta=-0.3456

That is 0.693 m from the nominal ``ST`` run and 0.045 rad more slip — 7% more
mass and 14% less grip, worth more than a car length in three seconds.
Swapping parameters on a live environment goes through
``env.unwrapped.configure(new_cfg)`` (:doc:`configuration`); redrawing them
once per episode goes through ``DomainRandomizationConfig`` (:doc:`sim2real`).

Why the parameter list is append-only
-------------------------------------

Parameters reach the numba kernels as a flat ``float32`` array that is indexed
*positionally*, so the layout is a wire format — and that layout is simply the
``VehicleParameters`` declaration order, exported as
``f1tenth_gym.envs.dynamic_models.PARAMETER_ORDER``. There is one array for
every model, because the parameters describe the *vehicle*, not the model
chosen to simulate it: ``to_array()`` emits all 88 fields and each model reads
the slots it needs. ``KS`` and ``ST`` read indices 0-17 and ignore the rest;
``MB`` additionally reads 20-87.

**Appending a field is safe. Inserting or reordering one is not** — it shifts
every slot after it and silently rewires each kernel, which is exactly how the
multi-body model was broken once, when ``collision_body_center_x``/``_y``
landed at positions 18/19 and moved the whole multi-body block by two.
``tests/test_vehicle_params_abi.py`` pins the full order against a literal list
and fails naming the field that moved.

The flat array is what makes the derivatives jittable at all: ``mu`` is
``params[0]``, ``v_max`` is ``params[15]``, ``width`` and ``length`` are 16 and
17, and no dict or dataclass can cross that boundary. ``to_array`` marshals by
name, so the price is paid once at construction and again in ``update_params``
— never per step. Indices 18/19 ride along in the array but no kernel reads
them: ``collision_body_center_x``/``_y`` are consumed by the simulator's
Python-level geometry helpers.

How far one step actually integrates
------------------------------------

``SimulationConfig`` carries two clocks. ``timestep`` (default ``0.01`` s) is
what one :meth:`~f1tenth_gym.envs.f110_env.F110Env.step` advances;
``integrator_timestep`` (also ``0.01`` s) is what one integrator call advances,
and their ratio is the number of substeps taken per step — see
:doc:`configuration` for how the pair is validated. ``integrator`` picks the
scheme: ``IntegratorType.RK4`` (= 2, the default) evaluates the derivative four
times per substep, ``IntegratorType.EULER`` (= 1) once.

>>> a = drive(DynamicModel.ST, integrator=IntegratorType.EULER)
>>> print(f"x={float(a['pose_x']):.3f}  y={float(a['pose_y']):.3f}")
x=28.149  y=2.517

Euler lands 0.040 m from the RK4 run after three seconds of hard cornering —
about a hundredth of the gap between ``KS`` and ``ST``, which is the useful
scale for the comparison. It is cheap: measured single-agent with the LiDAR
disabled, a step costs 0.049 ms under Euler against 0.060 ms under RK4, and at
the default 1080-beam LiDAR the whole step is ~0.22 ms and the difference is
inside run-to-run noise. Order is rarely the bottleneck here; beams are.

Two details blunt what RK4 earns. The actuator constraints
(``steering_constraint``, ``accl_constraints``) are applied *inside* the
derivative, so they re-clamp at every RK4 stage against intermediate states
rather than once per step. And the njit kernels return float64 even for
all-float32 inputs, while the state buffer is float32, so the integrated result
is re-cast at the step boundary. Neither is visible at ``timestep=0.01``; both
matter if you raise the step and lean on the integrator to absorb it.
