Choosing a dynamics model
=========================

.. seealso::

   :mod:`f1tenth_gym.envs.dynamic_models` — ``DynamicModel``,
   ``VehicleParameters``, the three presets;
   :class:`~f1tenth_gym.envs.integrators.IntegratorType`.

Two vehicles are supported. ``KS`` has no tyre slip — steering geometry alone
sets the heading. ``ST`` adds linear cornering stiffness, a yaw-rate state and
a body slip angle. Units throughout are SI: metres, radians, m/s, m/s², and
rad/m for curvature.

Slip is the whole difference
----------------------------

``SimulationConfig.dynamics_model`` selects one of two supported
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
     - CoG
   * - ``ST`` (= 2, default)
     - 7
     - ``[x, y, delta, v, yaw, yaw_rate, beta]``
     - CoG

``DynamicModel.MB`` remains temporarily as a transition marker, but
``EnvConfig`` rejects it as unsupported before constructing a simulator.

Pure JAX kernels
----------------

The functional implementation follows the same responsibility-first layout as
the mutable simulator. Dynamics live in
:mod:`f1tenth_gym.envs.dynamic_models.jax`, controls beside
:mod:`f1tenth_gym.envs.action`, and integration beside
:mod:`f1tenth_gym.envs.integrators`. Together they expose CoG KS and ST
derivatives, traced vehicle and episode parameters, controllers, actuator
noise/FIFO delays, immutable fixed-shape state, Euler/RK4 substeps and
``lax.scan`` free-flight rollouts.
Structural choices live in ``DynamicsConfig`` while values such as physical
parameters and noise scales remain traced in ``DynamicsRuntimeParams``. The
layer also exposes shared body geometry, clean exact LiDAR sensing, wall
contact and simultaneous vehicle-pair contact over fixed tables. Wall response
vmaps the manifold/Jacobi solve across agents. Pair response builds every manifold from
one shared state, computes each Jacobi sweep from common body velocities,
scatter-adds equal-and-opposite impulses/corrections and writes one KS/ST
response per body. The combined ``resolve_contacts`` function preserves the
mutable simulator's macro order—walls first, then pairs—and unions fresh
per-agent events; collisions never latch or freeze a vehicle.

``reset_core`` and ``step_core`` compose those layers into one immutable,
explicit-key environment transition. Reset samples poses, projects Frenet state
and produces a real first scan. ``reset_core_from_poses`` and
``reset_core_from_state`` provide fixed-shape override entry points with the
same episode and sensor initialization. Each step advances actuation/dynamics,
resolves optional contact, projects the corrected pose, observes LiDAR and
updates progress, laps, rewards and end status. ``CoreObservation`` is a fixed
batched vocabulary for later adapters; ``CoreMetrics`` distinguishes
termination from timeout truncation. The core does not auto-reset.

These functions support ``jax.jit``, environment-level ``jax.vmap``,
``lax.scan`` and differentiation away from geometry switches. The standalone
layers remain available for focused use and testing. Host-preprocessed
reference-line, reset and ray tables are described in :doc:`tracks`; pair-table
construction is listed in :doc:`api/index`.

The host-only simulator converts the supported parts of an ``EnvConfig`` and an
already-resolved ``Track`` into device-placed functional state::

   from f1tenth_gym.envs.jax_simulator import JaxSimulator

   simulator = JaxSimulator(config, track)

``simulator.reset(key)`` and ``simulator.step(key, state, actions)`` own the
compiled transition seam, matching the role of ``F110Simulator`` in the
mutable environment. Advanced batched code can still use
``simulator.config``, ``simulator.tables`` and ``simulator.params`` directly;
``simulator.track`` keeps the host track paired with those device arrays for
adapters.
``simulator.randomization`` contains
the finite active vehicle bounds used by device-native resets. This is
construction for the functional core, not a Gymnasium environment: it does not
render or turn ``EnvConfig.seed`` into JAX keys. An explicit ``device``
can place even a sensing/contact-disabled simulator on an accelerator; otherwise
the active subsystem device settings select the target and CPU is the fallback.
The deep-import ``GymObservationAdapter``
packages every current observation preset and ``DIRECT`` layout with finite
Gymnasium spaces, selective device transfer and independent float32 copies, but
does not own an environment lifecycle. Construct it with
``GymObservationAdapter.from_simulator(simulator)``; an optional
``ObservationConfig`` selects another view without replacing the simulator's
source configuration, so sensor values and declared bounds cannot diverge.
Construction still rejects ``CUSTOM`` rewards unless a host adapter supplies an
explicit built-in fallback whose result it discards. Varying domain-
randomization bounds remain traced metadata rather than requiring a host draw;
an optional explicit ``VehicleParameters`` still selects one checked host-
managed episode. Active contact and LiDAR must select the same JAX device unless
``device`` explicitly overrides both, and functional contact currently
requires the default wall-extraction tolerance.

The host-only Gymnasium boundary is a separate deep import::

   from f1tenth_gym.envs.jax_env import JaxF110Env

   env = JaxF110Env(config)

It resolves the track, builds and jits the functional reset/step, validates the
native ``(num_agents, 2)`` action, selects the ego reward, packages copied NumPy
observations and ``info``, and preserves reset pose/state overrides.  It also
implements host ``CUSTOM`` callbacks, per-reset shared domain-randomization
draws, reconfiguration and the existing renderer.  Those facilities are
adapter work: arbitrary Python and Gym/render objects never enter the pure
transition.  The class is not registered and does not replace
``f1tenth_gym:f1tenth-v0``.  Its Gym-compatible host transfer is useful for
SB3/SBX interoperability, but it is separate from the device-batched training
API and PPO example and carries no throughput claim; see :doc:`rl`.

``JaxF110Env.reset(seed=...)`` controls a host Gymnasium RNG from which the
adapter derives explicit JAX keys.  Replay is deterministic within that class,
but NumPy and JAX random streams are intentionally not byte-paired with
``F110Env``.  Functional pose/state overrides reserve and discard the same pose
key child instead of shifting the bias and scan children.

The responsibility-first :mod:`f1tenth_gym.envs.batching` module exposes
``reset_batch``, ``step_batch`` and ``step_batch_autoreset`` for shared-map
device-native rollouts. Batched state
and active parameters keep one leading environment axis; physical parameters
remain scalar per environment and are shared across that environment's agents.
``PolicyLayout`` builds ordered decentralized arrays without NumPy conversion,
and joint flattening is explicit. Selective auto-reset preserves the terminal
transition observation separately from the reset observation used as the next
carry. See :doc:`rl` for the complete contract and example.

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
KS  x= 28.738  y= -1.773  beta= 0.0000  yaw_rate= 0.000
ST  x= 28.110  y=  2.527  beta=-0.3005  yaw_rate= 2.999

Same seed, same spawn, same 300 actions: the two cars finish **4.35 m** apart,
and ``ST`` is carrying 0.30 rad of slip that ``KS`` reports as an exact zero.
Pick ``KS`` when the controller under test only needs a plausible kinematically
feasible path, and ``ST`` whenever grip is part of the problem.

One asymmetry is worth knowing before trusting a reverse manoeuvre: that
low-speed switch is ``if V < 0.5``, not ``if abs(V) < 0.5``, so reversing runs
the kinematic sub-model at any speed. At ``v = -3`` m/s the ``ST`` yaw-rate
derivative reproduces the kinematic ``V·cos(beta)·tan(delta)/wheelbase`` to
eight significant figures, and the yaw acceleration and the slip-angle rate are
both exactly zero.

Multi-body transition
---------------------

Only KS and ST are exposed through the environment's physics/contact surface.
The transitional enum is rejected regardless of parameter preset, before
loading a map or compiling a kernel:

>>> unsupported = SimulationConfig(dynamics_model=DynamicModel.MB)
>>> EnvConfig(simulation_config=unsupported)  # doctest: +ELLIPSIS
Traceback (most recent call last):
    ...
ValueError: DynamicModel.MB is unsupported...

The full-scale parameter measurements remain available as data, but do not
make MB selectable. Use ``ST`` for dynamic tyre behavior or ``KS`` for the
cheaper kinematic approximation.

Where the pose is measured
--------------------------

The supported models use one centre-of-gravity frame. Raw ``state``,
``state.poses``, ``std_state``, the derived ``pose_x``/``pose_y``,
``frenet_pose`` and the renderer therefore describe the same physical point.
``options={"poses": ...}`` is CoG-referenced too:

>>> kinematic = EnvConfig(render_enabled=False, simulation_config=SimulationConfig(
...     dynamics_model=DynamicModel.KS, max_laps=None))
>>> env = gym.make("f1tenth_gym:f1tenth-v0", config=kinematic)
>>> obs, _ = env.reset(options={"poses": np.array([[0.0, 0.0, 0.0]])})
>>> obs["agent_0"]["state"][:2]
array([0., 0.], dtype=float32)
>>> obs["agent_0"]["std_state"][:2]
array([0., 0.], dtype=float32)
>>> env.close()

The older rear-axle CommonRoad KS equation remains only as a numerical test
oracle for the explicit rear-axle-to-CoG transform. It is not selectable by an
environment configuration.

Normalising the pose is half of what ``std_state`` does; the other half is
normalising the width. It is ``(N, 7)`` for every model —
``[x, y, steering_angle, speed, yaw, yaw_rate, beta]`` — so downstream code
reads the same columns whether the raw state has 5 or 7 entries. Column 6 is
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
