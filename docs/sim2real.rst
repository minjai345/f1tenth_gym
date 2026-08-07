Sim2real
========

A policy trained against perfect actuators, a perfect LiDAR and one exact set
of vehicle parameters learns to exploit all three. The simulator can corrupt
each of them, and the useful way to hold the knobs in your head is by two
questions: **where does the imperfection enter the control loop**, and **how
often is it redrawn**.

.. list-table::
   :header-rows: 1
   :widths: 26 30 22 22

   * - Knob
     - Enters at
     - Redrawn
     - Config
   * - ``steer_noise_std``, ``accl_noise_std``
     - the command, before the delay buffers
     - every step
     - ``ControlConfig``
   * - ``steer_delay_steps``, ``throttle_delay_steps``
     - the command, after the noise
     - fixed lag
     - ``ControlConfig``
   * - ``noise_std``, ``dropout_prob``
     - the observed scan, after ray casting
     - every step
     - ``LiDARConfig``
   * - ``range_bias_std``
     - the observed scan, after ray casting
     - once per episode
     - ``LiDARConfig``
   * - ``param_ranges``
     - the vehicle itself, before the physics
     - once per episode
     - ``DomainRandomizationConfig``

Everything defaults to zero or disabled, so a config that sets none of these
runs the same deterministic simulation it always did.

Corrupting the command
----------------------

:class:`~f1tenth_gym.envs.env_config.ControlConfig` degrades the command on its
way from your action array into the dynamics. Gaussian noise is added first,
then the ring-buffer lag, so a delayed command carries the noise it was issued
with rather than picking up fresh noise on arrival:

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig, ControlConfig

   cfg = EnvConfig(
       control_config=ControlConfig(
           steer_noise_std=0.02,     # rad, per step
           accl_noise_std=0.1,       # command units, per step
           steer_delay_steps=2,      # steps of servo lag
           throttle_delay_steps=2,   # steps of drivetrain/ESC lag
       ),
   )

The noise comes from a dedicated control RNG reseeded off the reset seed, so
two runs at the same seed corrupt the commands identically. Fifty steps of a
constant ``[0.1, 3.0]`` command from the same spawn diverge by only 3 mm at
these settings — the drift is a slow accumulation, not a per-step jolt, which
is exactly why a policy that never sees it can still fail on hardware.

Corrupting the sensing
----------------------

:class:`~f1tenth_gym.envs.lidar.LiDARConfig` carries three sensor knobs that
differ mainly in cadence. ``noise_std`` and ``dropout_prob`` are redrawn every
step; ``range_bias_std`` is drawn once at ``reset()`` and held for the whole
episode, which is what makes it a model of calibration error rather than of
sensor jitter.

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig
   from f1tenth_gym.envs.lidar import LiDARConfig

   cfg = EnvConfig(
       lidar_config=LiDARConfig(
           noise_std=0.01,        # m, fresh every step
           dropout_prob=0.02,     # 2% of beams return no-range each step
           range_bias_std=0.03,   # m, per-beam, fixed for the episode
       ),
   )

The cadence is observable. Park the car and step it with the bias alone
enabled: successive scans are bit-identical, because the only perturbation was
drawn at reset. Swap the bias for the same amount of ``noise_std`` and
successive scans differ by up to a quarter of a metre. Dropped beams are not
noisy values but exactly ``range_max`` — at ``dropout_prob=0.05`` about 64 of
1080 beams read 30.0 m on a given step.

.. warning::

   Every LiDAR knob perturbs the **observed** scan only. Collision detection
   runs on the clean, noise-free scan, so no amount of sensor noise changes
   when a crash fires — a policy cannot dodge a wall by hallucinating range.

Randomizing the vehicle
-----------------------

:class:`~f1tenth_gym.envs.env_config.DomainRandomizationConfig` redraws vehicle
parameters once per episode, at ``reset()``. Ranges are **absolute physical
units**, not multipliers, and arrive as a typed
:class:`~f1tenth_gym.envs.dynamic_models.VehicleParamRanges` — one optional
slot per ``VehicleParameters`` field, so a typo is a ``TypeError`` at
construction rather than a silently-ignored key:

.. code-block:: python

   from f1tenth_gym.envs.env_config import EnvConfig, DomainRandomizationConfig
   from f1tenth_gym.envs.dynamic_models import VehicleParamRanges

   cfg = EnvConfig(
       domain_randomization_config=DomainRandomizationConfig(
           enabled=True,
           param_ranges=VehicleParamRanges(
               m=(3.0, 4.0),      # mass, kg
               mu=(0.9, 1.1),     # tyre friction coefficient
               lf=(0.14, 0.18),   # CoG-to-front-axle distance, m
           ),
       ),
   )

Two consecutive resets of that config draw masses of 3.144 kg and 3.814 kg
around the 3.74 kg nominal. The draws use the env RNG, so ``reset(seed=...)``
reproduces them. Each episode re-derives from the configured base rather than
from the previous episode's draw, so there is no compounding drift across a
long training run.

The randomized values reach the physics but not the config: ``env.vehicle_params``
and ``env.env_config.params`` keep reporting the nominal 3.74 kg while the
kernels integrate 3.144. **The ground truth for what the car currently is is
``sim.params_array``.**

Randomizing actuation limits — ``v_max``, ``s_max``, ``sv_max``, ``a_max`` and
their minima — is supported. Gymnasium requires the action and observation
spaces to be fixed for an env's lifetime, so they are built from the *widest*
parameters across the declared ranges: a fixed superset that every episode
stays inside, rather than a point estimate that randomized episodes escape.

Names are the real ``VehicleParameters`` fields — ``m`` for mass, ``h`` for CoG
height, not ``mass``/``h_cg``. :doc:`configuration` lists them, and
:doc:`dynamics` explains which ones the kernels actually read: under the
kinematic and single-track models only the first 18 reach the njit boundary, so
randomizing a multi-body-only parameter is a silent no-op.

What one seed pins
------------------

Every knob on this page draws from a stream seeded by ``reset(seed=...)``, so a
seeded episode replays exactly. What is easy to get wrong is comparing two
*different* configs: enabling domain randomization inserts extra draws into the
shared stream, which shifts the LiDAR noise sequence downstream of it even when
the randomized range is degenerate. Two configs are not noise-paired just
because they share a seed — :doc:`reproducibility` covers what that means for
A/B tests.
