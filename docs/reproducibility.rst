Reproducibility
===============

.. seealso::

   :class:`~f1tenth_gym.envs.f110_env.F110Env` — ``reset``, ``step``;
   :class:`~f1tenth_gym.envs.env_config.EnvConfig` — ``seed``.

One call fixes an entire episode: ``reset(seed=...)`` seeds the env's
``np_random`` generator, and the spawn pose, the LiDAR noise, the actuator
noise and the domain-randomization draw all descend from it. Replaying a
run is a matter of holding four inputs fixed.

Pin these four things
---------------------

* the seed — an explicit ``seed`` on every ``reset`` you want to replay,
  or ``EnvConfig.seed`` set once for the whole run (next section);
* the actions — a fixed sequence, or a seeded policy;
* the configuration — ``env.unwrapped.configure(new_cfg)`` changes the
  physics and the observation surface, and even a switch that changes no
  physics moves the noise streams (last section);
* the software — trajectories are stored at float32 and replay bit-for-bit
  only on the same simulator version and NumPy build.

Nothing needs seeding separately — domain randomization and every sim2real
noise knob fold into the reset seed (:doc:`sim2real`). The same seed and
the same actions replay exactly; changing the seed alone moves the car:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig

   env = gym.make(
       "f1tenth_gym:f1tenth-v0",
       config=EnvConfig(map_name="Spielberg", render_enabled=False),
   )

   def rollout(seed):
       obs, _ = env.reset(seed=seed)
       xs = []
       for t in range(40):
           steer = 0.1 * np.sin(t * 0.05)
           obs, *_ = env.step(np.array([[steer, 3.0]], dtype=np.float32))
           xs.append(float(obs["agent_0"]["std_state"][0]))
       return np.array(xs)

   a = rollout(seed=42)
   b = rollout(seed=42)
   c = rollout(seed=7)
   assert np.array_equal(a, b), "seeded rollouts diverged"
   print(f"seed 7 diverges by {abs(a[-1] - c[-1]):.4f} m")
   env.close()

The assert passes and the print reads ``seed 7 diverges by 0.7726 m``. The
actions and the configuration are identical across all three rollouts, so
the 0.77 m is the seed's doing alone — it selected a different spawn
waypoint.

Why one seed is enough
----------------------

Every step advances all agents together through one struct-of-arrays
simulator: no per-agent threads, no asynchronous work, and nothing in the
physics reads the wall clock — only the renderer paces against it
(:doc:`rendering`). The state transition is a pure function of

* the current simulation state,
* the action you pass to :func:`step`, and
* the random draws seeded at ``reset`` time (LiDAR noise, actuator
  noise, domain randomization, and the spawn pose).

Every one of those draws descends from the single generator that
``reset(seed=...)`` seeds:

* **Spawn pose** — the reset strategy samples from ``np_random``. Under
  the default ``RL_GRID_STATIC`` reset this draws the spawn waypoint via
  ``rng.choice`` on the raceline, so the start ``x``/``y`` depend on the
  seed.
* **Per-agent LiDAR-noise streams** — ``reset`` draws an integer from
  ``np_random`` and passes it to the simulator as ``noise_seed``; each
  agent ``idx`` gets its own generator seeded at ``noise_seed + idx``.
  This drives the Gaussian scan ``noise_std``, per-beam ``dropout_prob``,
  and the once-per-episode per-beam ``range_bias_std`` (see
  :doc:`observations`).
* **Actuator-noise stream** — the simulator's ``control_rng`` is seeded
  at ``noise_seed + 2**20`` (offset well away from the per-agent scan
  seeds so the streams never collide). This drives ``steer_noise_std``
  and ``accl_noise_std`` from :class:`~f1tenth_gym.envs.env_config.ControlConfig`.
* **Domain-randomization draw** — when
  :class:`~f1tenth_gym.envs.env_config.DomainRandomizationConfig` is
  enabled, the per-episode vehicle-parameter sample is drawn from
  ``np_random`` before the sim resets, so randomized dynamics are
  reproducible with the reset seed too (see :doc:`sim2real`).

Inside ``reset`` the generator is consumed in a fixed order: the
spawn-pose draw, then the domain-randomization draw (only when enabled),
then the noise seed. The last section hangs off that order.

.. note::

   The physics kernels compute in float64 and the result is re-cast to
   float32 in the state buffers every step (:doc:`dynamics`).
   Deterministic — the same inputs always produce the same float32
   result — but trajectories are stored at float32 precision, and
   bit-for-bit portability assumes the same NumPy build.

``EnvConfig.seed`` seeds the whole run
--------------------------------------

``EnvConfig.seed`` (default ``None``) covers the first unseeded
``reset()``: that reset behaves as ``reset(seed=cfg.seed)``, and every
later unseeded reset continues the stream it started. The whole run —
however many episodes — becomes a deterministic function of the config:
episodes still differ from one another, but a rerun replays them exactly.
Two environments built from the same seeded config produce identical
streams, reset for reset:

>>> import gymnasium as gym
>>> import numpy as np
>>> from f1tenth_gym.envs.env_config import EnvConfig
>>> cfg = EnvConfig(seed=2024, render_enabled=False)
>>> e1 = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> e2 = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
>>> o1, _ = e1.reset()      # first unseeded reset: the config seed applies
>>> o2, _ = e2.reset()
>>> o1b, _ = e1.reset()     # later unseeded resets continue the stream...
>>> o2b, _ = e2.reset()     # ...identically in both envs
>>> (np.array_equal(o1["agent_0"]["scan"], o2["agent_0"]["scan"]),
...  np.array_equal(o1b["agent_0"]["scan"], o2b["agent_0"]["scan"]))
(True, True)
>>> o, _ = e1.reset(seed=42)          # an explicit seed always wins
>>> print(f"{float(o['agent_0']['std_state'][0]):.4f}")
-0.0441
>>> e1.close(); e2.close()

An explicit ``reset(seed=...)`` takes precedence whatever the config
says: ``-0.0441`` is the seed-42 spawn ``x`` under any ``EnvConfig.seed``.
``configure()`` re-arms the config seed, so the next unseeded reset after
a reconfiguration starts the stream over. The identical-streams property
cuts the other way in a vector env — sub-envs sharing one seeded config
all replay the *same* rollout, so give each sub-env its own seed there.

What an unseeded reset varies
-----------------------------

Leave both seeds unset and gymnasium seeds ``np_random`` from OS entropy
on every ``reset()``: episodes are unrepeatable by design — and the
variation is narrower than it looks. Measured over seeds 0–49 on
Spielberg, the spawn spans ``x`` in ``[-0.8167, -0.0441]`` and ``y`` in
``[-1.0562, -0.8492]`` — exactly five distinct spawn points inside a
window about one metre long. The default ``RL_GRID_STATIC`` strategy
masks the first ``int(start_width / (raceline.length / raceline.n))``
waypoints — 5 of Spielberg's 1692 at the default ``start_width=1.0`` —
and draws one via ``rng.choice``. The lateral offset is structural for
the same reason: ``ey`` stays inside ``[0.8080, 0.8086]`` across all 50
draws, a property of the reference lines rather than of the seed
(:doc:`tracks`). ``ResetConfig.start_width`` widens the window (metres,
clamped to at least one waypoint), and
``ResetConfig(reference_line=ReferenceLine.CENTERLINE)`` spawns on the
line the Frenet frame measures against.

First scans vary too: ``reset`` ends with a real LiDAR sweep (it informs
the first observation only — a spawn is never adjudicated as a
collision), so each agent's scan generator has already drawn
``num_beams`` noise values by the time you read ``scan``. Two unseeded
resets differ scan-for-scan; seed any equivalence test. For per-episode
variation a rerun can still replay, seed with a sequence —
``reset(seed=episode_index)`` — or set ``EnvConfig.seed`` and let the
stream run.

What silently shifts the stream
-------------------------------

Three ``np_random`` draws happen in a fixed order inside ``reset``, and
two of them are conditional, so a configuration switch moves every draw
after it — at the same seed:

* Enabling domain randomization inserts the parameter draw between the
  spawn draw and the noise seed. Even a degenerate range that changes no
  physics — ``low`` and ``high`` both equal to the base parameters — leaves
  ``reset(seed=42)`` with the same spawn but a first scan that differs by
  up to 0.047 m.
* ``options={"poses": ...}`` skips the spawn draw entirely. Handing
  ``reset(seed=42)`` the exact pose it would have sampled anyway still
  shifts the noise seed one draw earlier: same physical state, scans
  differing by up to 0.046 m.
* ``LiDARConfig.range_bias_std`` draws its per-episode bias from the
  same per-agent generator that then feeds the per-step scan noise, so
  two configs differing only in that field produce entirely different
  noise — not the same noise plus a bias.

.. warning::

   Two configs are not noise-paired even at the same seed. A
   single-rollout A/B difference mixes the setting's effect with a fresh
   noise realisation — compare distributions over seeds, and difference
   individual rollouts only when their configs are identical.
