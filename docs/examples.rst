How to run the bundled examples
===============================

Six scripts live in ``examples/``. Five drive the simulator through the
ordinary Gymnasium API; ``jax_ppo_training.py`` compiles a complete PPO
training job over the pure JAX environment batch. Invoke any of them by file
path from the repository root: Python puts the script's own directory on
``sys.path``, which is what resolves their sibling imports of
``PurePursuitPlanner``.

Run the first example
---------------------

``uv sync`` already installs what all six scripts need beyond the library. The
``examples`` dependency group pulls in ``video`` (``moviepy``), ``trackgen``
(``shapely`` and ``matplotlib``), and ``train`` (Flax and Optax). SBX remains
a separate opt-in integration because it also brings the sizeable SB3/PyTorch
stack; see :doc:`installation`. From the repository root:

.. code-block:: bash

   uv run python examples/waypoint_follow.py

A window opens on Spielberg, a car follows the optimised racing line for one
lap, and the script prints::

   Sim elapsed time: 45.99999999999942 Real elapsed time: 46.0000114440918

Simulated and wall time agree because ``render_mode="human"`` paces the loop
at one simulated second per wall second; the same lap takes 4.5 s once
``render_enabled=False`` removes the renderer. Anything that draws needs an X
display, real or virtual — the OpenGL backend has no offscreen path
(:doc:`rendering`) — and the first run of a script that names a track
downloads it from the F1TENTH map server, so allow network access once.
Stopping a script that holds a window is abrupt: the display render modes
hand ``SIGINT`` back to the default handler, so Ctrl-C ends the process
without unwinding through ``env.close()``.

Pick a script
-------------

The six differ in what they draw and in what they cost to start:

.. list-table::
   :header-rows: 1
   :widths: 44 30 26

   * - To see
     - Run
     - Needs
   * - A lap on the racing line, and every config field written out
     - ``waypoint_follow.py``
     - display
   * - An MP4 of a rollout
     - ``video_recording.py``
     - display, ``video``
   * - Speed, steering, yaw rate and slip plotted live
     - ``telemetry_plot.py``
     - display (Qt, no GL)
   * - A car on a hand-written reference line, no map
     - ``run_in_empty_track.py``
     - display
   * - Randomly generated circuits written to disk
     - ``random_trackgen.py``
     - ``trackgen``
   * - End-to-end JAX PPO over a GPU-vectorized simulator batch
     - ``jax_ppo_training.py``
     - ``train``, CUDA for ``--device gpu``

Train PPO entirely in JAX
-------------------------

Launch the default 256-environment PPO job on one GPU:

.. code-block:: bash

   uv sync --extra train
   uv run --extra train python examples/jax_ppo_training.py \
       --device gpu --num-envs 256

``jax_ppo_training.py`` follows the PureJaxRL single-file structure. One
:class:`~f1tenth_gym.envs.jax_simulator.JaxSimulator` owns the fixed topology;
an outer ``lax.scan`` performs PPO updates, with nested scans for rollouts,
timeout-correct GAE, epochs and minibatches. Environment state, independently
domain-randomized vehicle parameters, policy inference and Optax state remain
on the selected device. Only the final metrics and optional Flax checkpoint
cross to the host.

The policy observes ego steering, speed, yaw rate, slip angle and Frenet
errors. Its tanh-bounded actions are scaled against each environment row's
active steering-rate and acceleration limits. The regular job keeps
Spielberg wall contact, collision termination, randomized raceline resets and
the built-in progress reward enabled; LiDAR is disabled to keep the example
focused on the training architecture. ``--num-envs * --rollout-steps`` is one
PPO sample batch and must divide ``--total-timesteps``.

This command proves the compiled native training path and one PPO update
without downloading a map or requiring CUDA:

.. code-block:: bash

   uv run --extra train python examples/jax_ppo_training.py \
       --smoke-test --device cpu --num-envs 4 \
       --rollout-steps 2 --total-timesteps 8 \
       --update-epochs 1 --minibatches 2 --no-save

Drive a lap with pure pursuit
-----------------------------

``waypoint_follow.py`` does two jobs at once. ``PurePursuitPlanner`` binds to
``track.raceline`` — the optimised line, not the centerline — and each step
turns the ego pose into a ``(speed, steering)`` command chasing a lookahead
point, through numba-jitted geometry helpers. Its two render callbacks draw
that point and the local plan over the map, and ``make_lidar_scan_callback``
adds the live scan as a point cloud.

``build_config()`` is the other job: an
:class:`~f1tenth_gym.envs.env_config.EnvConfig` with every field of every
nested config dataclass written out and annotated, almost all at their
default. Three fields added in this release are the only omissions —
``ControlConfig.steer_kp``, ``ResetConfig.reference_line`` and
``ResetConfig.start_width``, each left at its default (:doc:`configuration`).
The one deliberate deviation is the observation type:
``ObservationType.KINEMATIC_STATE`` hands the follower ``pose_x``, ``pose_y``
and ``pose_theta`` as separate scalars, which the default preset packs into
``std_state`` instead (:doc:`observations`). Set ``render_enabled=False`` in
``build_config()`` to run the whole thing headless — callbacks are wired up
only when a renderer exists, so nothing else changes.

The follower degrades in two stages as it leaves the line. Between the
lookahead radius and ``max_reacquire`` (20 m) it steers at the nearest
raceline point rather than an interpolated one; past 20 m ``plan()`` gives up
and returns a fixed ``speed=4.0, steer=0.0`` until the car is back in range.

Record the rollout to MP4
-------------------------

``video_recording.py`` wraps an ``rgb_array`` env in gymnasium's
``RecordVideo``, resets from an explicit spawn pose
(``env.reset(options={"poses": poses})``) at the first raceline point, and
drives five simulated seconds of pure pursuit:

.. code-block:: bash

   uv run python examples/video_recording.py

``env.close()`` flushes the encode, and moviepy writes 502 frames to
``video_<timestamp>/rl-video-episode-0.mp4`` under the current working
directory, at the frame rate the env advertises rather than ``render_fps``
(:doc:`rendering`). Gymnasium's passive checker warns once at that reset: the
raceline's stored yaw for the spawn point is 3.403 rad, outside the
``pose_theta`` bound of ±π, and the wrap to -2.880 happens on the first step.

Watch the state live
--------------------

``telemetry_plot.py`` opens a `pyqtgraph <https://www.pyqtgraph.org/>`_
window and plots four channels — speed, steering angle, yaw rate and slip
angle — straight out of ``obs["agent_0"]["std_state"]``, the ``(7,)``
standardized vector the default observation preset carries:

.. code-block:: bash

   uv run python examples/telemetry_plot.py --map Spielberg --rtf 1.0

Nothing in it imports the OpenGL backend — it builds the env with
``render_enabled=False`` — so it runs without a GPU, though the Qt window
still wants a display. Refresh rate and physics rate stay independent, so the
sim can take several steps per plotted frame. ``max_laps=None`` keeps the
rollout going and the dashboard resets itself if the car crashes.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Flag (default)
     - Meaning
   * - ``--map`` (``Spielberg``)
     - track name
   * - ``--rtf`` (``1.0``)
     - simulated seconds per wall second
   * - ``--fps`` (``50``)
     - plot refresh rate
   * - ``--window`` (``10``)
     - seconds of history on screen

Build your own track
--------------------

Two scripts author a circuit instead of downloading one.
``run_in_empty_track.py`` builds a mapless track from a hand-written
reference line with ``Track.from_refline(x, y, velx)`` and passes the
resulting ``Track`` object straight through as ``map_name``, which also shares
cached wall and ray-tile preprocessing between environments (:doc:`tracks`):

.. code-block:: bash

   uv run python examples/run_in_empty_track.py

Its line runs from ``(0, 0)`` to ``(10, 0)`` and is force-closed into a 20 m
loop with a phantom return leg; no open-path mode exists (:doc:`tracks`). The
car reaches the far end, loses the line past ``max_reacquire``, and then
drives straight ahead at 4 m/s indefinitely — it completes no lap and never
terminates, so stop it yourself.

``random_trackgen.py`` generates closed circuits with shapely and draws them
with matplotlib:

.. code-block:: bash

   uv run python examples/random_trackgen.py --n-maps 1 --outdir my_tracks

Each track lands as five flat files in ``--outdir`` (default ``./maps``):
``map0.png`` and ``map0.pgm``, a ``map0.yaml`` spec, and
``map0_centerline.csv`` / ``map0_raceline.csv``. ``--seed`` fixes the
geometry. Load one back by its stem — ``EnvConfig(map_name="my_tracks/map0")``
— which resolves through ``Track.from_track_path`` and drives like any
downloaded map.
