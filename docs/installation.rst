How to install f1tenth_gym
==========================

``f1tenth_gym`` resolves its ``maps/`` directory four ``.parent`` hops up from its
own source file and downloads tracks into it on first use, so an editable clone is
the layout that behaves — installed from a wheel, the package would fetch maps into
``site-packages/maps/`` instead. Python 3.12 or newer is required, because the
JAX kernels the contact and collision code is built on need it.

Install from a clone
--------------------

Clone the repository and sync the environment with `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/f1tenth/f1tenth_gym.git
   cd f1tenth_gym
   uv sync

``uv sync`` creates ``.venv/`` and installs the package (editable) with its runtime
dependencies plus two default groups: ``dev`` (pytest, flake8, black, ...) and
``examples`` (``moviepy``, ``shapely``, ``matplotlib`` — what the bundled scripts
need beyond the library; skip them with ``uv sync --no-group examples``). Run
subsequent commands through ``uv run``, e.g. ``uv run python your_script.py``.

.. warning::

   ``uv sync`` is an *exact* sync: it uninstalls anything it cannot derive from
   ``pyproject.toml`` and ``uv.lock``. Install extra packages by adding them to
   the project rather than with ``uv pip install``, which the next sync undoes.

``uv sync`` installs GPU JAX by default. On a machine without CUDA, add
``--no-group gpu`` to skip roughly 3 GB of GPU wheels; plain ``jax`` remains a
hard dependency either way and runs CPU-only.

Verify the install
------------------

One seeded run exercises the map download, the physics and the observation
pipeline end to end:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import (
       EnvConfig,
       ObservationConfig,
       SimulationConfig,
   )
   from f1tenth_gym.envs.observation import ObservationType

   cfg = EnvConfig(
       map_name="Spielberg",
       num_agents=1,
       simulation_config=SimulationConfig(max_laps=None),  # default 1: one lap
       observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE),
       render_enabled=False,
   )
   env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
   obs, info = env.reset(seed=42)

   for _ in range(100):
       action = np.array([[0.0, 2.0]], dtype=np.float32)  # [[steer_rad, speed_mps]]
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break

   print(obs["agent_0"]["pose_x"], obs["agent_0"]["pose_y"],
         obs["agent_0"]["linear_vel_x"], info["sim_time"])
   env.close()

On a fresh machine this starts with ``Downloading Files for: Spielberg``: the track
comes from ``https://api.f1tenth.org/Spielberg.tar.xz`` and lands in the repo-root
``maps/`` directory — the path the editable layout keeps inside the clone. Network
access is needed once per track; the cached map is then reused offline. The loop
prints::

   -1.5670079 -1.257271 1.9840685 1.0000000000000007

A healthy install prints a speed near the commanded ``2.0`` m/s and a ``sim_time``
around ``1.0`` after 100 steps of ``timestep=0.01``. With ``seed=42`` the whole run
is reproducible: ``x=-1.567``, ``y=-1.257`` is where the loop ends (the spawn is
``x=-0.044``, ``y=-0.849``).

The snippet asks for ``KINEMATIC_STATE`` because the default observation preset
carries the pose inside ``std_state`` rather than as separate scalars — see
:doc:`observations`, and :doc:`actions` for what the two action columns mean.

Turn on rendering
-----------------

The renderer is an OpenGL backend (``pyqtgraph.opengl`` / PyQt6) and needs an X
display — a real one, or a virtual one via ``xvfb``; there is no offscreen
fallback. Grab a frame to prove the display path works:

.. code-block:: python

   import gymnasium as gym
   from f1tenth_gym.envs.env_config import EnvConfig

   env = gym.make(
       "f1tenth_gym:f1tenth-v0", config=EnvConfig(), render_mode="rgb_array"
   )
   env.reset(seed=42)
   frame = env.render()
   print(frame.shape, frame.dtype)
   env.close()

This prints ``(800, 800, 3) uint8`` — an RGB frame with the cars drawn, sized by
``RenderConfig.window_size``. With no ``$DISPLAY`` the same script raises a
``RuntimeError`` whose message walks through the fix — ``xvfb-run -a python
render_check.py`` on a headless server, ``pyvirtualdisplay`` on Colab. Render
modes, pacing and video recording are covered in :doc:`rendering`.

Run the test suite
------------------

Expect network traffic and a rewritten ``maps/`` directory: the download test
renames and restores ``maps/Spielberg``, and the renderer cases skip themselves
when ``$DISPLAY`` is unset. From the repository root:

.. code-block:: bash

   env -u PYTHONPATH uv run pytest

The ``env -u PYTHONPATH`` prefix matters only when ROS 2 Humble is on your
``PYTHONPATH``: ``/opt/ros/humble`` registers a pytest plugin whose import chain is
not available in the venv, which breaks collection before any test runs. Without
ROS on the path, plain ``uv run pytest`` works.

Next: :doc:`quickstart` drives one episode to its end and explains why it stopped.
