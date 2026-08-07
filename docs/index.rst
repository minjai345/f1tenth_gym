.. image:: assets/f1_stickers_01.png
  :width: 60
  :align: left

F1TENTH Gym
===========

``f1tenth_gym`` races one or more 1/10th-scale cars on a real racetrack map
behind a single Gymnasium ``Env`` — single-track or kinematic vehicle dynamics,
a ray-cast 2D LiDAR, wall and car-to-car collisions, and a Frenet-frame view of
the track. Every agent is a row in one shared state buffer, and a frozen
:class:`~f1tenth_gym.envs.env_config.EnvConfig` is the only way in.

The shortest complete program
-----------------------------

Build, reset, apply one command, and read back what the car actually did:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from f1tenth_gym.envs.env_config import EnvConfig

   env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
   obs, info = env.reset(seed=42)
   action = np.array([[0.0, 2.0]], dtype=np.float32)   # [[steer_rad, speed_mps]]
   obs, reward, terminated, truncated, info = env.step(action)
   print(round(float(obs["agent_0"]["std_state"][3]), 3), reward)
   env.close()

This prints ``0.019 0.01``. The commanded 2.0 m/s is a target, not a state
write: one 10 ms step of the longitudinal controller reaches 0.019 m/s. The
0.01 is one step of the default survival reward — simulated seconds alive.

.. note::

   The id is namespaced — ``"f1tenth_gym:f1tenth-v0"`` tells Gymnasium which
   module to import first. Tutorials written for the legacy ``f110_gym``
   package (dict config, 4-tuple ``step``) will not run against this one.

Where to start
--------------

Pick the door that matches what you already have:

- Never run the simulator: :doc:`quickstart` drives one episode to its end and
  works out which of the two exit conditions stopped it.
- No working install yet: :doc:`installation` gets you an editable clone whose
  physics, map download and rendering are each verified by a printed number.
- Loop already running, need a knob: :doc:`configuration` lists every
  ``EnvConfig`` field, its default, and what it rejects.

Citing
------

If you use this environment in your work, please cite:

.. code-block:: bibtex

   @inproceedings{okelly2020f1tenth,
     title={{F1TENTH}: An Open-source Evaluation Environment for Continuous
            Control and Reinforcement Learning},
     author={O'Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and
             Mangharam, Rahul},
     booktitle={NeurIPS 2019 Competition and Demonstration Track},
     pages={77--89},
     year={2020},
     organization={PMLR}
   }

To build a physical 1/10th-scale car, follow the guide at
https://f1tenth.org/build.html.

.. toctree::
   :caption: Getting started
   :hidden:

   installation
   quickstart

.. toctree::
   :caption: Guides
   :hidden:

   rl
   rendering
   examples

.. toctree::
   :caption: Explanation
   :hidden:

   dynamics
   tracks
   sim2real
   reproducibility

.. toctree::
   :caption: Reference
   :hidden:

   configuration
   observations
   actions
   api/index
