API Reference
=============

Generated from the docstrings of the live modules. The paths documented here
are the paths you import from — nothing is re-exported at the package roots,
so ``from f1tenth_gym import EnvConfig`` raises ``ImportError``; always
deep-import (``from f1tenth_gym.envs.env_config import EnvConfig``). For a
task-oriented introduction start with :doc:`../configuration`.

Environment & configuration
---------------------------

.. autosummary::

   f1tenth_gym.envs.env_config.EnvConfig
   f1tenth_gym.envs.env_config.ControlConfig
   f1tenth_gym.envs.env_config.SimulationConfig
   f1tenth_gym.envs.env_config.ObservationConfig
   f1tenth_gym.envs.env_config.ResetConfig
   f1tenth_gym.envs.env_config.RenderConfig
   f1tenth_gym.envs.env_config.TerminationConfig
   f1tenth_gym.envs.env_config.RewardConfig
   f1tenth_gym.envs.env_config.DomainRandomizationConfig
   f1tenth_gym.envs.env_config.LoopCounterMode
   f1tenth_gym.envs.env_config.RewardMode
   f1tenth_gym.envs.f110_env.F110Env
   f1tenth_gym.envs.wrappers.SingleAgentWrapper
   f1tenth_gym.envs.wrappers.ObservationDelayWrapper

.. automodule:: f1tenth_gym.envs.env_config

.. automodule:: f1tenth_gym.envs.f110_env

.. automodule:: f1tenth_gym.envs.wrappers

Simulation core
---------------

.. autosummary::

   f1tenth_gym.envs.simulator.F110Simulator
   f1tenth_gym.envs.state.SimulationState
   f1tenth_gym.envs.integrators.IntegratorType

.. automodule:: f1tenth_gym.envs.simulator

.. automodule:: f1tenth_gym.envs.state

.. automodule:: f1tenth_gym.envs.integrators

Vehicle dynamics
----------------

.. autosummary::

   f1tenth_gym.envs.dynamic_models.DynamicModel
   f1tenth_gym.envs.dynamic_models.PoseReference
   f1tenth_gym.envs.dynamic_models.VehicleParameters
   f1tenth_gym.envs.dynamic_models.VehicleParamRanges

.. automodule:: f1tenth_gym.envs.dynamic_models

Actions
-------

.. autosummary::

   f1tenth_gym.envs.action.LongitudinalActionType
   f1tenth_gym.envs.action.SteerActionType
   f1tenth_gym.envs.action.get_action_space

.. automodule:: f1tenth_gym.envs.action

Observations
------------

The public entry points live on the package (``f1tenth_gym.envs.observation``);
``FullObservation`` and ``RawObservation`` are the two concrete providers.

.. autosummary::

   f1tenth_gym.envs.observation.ObservationType
   f1tenth_gym.envs.observation.observation_factory
   f1tenth_gym.envs.observation.FullObservation
   f1tenth_gym.envs.observation.RawObservation

.. automodule:: f1tenth_gym.envs.observation

.. automodule:: f1tenth_gym.envs.observation.full

.. automodule:: f1tenth_gym.envs.observation.raw

Sensing & collision
-------------------

.. autosummary::

   f1tenth_gym.envs.lidar.LiDARConfig
   f1tenth_gym.envs.lidar.ScanSimulator2D
   f1tenth_gym.envs.lidar.check_collision
   f1tenth_gym.envs.lidar.ray_cast
   f1tenth_gym.envs.collision_models.CollisionCheckMode

.. automodule:: f1tenth_gym.envs.lidar

.. automodule:: f1tenth_gym.envs.collision_models

Track & Frenet frame
--------------------

.. autosummary::

   f1tenth_gym.envs.track.Track
   f1tenth_gym.envs.track.TrackSpec
   f1tenth_gym.envs.track.Raceline
   f1tenth_gym.envs.track.find_track_dir

.. automodule:: f1tenth_gym.envs.track

.. automodule:: f1tenth_gym.envs.track.cubic_spline

Reset strategies
----------------

.. autosummary::

   f1tenth_gym.envs.reset.ResetStrategy
   f1tenth_gym.envs.reset.ReferenceLine
   f1tenth_gym.envs.reset.make_reset_fn

.. automodule:: f1tenth_gym.envs.reset

Rendering
---------

.. autosummary::

   f1tenth_gym.envs.rendering.make_renderer
   f1tenth_gym.envs.rendering.renderer.EnvRenderer
   f1tenth_gym.envs.rendering.renderer.ObjectRenderer

.. automodule:: f1tenth_gym.envs.rendering.renderer

.. automodule:: f1tenth_gym.envs.rendering.callbacks
