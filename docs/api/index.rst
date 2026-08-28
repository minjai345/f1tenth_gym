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
   f1tenth_gym.envs.env_config.AgentTerminationMode
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
   f1tenth_gym.envs.dynamic_models.VehicleParameters

.. automodule:: f1tenth_gym.envs.dynamic_models

Functional JAX kernels
----------------------

These are the device-compatible building blocks under active migration. They
are pure functions, not yet a complete Gymnasium environment.

.. autosummary::

   f1tenth_gym.jax.DynamicsParams
   f1tenth_gym.jax.DynamicsConfig
   f1tenth_gym.jax.DynamicsState
   f1tenth_gym.jax.EpisodeParams
   f1tenth_gym.jax.SplineTable
   f1tenth_gym.jax.TrackTable
   f1tenth_gym.jax.ResetTable
   f1tenth_gym.jax.ResetConfig
   f1tenth_gym.jax.BodyParams
   f1tenth_gym.jax.ScanConfig
   f1tenth_gym.jax.ScanParams
   f1tenth_gym.jax.ContactParams
   f1tenth_gym.jax.WallContactConfig
   f1tenth_gym.jax.LongitudinalControlMode
   f1tenth_gym.jax.SteeringControlMode
   f1tenth_gym.jax.kinematic_single_track
   f1tenth_gym.jax.single_track
   f1tenth_gym.jax.adapt_actions
   f1tenth_gym.jax.euler_step
   f1tenth_gym.jax.rk4_step
   f1tenth_gym.jax.integrate_substeps
   f1tenth_gym.jax.make_dynamics_state
   f1tenth_gym.jax.step_dynamics
   f1tenth_gym.jax.rollout_dynamics
   f1tenth_gym.jax.evaluate_spline
   f1tenth_gym.jax.cartesian_to_frenet
   f1tenth_gym.jax.frenet_to_cartesian
   f1tenth_gym.jax.sample_reset_poses
   f1tenth_gym.jax.reset_dynamics_state
   f1tenth_gym.jax.body_vertices
   f1tenth_gym.jax.lidar_poses
   f1tenth_gym.jax.clean_scan
   f1tenth_gym.jax.world_velocity
   f1tenth_gym.jax.apply_contact_response
   f1tenth_gym.jax.resolve_wall_contacts
   f1tenth_gym.jax.preprocess.build_track_table
   f1tenth_gym.jax.preprocess.build_track_table_set
   f1tenth_gym.jax.preprocess.build_reset_table
   f1tenth_gym.jax.preprocess.build_scan_params
   f1tenth_gym.jax.preprocess.bucket_track_tables
   f1tenth_gym.jax.preprocess.compare_batch_layout

.. automodule:: f1tenth_gym.jax

.. automodule:: f1tenth_gym.jax.dynamics

.. automodule:: f1tenth_gym.jax.controls

.. automodule:: f1tenth_gym.jax.core

.. automodule:: f1tenth_gym.jax.contact

.. automodule:: f1tenth_gym.jax.integrators

.. automodule:: f1tenth_gym.jax.geometry

.. automodule:: f1tenth_gym.jax.lidar

.. automodule:: f1tenth_gym.jax.lidar_kernels

.. automodule:: f1tenth_gym.jax.track

.. automodule:: f1tenth_gym.jax.reset

.. automodule:: f1tenth_gym.jax.preprocess

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

One OpenGL backend sits behind the :class:`EnvRenderer` ABC; the factory
picks it from the render mode.

.. autosummary::

   f1tenth_gym.envs.rendering.make_renderer
   f1tenth_gym.envs.rendering.renderer.EnvRenderer
   f1tenth_gym.envs.rendering.renderer.ObjectRenderer

.. automodule:: f1tenth_gym.envs.rendering.renderer

.. automodule:: f1tenth_gym.envs.rendering.callbacks
