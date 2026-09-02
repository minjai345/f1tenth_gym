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

Functional JAX simulation
-------------------------

JAX support follows the same responsibility-first package layout as the mutable
simulator. :class:`JaxSimulator` is the single host construction surface;
``CoreConfig``, ``CoreTables`` and ``CoreParams`` remain separate because JAX
needs static topology, fixed map tables and traced episode values to have
different compilation behavior.

.. autosummary::

   f1tenth_gym.envs.jax_simulator.JaxSimulator
   f1tenth_gym.envs.jax_simulator.IndexedJaxSimulator
   f1tenth_gym.envs.jax_simulator.JaxSimulator.reset
   f1tenth_gym.envs.jax_simulator.JaxSimulator.step
   f1tenth_gym.envs.jax_simulator.JaxSimulator.reset_batch
   f1tenth_gym.envs.jax_simulator.JaxSimulator.step_batch
   f1tenth_gym.envs.jax_simulator.JaxSimulator.step_batch_autoreset
   f1tenth_gym.envs.jax_env.JaxF110Env
   f1tenth_gym.envs.observation.jax_adapter.GymObservationAdapter
   f1tenth_gym.envs.jax_core.CoreConfig
   f1tenth_gym.envs.jax_core.CoreTables
   f1tenth_gym.envs.jax_core.CoreParams
   f1tenth_gym.envs.jax_core.CoreState
   f1tenth_gym.envs.jax_core.reset_core
   f1tenth_gym.envs.jax_core.step_core
   f1tenth_gym.envs.batching.PolicyLayout
   f1tenth_gym.envs.batching.reset_batch
   f1tenth_gym.envs.batching.step_batch
   f1tenth_gym.envs.batching.step_batch_autoreset
   f1tenth_gym.envs.indexed_batching.reset_indexed_batch
   f1tenth_gym.envs.indexed_batching.step_indexed_batch

.. automodule:: f1tenth_gym.envs.jax_simulator

.. automodule:: f1tenth_gym.envs.jax_env

.. automodule:: f1tenth_gym.envs.observation.jax_adapter

.. automodule:: f1tenth_gym.envs.jax_core

.. automodule:: f1tenth_gym.envs.batching

.. automodule:: f1tenth_gym.envs.indexed_batching

Optional trainer integrations
-----------------------------

Trainer dependencies stay outside the core package import path. The SBX
adapter is available after installing the ``sbx`` extra.

.. autosummary::

   f1tenth_gym.envs.sbx.F110SBXVecEnv

.. automodule:: f1tenth_gym.envs.sbx

Feature-owned functional modules
--------------------------------

Advanced callers can deep-import the pure fixed-shape implementation from the
subsystem that owns each operation. Package roots intentionally stay shallow.

.. automodule:: f1tenth_gym.envs.dynamic_models.jax

.. automodule:: f1tenth_gym.envs.dynamic_models.jax_core

.. automodule:: f1tenth_gym.envs.dynamic_models.randomization

.. automodule:: f1tenth_gym.envs.action_jax

.. automodule:: f1tenth_gym.envs.integrators_jax

.. automodule:: f1tenth_gym.envs.episode

.. automodule:: f1tenth_gym.envs.contact.functional

.. automodule:: f1tenth_gym.envs.contact.geometry

.. automodule:: f1tenth_gym.envs.contact.pairs

.. automodule:: f1tenth_gym.envs.lidar.functional

.. automodule:: f1tenth_gym.envs.lidar.kernels

.. automodule:: f1tenth_gym.envs.track.functional

.. automodule:: f1tenth_gym.envs.track.preprocessing

.. automodule:: f1tenth_gym.envs.reset.functional

.. automodule:: f1tenth_gym.envs.reset.preprocessing

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
