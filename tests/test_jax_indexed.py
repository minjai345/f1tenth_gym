"""Exact-shape indexed-map rollout and selective-reset gates."""

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.jax_core import reset_core, step_core
from f1tenth_gym.envs.indexed_batching import (
    IndexedCoreTables,
    reset_indexed_batch,
    reset_indexed_batch_from_poses,
    reset_indexed_batch_from_state,
    stack_core_tables,
    step_indexed_batch,
    step_indexed_batch_autoreset,
)
from tests.test_jax_batched import (
    _assert_tree_equal,
    _fixture,
    _randomization,
)


def _translated_tables(tables, distance=30.0):
    offset = jnp.asarray((distance, 0.0), dtype=jnp.float32)
    return replace(
        tables,
        reset=replace(
            tables.reset,
            waypoints=tables.reset.waypoints + offset,
        ),
    )


def _table_row(indexed, index):
    return jax.tree.map(lambda leaf: leaf[index], indexed.tables)


class TestIndexedTableStack(unittest.TestCase):
    def setUp(self):
        self.config, self.tables, self.params = _fixture(num_agents=1)

    def test_stack_requires_nonempty_exact_shapes(self):
        translated = _translated_tables(self.tables)
        indexed = stack_core_tables((self.tables, translated))
        self.assertIsInstance(indexed, IndexedCoreTables)
        self.assertEqual(indexed.num_maps, 2)
        self.assertEqual(indexed.tables.reset.waypoints.shape[0], 2)

        with self.assertRaisesRegex(ValueError, "at least one"):
            stack_core_tables(())
        with self.assertRaisesRegex(TypeError, "CoreTables"):
            stack_core_tables((object(),))
        incompatible = replace(
            translated,
            reset=replace(
                translated.reset,
                waypoints=translated.reset.waypoints[:-1],
            ),
        )
        with self.assertRaisesRegex(ValueError, "leaf shapes and dtypes"):
            stack_core_tables((self.tables, incompatible))

    def test_indexed_wrapper_validates_every_leading_axis(self):
        stacked = jax.tree.map(lambda leaf: leaf[None], self.tables)
        broken = replace(
            stacked,
            reset=replace(
                stacked.reset,
                waypoints=jnp.repeat(stacked.reset.waypoints, 2, axis=0),
            ),
        )
        with self.assertRaisesRegex(ValueError, "leading map axis"):
            IndexedCoreTables(broken, 1)


class TestIndexedBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.tables, cls.params = _fixture(num_agents=1)
        cls.indexed = stack_core_tables(
            (cls.tables, _translated_tables(cls.tables))
        )
        cls.randomization = _randomization(cls.params)

    def test_jitted_reset_matches_each_scalar_selected_table(self):
        keys = jax.random.split(jax.random.key(201), 4)
        map_indices = jnp.asarray((0, 1, 1, 0), dtype=jnp.int32)
        observation, state = jax.jit(
            reset_indexed_batch, static_argnums=3
        )(
            keys,
            map_indices,
            self.indexed,
            self.config,
            self.params,
            self.randomization,
        )
        self.assertEqual(observation.state.shape, (4, 1, 5))
        for row, map_index in enumerate(np.asarray(map_indices)):
            params = jax.tree.map(lambda leaf: leaf[row], state.params)
            expected = reset_core(
                keys[row],
                _table_row(self.indexed, map_index),
                self.config,
                params,
            )
            _assert_tree_equal(
                self,
                jax.tree.map(lambda leaf: leaf[row], observation),
                expected[0],
            )
            _assert_tree_equal(
                self,
                jax.tree.map(lambda leaf: leaf[row], state.core),
                expected[1],
            )

    def test_pose_and_state_overrides_keep_the_selected_map_for_geometry(self):
        keys = jax.random.split(jax.random.key(202), 2)
        map_indices = jnp.asarray((0, 1), dtype=jnp.int32)
        poses = jnp.asarray(
            [[[8.0, 0.0, 1.57]], [[38.0, 0.0, 1.57]]],
            dtype=jnp.float32,
        )
        observation, _state = jax.jit(
            reset_indexed_batch_from_poses, static_argnums=4
        )(
            keys,
            map_indices,
            poses,
            self.indexed,
            self.config,
            self.params,
            self.randomization,
        )
        np.testing.assert_allclose(observation.state[..., :2], poses[..., :2])

        model = observation.state.at[..., 3].set(2.0)
        replay, _state = jax.jit(
            reset_indexed_batch_from_state, static_argnums=4
        )(
            keys,
            map_indices,
            model,
            self.indexed,
            self.config,
            self.params,
            self.randomization,
        )
        np.testing.assert_allclose(replay.state, model)

    def test_jitted_step_matches_scalar_core_on_each_map(self):
        keys = jax.random.split(jax.random.key(203), 3)
        map_indices = jnp.asarray((0, 1, 0), dtype=jnp.int32)
        _observation, state = reset_indexed_batch(
            keys,
            map_indices,
            self.indexed,
            self.config,
            self.params,
            self.randomization,
        )
        step_keys = jax.random.split(jax.random.key(204), 3)
        actions = jnp.zeros((3, 1, 2), dtype=jnp.float32)
        transition = jax.jit(
            step_indexed_batch, static_argnums=5
        )(
            step_keys,
            map_indices,
            state,
            actions,
            self.indexed,
            self.config,
        )
        for row, map_index in enumerate(np.asarray(map_indices)):
            expected = step_core(
                step_keys[row],
                jax.tree.map(lambda leaf: leaf[row], state.core),
                actions[row],
                _table_row(self.indexed, map_index),
                self.config,
                jax.tree.map(lambda leaf: leaf[row], state.params),
            )
            actual = (
                jax.tree.map(lambda leaf: leaf[row], transition.observation),
                jax.tree.map(lambda leaf: leaf[row], transition.state.core),
                transition.rewards[row],
                jax.tree.map(lambda leaf: leaf[row], transition.events),
                jax.tree.map(lambda leaf: leaf[row], transition.metrics),
            )
            _assert_tree_equal(self, actual, expected)

    def test_autoreset_uses_each_rows_selected_map(self):
        episode_params = replace(
            self.params.episode,
            step_limit_enabled=jnp.asarray(True),
            max_episode_steps=jnp.asarray(1, dtype=jnp.int32),
        )
        params = replace(self.params, episode=episode_params)
        one_key = jax.random.key(205)
        keys = jnp.stack((one_key, one_key))
        map_indices = jnp.asarray((0, 1), dtype=jnp.int32)
        _observation, state = reset_indexed_batch(
            keys,
            map_indices,
            self.indexed,
            self.config,
            params,
            self.randomization,
        )
        transition = jax.jit(
            step_indexed_batch_autoreset, static_argnums=6
        )(
            keys,
            keys,
            map_indices,
            state,
            jnp.zeros((2, 1, 2), dtype=jnp.float32),
            self.indexed,
            self.config,
            params,
            self.randomization,
        )
        np.testing.assert_array_equal(transition.reset, (True, True))
        x = np.asarray(transition.next_observation.state[:, 0, 0])
        self.assertAlmostEqual(float(x[1] - x[0]), 30.0, places=5)
        np.testing.assert_array_equal(
            transition.state.core.episode.elapsed_steps,
            (0, 0),
        )

    def test_map_index_shape_and_dtype_are_explicit(self):
        keys = jax.random.split(jax.random.key(206), 2)
        with self.assertRaisesRegex(ValueError, "map_indices must have shape"):
            reset_indexed_batch(
                keys,
                jnp.asarray((0,), dtype=jnp.int32),
                self.indexed,
                self.config,
                self.params,
                self.randomization,
            )
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            reset_indexed_batch(
                keys,
                jnp.asarray((0.0, 1.0), dtype=jnp.float32),
                self.indexed,
                self.config,
                self.params,
                self.randomization,
            )


if __name__ == "__main__":
    unittest.main()
