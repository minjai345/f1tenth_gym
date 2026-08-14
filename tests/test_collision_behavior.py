"""Env-level collision behaviour regressions.

Covers two things:
* a collision must halt the car (zero velocity) but PRESERVE its yaw -- the old
  code did ``state[3:] = 0`` which also wiped yaw (index 4), snapping the
  heading to 0/east.
* a car centred in the track must never falsely collide (f1tenth/f1tenth_gym
  issue #91 "unexpected collision ... without any visible collision").
"""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig


def _env(map_name="Spielberg"):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            map_name=map_name,
            simulation_config=SimulationConfig(max_laps=None),
            render_enabled=False,
        ),
    )


class TestCollisionBehaviour(unittest.TestCase):
    def test_collision_preserves_yaw_and_halts(self):
        env = _env()
        env.reset(seed=1)
        prev_yaw = None
        for _ in range(400):
            obs, _, done, _, _ = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            s = obs["agent_0"]["std_state"]
            yaw, speed = float(s[4]), float(s[3])
            if obs["agent_0"]["collision"] > 0:
                # velocity halted...
                self.assertAlmostEqual(speed, 0.0, places=5, msg="collision did not halt the car")
                # ...but yaw preserved (not snapped to 0, continuous with prev step)
                self.assertNotEqual(yaw, 0.0, "collision zeroed the yaw")
                if prev_yaw is not None:
                    self.assertLess(
                        abs(yaw - prev_yaw), 0.3,
                        f"yaw jumped on collision ({prev_yaw:.3f} -> {yaw:.3f}); likely reset to 0",
                    )
                # state and standard_state must agree on the halt
                self.assertAlmostEqual(float(obs["agent_0"]["state"][3]), 0.0, places=5)
                break
            prev_yaw = yaw
        else:
            self.fail("no collision produced in 400 steps of hard cornering")
        env.close()

    def test_no_false_collision_at_centerline(self):
        """A stationary car centred on the track centreline must not collide (#91)."""
        env = _env()
        cl = env.unwrapped.track.centerline
        cx = np.asarray(cl.xs, dtype=float)
        cy = np.asarray(cl.ys, dtype=float)
        yaw = np.arctan2(np.gradient(cy), np.gradient(cx))
        n = len(cx)
        stride = max(1, n // 100)
        false_collisions = []
        for k in range(0, n, stride):
            env.reset(options={"poses": np.array([[cx[k], cy[k], yaw[k]]], dtype=np.float32)})
            obs, *_ = env.step(np.array([[0.0, 0.0]], dtype=np.float32))
            if obs["agent_0"]["collision"] > 0:
                false_collisions.append((k, float(obs["agent_0"]["scan"].min())))
        self.assertEqual(
            false_collisions, [],
            f"false collisions at centreline points (idx, min_scan): {false_collisions}",
        )
        env.close()


class TestCollisionModes(unittest.TestCase):
    def test_none_mode_disables_collisions(self):
        """CollisionCheckMode.NONE: driving into a wall never flags a collision (#124)."""
        from f1tenth_gym.envs.collision_models import CollisionCheckMode
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                simulation_config=SimulationConfig(max_laps=None),
                collision_check=CollisionCheckMode.NONE,
                render_enabled=False,
            ),
        )
        env.reset(seed=1)
        for _ in range(400):
            obs, _, done, _, _ = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            self.assertEqual(float(obs["agent_0"]["collision"]), 0.0, "NONE mode flagged a collision")
            self.assertFalse(done, "NONE mode terminated on collision")
        env.close()

    def test_lidar_disabled_does_not_crash(self):
        """lidar_config.enabled=False must not crash step() in any collision mode (#124)."""
        from f1tenth_gym.envs.collision_models import CollisionCheckMode
        from f1tenth_gym.envs.lidar import LiDARConfig
        for mode in (CollisionCheckMode.LIDAR_SCAN, CollisionCheckMode.BOUNDING_BOX, CollisionCheckMode.NONE):
            env = gym.make(
                "f1tenth_gym:f1tenth-v0",
                config=EnvConfig(
                    simulation_config=SimulationConfig(max_laps=None),
                    lidar_config=LiDARConfig(enabled=False),
                    collision_check=mode,
                    num_agents=2,
                    render_enabled=False,
                ),
            )
            env.reset(seed=1)
            for _ in range(20):
                env.step(np.zeros((2, 2), dtype=np.float32))
            env.close()

    def test_bounding_box_does_not_rebind_collisions(self):
        """BOUNDING_BOX must update state.collisions in place, not rebind it
        (a rebind leaves any captured sim.state.collisions handle stale)."""
        from f1tenth_gym.envs.collision_models import CollisionCheckMode
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                simulation_config=SimulationConfig(max_laps=None),
                collision_check=CollisionCheckMode.BOUNDING_BOX,
                render_enabled=False,
            ),
        )
        env.reset(seed=1)
        handle = env.unwrapped.sim.state.collisions
        for _ in range(5):
            env.step(np.array([[0.2, 3.0]], dtype=np.float32))
        self.assertIs(handle, env.unwrapped.sim.state.collisions, "collisions array was rebound")
        env.close()


class TestHaltIsModelAware(unittest.TestCase):
    """Pins ISSUES_PLAN.md #26: the halt zeroes velocities only, per model."""

    def test_mb_stays_finite_after_a_halt(self):
        from f1tenth_gym.envs.dynamic_models import DynamicModel, FULLSCALE_VEHICLE_PARAMETERS
        from f1tenth_gym.envs.env_config import ResetConfig
        from f1tenth_gym.envs.reset import ResetStrategy

        cfg = EnvConfig(
            map_scale=10.0,
            params=FULLSCALE_VEHICLE_PARAMETERS,
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.MB, max_laps=None
            ),
            reset_config=ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC),
            render_enabled=False,
        )
        env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
        env.reset(seed=3)
        for _ in range(15):
            env.step(np.array([[0.0, 2.0]], dtype=np.float32))
        sim = env.unwrapped.sim
        sim._halt_on_collision(0)
        # ride height must survive the halt -- the old [5:]=0 wiped it and the
        # suspension math divided by zero on the next step
        self.assertNotEqual(float(sim.state.state[0, 11]), 0.0)
        for _ in range(20):
            env.step(np.array([[0.0, 2.0]], dtype=np.float32))
            self.assertTrue(np.isfinite(sim.state.state[0]).all())
        env.close()

    def test_st_halt_zeroes_velocities_and_rewinds_the_pose(self):
        """ST halt semantics: velocities die, the pose rewinds one step.

        Was ``test_st_halt_semantics_unchanged``. #28 added the pose rewind, so
        ``(x, y, yaw)`` now come back as the pre-step values rather than the
        post-integration ones. The steering angle is still carried over
        untouched, and yaw is still never snapped to 0/east (the original bug
        this test was written for).
        """
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
        env.reset(seed=1)
        for _ in range(10):
            env.step(np.array([[0.2, 3.0]], dtype=np.float32))
        sim = env.unwrapped.sim
        pre = sim._pre_pose[0].copy()          # pose at the top of the last step
        moved_yaw = float(sim.state.state[0, 4])
        steer = float(sim.state.state[0, 2])

        sim._halt_on_collision(0)
        s = sim.state.state[0]

        self.assertEqual(float(s[3]), 0.0)     # speed
        self.assertEqual(float(s[5]), 0.0)     # yaw rate
        self.assertEqual(float(s[6]), 0.0)     # slip angle
        self.assertEqual(float(s[2]), steer)   # steering angle carried over
        # pose rewound to the start of the step, NOT left at the moved pose
        self.assertEqual(float(s[0]), float(pre[0]))
        self.assertEqual(float(s[1]), float(pre[1]))
        self.assertEqual(float(s[4]), float(pre[4]))
        # ...and still a real heading, never snapped to east
        self.assertNotEqual(float(s[4]), 0.0)
        self.assertAlmostEqual(float(s[4]), moved_yaw, places=2)
        env.close()


class TestHaltRejectsTheMove(unittest.TestCase):
    """Pins ISSUES_PLAN.md #28: a halted car never ends up inside geometry.

    Zeroing the velocity is not enough on its own. The dynamics integrate
    before the collision check, so a car held against a wall re-accelerates
    from v=0 every step and keeps the few hundred micrometres of penetration
    it gained. That accumulates monotonically: before the fix it crossed
    Spielberg's 23 cm walls in ~1800 steps and drove out the far side.
    """

    @staticmethod
    def _occupancy_probe(track):
        """(x, y) -> occupancy cell value; 0.0 is a wall, 255.0 is free."""
        from f1tenth_gym.envs.lidar.laser_models import xy_2_rc

        occ = track.occupancy_map
        h, w = occ.shape
        ox, oy = track.spec.origin[0], track.spec.origin[1]
        res = track.spec.resolution

        def probe(x, y):
            r, c = xy_2_rc(float(x), float(y), ox, oy, 1.0, 0.0, h, w, res)
            return float(occ[r, c])

        return probe

    def _drive_into_wall(self, steps, terminate=False):
        from f1tenth_gym.envs.env_config import TerminationConfig

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                simulation_config=SimulationConfig(max_laps=None),
                termination_config=TerminationConfig(terminate_on_collision=terminate),
                render_enabled=False,
            ),
        )
        env.reset(seed=42)
        sim = env.unwrapped.sim
        probe = self._occupancy_probe(env.unwrapped.track)
        action = np.array([[0.41, 6.0]], dtype=np.float32)  # full lock, hard on the throttle
        worst = 255.0
        hit = False
        for _ in range(steps):
            obs, _r, term, _t, info = env.step(action)
            if info["collisions"][0]:
                hit = True
            s = sim.state.standard_state[0]
            worst = min(worst, probe(s[0], s[1]))
            if term:
                break
        pose = sim.state.standard_state[0].copy()
        env.close()
        return hit, worst, pose

    def test_car_never_enters_the_wall(self):
        hit, worst_cell, _pose = self._drive_into_wall(steps=2500)
        self.assertTrue(hit, "the car never reached the wall -- scenario is wrong")
        # 0.0 would mean the car's centre occupied a wall cell at some point
        self.assertEqual(
            worst_cell, 255.0,
            "the car's centre entered an occupied cell: the halt let it tunnel",
        )

    def test_car_stays_put_while_pinned(self):
        _hit, _worst, pose_a = self._drive_into_wall(steps=800)
        _hit2, _worst2, pose_b = self._drive_into_wall(steps=2500)
        # 1700 extra steps of pushing must not move it appreciably
        drift = float(np.hypot(pose_b[0] - pose_a[0], pose_b[1] - pose_a[1]))
        self.assertLess(drift, 0.05, f"pinned car drifted {drift:.3f} m over 1700 steps")

    def test_car_can_still_reverse_off_the_wall(self):
        """The rejection must not weld the car to the wall."""
        from f1tenth_gym.envs.env_config import TerminationConfig

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                simulation_config=SimulationConfig(max_laps=None),
                termination_config=TerminationConfig(terminate_on_collision=False),
                render_enabled=False,
            ),
        )
        env.reset(seed=42)
        sim = env.unwrapped.sim
        for _ in range(120):  # drive in and pin
            env.step(np.array([[0.41, 6.0]], dtype=np.float32))
        pinned = sim.state.standard_state[0].copy()
        for _ in range(150):  # now reverse
            obs, _r, _term, _t, info = env.step(np.array([[0.0, -2.0]], dtype=np.float32))
        freed = sim.state.standard_state[0].copy()
        env.close()
        backed_off = float(np.hypot(freed[0] - pinned[0], freed[1] - pinned[1]))
        self.assertGreater(backed_off, 0.2, "car could not reverse off the wall")
        self.assertEqual(float(info["collisions"][0]), 0.0, "still flagged after backing off")


class TestHaltRefreshesDerivedState(unittest.TestCase):
    """A rejected move must not survive in anything derived from the pose.

    `step` computes the Frenet frame before the collision pass, and
    `_update_scans` snapshots collision vertices before the per-agent loop, so
    all of it described the pose the halt undoes. Before this was fixed the
    published observation was internally inconsistent on every contact step:
    `frenet_pose` sat ~2 cm of arclength ahead of the `pose_x`/`pose_y` beside
    it, and the scan belonged to the rejected pose (23 m out on grazing beams).
    """

    def _crash(self, noise_std=0.0, num_agents=1):
        from f1tenth_gym.envs.env_config import TerminationConfig
        from f1tenth_gym.envs.lidar import LiDARConfig

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                num_agents=num_agents,
                simulation_config=SimulationConfig(max_laps=None),
                termination_config=TerminationConfig(terminate_on_collision=False),
                lidar_config=LiDARConfig(noise_std=noise_std),
                render_enabled=False,
            ),
        )
        obs, _ = env.reset(seed=42)
        action = np.array([[0.41, 6.0]] * num_agents, dtype=np.float32)
        for _ in range(400):
            obs, _, _, _, info = env.step(action)
            if float(info["collisions"][0]):
                return env, obs, info
        env.close()
        self.fail("no collision within 400 steps")

    def test_frenet_matches_the_published_pose(self):
        env, obs, _ = self._crash()
        sim = env.unwrapped.sim
        std = sim.state.standard_state[0]
        expected = np.array(
            env.unwrapped.track.cartesian_to_frenet(
                float(std[0]), float(std[1]), float(std[4])
            )
        )
        published = np.asarray(obs["agent_0"]["frenet_pose"], dtype=float)
        # float32 storage is the only slack; the stale value was ~2.1e-2 out
        self.assertLess(abs(published[0] - expected[0]), 1e-4)
        env.close()

    def test_scan_matches_the_published_pose(self):
        env, obs, _ = self._crash(noise_std=0.0)
        sim = env.unwrapped.sim
        scan_pose = sim._lidar_pose_from_base(sim.state.poses[0])
        expected = sim.scan_sims[0].scan(scan_pose, rng=None)
        published = np.asarray(obs["agent_0"]["scan"], dtype=float)
        self.assertLess(float(np.max(np.abs(published - expected))), 1e-4)
        env.close()

    def test_collision_vertices_match_the_published_pose(self):
        from f1tenth_gym.envs.collision_models import get_vertices

        env, _, _ = self._crash(num_agents=2)
        sim = env.unwrapped.sim
        cp = sim._collision_pose_from_base(sim.state.poses[0])
        expected = get_vertices(
            np.array([cp[0], cp[1], cp[2]], dtype=np.float64),
            sim.vehicle_params.length,
            sim.vehicle_params.width,
        )
        # stale snapshot was one rejected step of motion out (~2.5 cm at 6 m/s)
        self.assertLess(float(np.max(np.abs(sim._all_vertices[0] - expected))), 1e-6)
        env.close()


if __name__ == "__main__":
    unittest.main()
