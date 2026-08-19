import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.collision_models import get_vertices
from f1tenth_gym.envs.contact import Manifold, segment_contact, speculative_gap
from f1tenth_gym.envs.contact.kernels import NO_CONTACT_GAP

CAR_L, CAR_W = 0.58, 0.31
CIRCUMRADIUS = float(np.hypot(CAR_L, CAR_W) / 2)


def body(x, y, yaw):
    return jnp.asarray(get_vertices(np.array([x, y, yaw]), CAR_L, CAR_W), jnp.float32)


def wall(ax, ay, bx, by):
    """A segment with a normal perpendicular by construction, as walls.py emits."""
    a, b = np.array([ax, ay]), np.array([bx, by])
    edge = b - a
    n = np.array([-edge[1], edge[0]])
    n = n / np.linalg.norm(n)
    return jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32), jnp.asarray(n, jnp.float32)


def segment_hits_rect(verts, a, b):
    """Cyrus-Beck clip of the segment against the rect: the ground-truth oracle."""
    direction = b - a
    t0, t1 = 0.0, 1.0
    centre = verts.mean(axis=0)
    for k in range(4):
        p0, p1 = verts[k], verts[(k + 1) % 4]
        edge = p1 - p0
        n = np.array([edge[1], -edge[0]])
        if n @ (centre - p0) > 0:
            n = -n
        den, num = n @ direction, n @ (p0 - a)
        if abs(den) < 1e-15:
            if num < 0:
                return False
            continue
        t = num / den
        if den > 0:
            t1 = min(t1, t)
        else:
            t0 = max(t0, t)
        if t0 > t1:
            return False
    return True


def without_body_axes(verts, a, b, n):
    """The kernel minus its two body-axis gates: a box test in the contact frame."""
    along = verts @ n
    plane = a @ n
    ok = (plane - along.min() > 0) & (along.max() > plane)
    t = jnp.array([-n[1], n[0]])
    proj = verts @ t
    lo = jnp.minimum(a @ t, b @ t)
    hi = jnp.maximum(a @ t, b @ t)
    return bool(ok & (proj.min() <= hi) & (proj.max() >= lo))


class TestManifoldShape(unittest.TestCase):
    """3.5: one point cannot resist rotation about itself."""

    def test_a_flat_face_gives_two_points(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        manifold = segment_contact(body(0.0, CAR_W / 2 - 0.01, 0.0), a, b, n)
        self.assertEqual(int(manifold.count), 2)
        self.assertTrue(np.allclose(np.asarray(manifold.depths), 0.01, atol=1e-4))

    def test_a_corner_gives_one_point(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        manifold = segment_contact(body(0.0, CAR_W / 2 - 0.01, np.pi / 4), a, b, n)
        self.assertEqual(int(manifold.count), 1)

    def test_contact_points_lie_on_the_body(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        for yaw in np.linspace(0.0, np.pi, 12):
            manifold = segment_contact(body(0.0, CAR_W / 2 - 0.02, float(yaw)), a, b, n)
            live = np.asarray(manifold.depths) > 0
            if not live.any():
                continue
            radius = np.linalg.norm(np.asarray(manifold.points)[live], axis=1)
            self.assertLessEqual(radius.max(), CIRCUMRADIUS + 1e-6)

    def test_the_normal_is_carried_only_when_live(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        hit = segment_contact(body(0.0, CAR_W / 2 - 0.01, 0.0), a, b, n)
        clear = segment_contact(body(0.0, 5.0, 0.0), a, b, n)
        self.assertTrue(np.allclose(np.asarray(hit.normal), np.asarray(n)))
        self.assertTrue(np.allclose(np.asarray(clear.normal), 0.0))


class TestOneSidedFace(unittest.TestCase):
    """3.2: the far side of a two-pixel strip must refuse a body on the near side."""

    def test_the_far_face_of_a_strip_reports_nothing(self):
        strip = 0.116
        near = wall(-2.0, 0.0, 2.0, 0.0)
        far = wall(2.0, -strip, -2.0, -strip)
        resting = body(0.0, CAR_W / 2 - 0.01, 0.0)
        self.assertEqual(int(segment_contact(resting, *near).count), 2)
        self.assertEqual(int(segment_contact(resting, *far).count), 0)


class TestBodyAxesAreRequired(unittest.TestCase):
    """3.3: without them the test is a bounding box in the contact frame."""

    SEGMENTS = (
        wall(-0.05, 0.0, 0.05, 0.0),
        wall(-0.10, -0.06, 0.10, 0.06),
        wall(0.0, -0.05, 0.0, 0.05),
    )

    def test_the_kernel_never_invents_a_contact(self):
        rng = np.random.default_rng(0)
        false_positives = 0
        for _ in range(400):
            x, y = rng.uniform(-0.55, 0.55, 2)
            verts = get_vertices(np.array([x, y, rng.uniform(-np.pi, np.pi)]), CAR_L, CAR_W)
            for a, b, n in self.SEGMENTS:
                got = float(np.asarray(
                    segment_contact(jnp.asarray(verts, jnp.float32), a, b, n).depths
                ).max()) > 0
                if got and not segment_hits_rect(verts, np.asarray(a), np.asarray(b)):
                    false_positives += 1
        self.assertEqual(false_positives, 0)

    def test_dropping_them_invents_many(self):
        rng = np.random.default_rng(0)
        false_positives = 0
        for _ in range(400):
            x, y = rng.uniform(-0.55, 0.55, 2)
            verts = get_vertices(np.array([x, y, rng.uniform(-np.pi, np.pi)]), CAR_L, CAR_W)
            for a, b, n in self.SEGMENTS:
                naive = without_body_axes(jnp.asarray(verts, jnp.float32), a, b, n)
                if naive and not segment_hits_rect(verts, np.asarray(a), np.asarray(b)):
                    false_positives += 1
        self.assertGreater(false_positives, 40)


class TestStrictSeparation(unittest.TestCase):
    """3.4: zero overlap on an axis is touching, not separated."""

    def test_a_flat_face_at_one_centimetre_still_contacts(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        for depth in (0.01, 1e-4, 1e-6):
            manifold = segment_contact(body(0.0, CAR_W / 2 - depth, 0.0), a, b, n)
            self.assertEqual(int(manifold.count), 2, f"depth {depth}")


class TestSpeculativeGap(unittest.TestCase):
    """3.7: the clamp scales the normal component, so it needs a normal-direction gap."""

    def test_the_gap_is_signed_and_metric(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        for centre_y, expected in ((0.30, 0.145), (0.20, 0.045), (0.15, -0.005)):
            got = float(speculative_gap(body(0.0, centre_y, 0.0), a, b, n))
            self.assertAlmostEqual(got, expected, places=4)

    def test_an_unreachable_segment_cannot_trigger_a_clamp(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        self.assertEqual(float(speculative_gap(body(9.0, 0.2, 0.0), a, b, n)), NO_CONTACT_GAP)

    def test_a_padded_slot_cannot_trigger_a_clamp(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        gap = speculative_gap(body(0.0, 0.15, 0.0), a, b, n, valid=False)
        self.assertEqual(float(gap), NO_CONTACT_GAP)


class TestJaxContract(unittest.TestCase):
    def test_jit_and_vmap_over_padded_candidates(self):
        a0, b0, n0 = wall(-2.0, 0.0, 2.0, 0.0)
        a1, b1, n1 = wall(-2.0, 1.0, 2.0, 1.0)
        A, B, N = jnp.stack([a0, a1]), jnp.stack([b0, b1]), jnp.stack([n0, n1])
        valid = jnp.array([True, False])
        verts = body(0.0, CAR_W / 2 - 0.01, 0.0)
        run = jax.jit(jax.vmap(lambda a, b, n, ok: segment_contact(verts, a, b, n, ok)))
        out = run(A, B, N, valid)
        self.assertIsInstance(out, Manifold)
        self.assertEqual(out.depths.shape, (2, 2))
        self.assertGreater(float(out.depths[0].max()), 0.0)
        self.assertEqual(float(out.depths[1].max()), 0.0)

    def test_an_invalid_slot_is_empty(self):
        a, b, n = wall(-2.0, 0.0, 2.0, 0.0)
        manifold = segment_contact(body(0.0, CAR_W / 2 - 0.01, 0.0), a, b, n, valid=False)
        self.assertEqual(int(manifold.count), 0)
        self.assertTrue(np.allclose(np.asarray(manifold.points), 0.0))


class TestContinuousWalls(unittest.TestCase):
    """A single segment may decline an overlap; a chain of them may not."""

    def test_every_overlapping_pose_produces_a_contact(self):
        from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as params
        from f1tenth_gym.envs.track import Track
        from f1tenth_gym.envs.track.accel import build_for_track, gather
        from f1tenth_gym.envs.track.walls import _occupied_at

        track = Track.from_track_name("Monza", 1.0)
        walls, _budget, index = build_for_track(track, params)
        A, B, N = jnp.asarray(walls.a), jnp.asarray(walls.b), jnp.asarray(walls.n)
        occupied = track.occupancy_map == 0.0
        res = float(track.spec.resolution)
        origin = tuple(float(v) for v in track.spec.origin[:3])

        @jax.jit
        def manifolds(verts, cand):
            ok = cand >= 0
            i = jnp.where(ok, cand, 0)
            return jax.vmap(lambda k, o: segment_contact(verts, A[k], B[k], N[k], o))(i, ok)

        rng = np.random.default_rng(0)
        mid = 0.5 * (walls.a + walls.b)
        grid = np.stack(
            np.meshgrid(np.linspace(-0.5, 0.5, 11) * CAR_L,
                        np.linspace(-0.5, 0.5, 11) * CAR_W), axis=-1
        ).reshape(-1, 2)
        overlapping = missed = 0
        for k in rng.choice(len(mid), 120, replace=False):
            for push in (0.02, 0.08):
                centre = mid[k] - walls.n[k] * (CAR_W / 2 - push)
                yaw = float(rng.uniform(-np.pi, np.pi))
                verts = get_vertices(np.array([centre[0], centre[1], yaw]), CAR_L, CAR_W)
                axis = verts[3] - verts[0]
                axis = axis / np.linalg.norm(axis)
                rot = np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])
                if not _occupied_at(grid @ rot.T + verts.mean(0), occupied, res, origin).any():
                    continue
                overlapping += 1
                out = manifolds(jnp.asarray(verts, jnp.float32),
                                jnp.asarray(gather(centre[None], index)[0]))
                if int((np.asarray(out.depths) > 0).sum()) == 0:
                    missed += 1
        self.assertGreater(overlapping, 30, "scenario produced too few overlaps")
        self.assertEqual(missed, 0)


if __name__ == "__main__":
    unittest.main()
