"""The contact kernels are the template for the JAX migration; keep them portable.

`kernels` and `solver` are meant to survive the rewrite unchanged: pure JAX, fixed
shapes, `vmap`-ready, and free of any gym import. That is a property nothing else
enforces, so a stray convenience import would go unnoticed until the seam is cut.
"""

import ast
import importlib.util
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ENVS = pathlib.Path(__file__).resolve().parents[1] / "f1tenth_gym" / "envs"
CONTACT = ENVS / "contact"
PORTABLE = (CONTACT / "kernels.py", CONTACT / "solver.py", ENVS / "lidar" / "kernels.py")
ALLOWED_ROOTS = {"jax", "numpy", "typing", "math", "dataclasses", "functools"}


def imported_roots(path):
    """Every top-level package a module imports, including inside functions."""
    tree = ast.parse(path.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add("." * node.level + (node.module or ""))
            elif node.module:
                roots.add(node.module.split(".")[0])
    return roots


class TestPortability(unittest.TestCase):
    def test_the_portable_modules_import_nothing_local(self):
        for path in PORTABLE:
            roots = imported_roots(path)
            stray = {r for r in roots if r.startswith(".") or r not in ALLOWED_ROOTS}
            self.assertEqual(stray, set(), f"{path.name} imports {sorted(stray)}")

    def test_they_load_with_no_package_at_all(self):
        """Loaded straight off disk, outside f1tenth_gym, they still work."""
        for path in PORTABLE:
            spec = importlib.util.spec_from_file_location(
                f"detached_{path.parent.name}_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(hasattr(module, "__name__"))

    def test_the_adapter_is_the_only_module_that_knows_numpy_marshalling(self):
        """Physics belongs in kernels/solver, or cutting the seam stops being a delete."""
        for path in PORTABLE:
            self.assertNotIn("numpy", imported_roots(path), path.name)


class TestKernelsAreBatchable(unittest.TestCase):
    """Fixed shapes and no data-dependent control flow, so `vmap` and `jit` apply."""

    def setUp(self):
        from f1tenth_gym.envs.collision_models import get_vertices

        rng = np.random.default_rng(0)
        poses = rng.uniform(-1.0, 1.0, (8, 3))
        self.verts = jnp.asarray(
            np.stack([get_vertices(p, 0.58, 0.31) for p in poses]), jnp.float32)
        self.other = jnp.asarray(
            np.stack([get_vertices(p + 0.2, 0.58, 0.31) for p in poses]), jnp.float32)
        self.a = jnp.asarray(rng.uniform(-1, 1, (8, 2)), jnp.float32)
        angle = rng.uniform(-np.pi, np.pi, 8)
        tangent = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        self.b = self.a + jnp.asarray(tangent, jnp.float32)
        self.n = jnp.asarray(np.stack([-tangent[:, 1], tangent[:, 0]], axis=1), jnp.float32)

    def test_segment_contact_vmaps_and_jits(self):
        from f1tenth_gym.envs.contact import segment_contact

        run = jax.jit(jax.vmap(segment_contact))
        out = run(self.verts, self.a, self.b, self.n)
        self.assertEqual(out.depths.shape, (8, 2))
        self.assertEqual(out.points.shape, (8, 2, 2))

    def test_body_contact_vmaps_and_jits(self):
        from f1tenth_gym.envs.contact import body_contact

        out = jax.jit(jax.vmap(body_contact))(self.verts, self.other)
        self.assertEqual(out.depths.shape, (8, 2))

    def test_the_surrogate_vmaps_jits_and_differentiates(self):
        from f1tenth_gym.envs.contact import deepest_depth

        grad = jax.jit(jax.vmap(jax.grad(deepest_depth)))
        out = grad(self.verts, self.a, self.b, self.n)
        self.assertEqual(out.shape, self.verts.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

    def test_the_scan_kernel_vmaps_jits_and_differentiates(self):
        from f1tenth_gym.envs.lidar.kernels import scan

        corner = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
        seg_a = jnp.asarray(corner)
        seg_b = jnp.asarray(np.roll(corner, -1, axis=0))
        angles = jnp.linspace(-1.0, 1.0, 16)
        run = jax.jit(jax.vmap(lambda p: scan(p, angles, seg_a, seg_b, 30.0)))
        self.assertEqual(run(jnp.zeros((8, 3))).shape, (8, 16))
        grad = jax.jit(jax.grad(lambda p: jnp.sum(scan(p, angles, seg_a, seg_b, 30.0))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad(jnp.zeros(3))))))

    def test_the_solver_vmaps_and_jits(self):
        from f1tenth_gym.envs.contact import ContactParams, resolve

        params = ContactParams(0.0, 0.6, 0.6, 0.4, 0.002)
        points = jnp.zeros((8, 4, 2), jnp.float32)
        depths = jnp.full((8, 4), 0.01, jnp.float32)
        normals = jnp.tile(jnp.array([[0.0, 1.0]], jnp.float32), (8, 4, 1))
        run = jax.jit(jax.vmap(
            lambda v, w, p, d, n, c: resolve(
                v, w, jnp.float32(3.74), jnp.float32(0.047), p, d, n, c, params, 8)))
        velocity, omega, correction = run(
            jnp.zeros((8, 2), jnp.float32), jnp.zeros(8, jnp.float32),
            points, depths, normals, jnp.zeros((8, 2), jnp.float32))
        self.assertEqual(velocity.shape, (8, 2))
        self.assertEqual(omega.shape, (8,))
        self.assertEqual(correction.shape, (8, 2))


class TestWallsCarryOrientation(unittest.TestCase):
    """Plan Phase 9: a shared wall array must carry outward normals.

    A scan backend needs only segment endpoints, so a wall array built for scans
    would drop the normals -- and segment-segment intersection alone returns a
    boolean with no side, which contact cannot use.
    """

    def test_wall_segments_expose_an_outward_normal_per_segment(self):
        from f1tenth_gym.envs.track import Track
        from f1tenth_gym.envs.track.walls import wall_segments

        walls = wall_segments(Track.from_track_name("Spielberg", 1.0))
        self.assertEqual(walls.n.shape, walls.a.shape)
        np.testing.assert_allclose(np.linalg.norm(walls.n, axis=1), 1.0, atol=1e-5)

    def test_a_manifold_without_a_normal_cannot_be_built(self):
        """The kernel signature requires it, so the coupling is not accidental."""
        import inspect

        from f1tenth_gym.envs.contact import segment_contact

        self.assertIn("normal", inspect.signature(segment_contact).parameters)


if __name__ == "__main__":
    unittest.main()
