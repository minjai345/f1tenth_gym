"""Independent GJK overlap oracle used only to validate production SAT."""

import numpy as np


def _triple_product(a, b, c):
    return b * a.dot(c) - a * b.dot(c)


def _support(vertices_a, vertices_b, direction):
    a = vertices_a[np.argmax(vertices_a.dot(direction))]
    b = vertices_b[np.argmax(vertices_b.dot(-direction))]
    return a - b


def collision(vertices_a, vertices_b):
    """Return whether two convex polygons overlap according to 2D GJK."""
    direction = np.mean(vertices_a, axis=0) - np.mean(vertices_b, axis=0)
    if np.all(direction == 0.0):
        direction = np.array([1.0, 0.0])
    simplex = [_support(vertices_a, vertices_b, direction)]
    if direction.dot(simplex[0]) <= 0.0:
        return False
    direction = -simplex[0]

    for _ in range(1000):
        point = _support(vertices_a, vertices_b, direction)
        if direction.dot(point) <= 0.0:
            return False
        simplex.append(point)
        origin = -point
        if len(simplex) == 2:
            edge = simplex[0] - point
            direction = _triple_product(edge, origin, edge)
            if np.linalg.norm(direction) < 1e-10:
                direction = np.array([edge[1], -edge[0]])
            continue

        b, c = simplex[1], simplex[0]
        ab, ac = b - point, c - point
        ac_perpendicular = _triple_product(ab, ac, ac)
        if ac_perpendicular.dot(origin) >= 0.0:
            simplex.pop(1)
            direction = ac_perpendicular
            continue
        ab_perpendicular = _triple_product(ac, ab, ab)
        if ab_perpendicular.dot(origin) < 0.0:
            return True
        simplex.pop(0)
        direction = ab_perpendicular
    return False
