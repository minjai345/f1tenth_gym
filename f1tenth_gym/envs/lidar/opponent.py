# MIT License
#
# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Numba geometry for opponent-body occlusion of host LiDAR scans."""

from __future__ import annotations

from numba import njit
import numpy as np


@njit(cache=True)
def cross(v1, v2):
    """Return the planar cross product of two two-vectors."""
    return v1[0] * v2[1] - v1[1] * v2[0]


@njit(cache=True)
def are_collinear(pt_a, pt_b, pt_c):
    """Return whether three planar points are collinear within tolerance."""
    tol = 1e-8
    ba = pt_b - pt_a
    ca = pt_a - pt_c
    return np.fabs(cross(ba, ca)) < tol


@njit(cache=True)
def get_range(pose, beam_theta, va, vb):
    """Return the ray distance from ``pose`` to the edge ``va``--``vb``."""
    origin = pose[0:2]
    origin_to_a = origin - va
    edge = vb - va
    perpendicular = np.array(
        [np.cos(beam_theta + np.pi / 2.0),
         np.sin(beam_theta + np.pi / 2.0)]
    )

    denominator = edge.dot(perpendicular)
    distance = np.inf
    if np.fabs(denominator) > 0.0:
        along_ray = cross(edge, origin_to_a) / denominator
        along_edge = origin_to_a.dot(perpendicular) / denominator
        if along_ray >= 0.0 and 0.0 <= along_edge <= 1.0:
            distance = along_ray
    elif are_collinear(origin, va, vb):
        distance = min(np.linalg.norm(va - origin), np.linalg.norm(vb - origin))
    return distance


@njit(cache=True)
def _nearest_beam(scan_angles, angle):
    """Return the index of the beam closest to ``angle``."""
    best = 0
    best_gap = np.abs(scan_angles[0] - angle)
    for index in range(1, scan_angles.shape[0]):
        gap = np.abs(scan_angles[index] - angle)
        if gap < best_gap:
            best_gap = gap
            best = index
    return best


@njit(cache=True)
def _beam_range(scan_angles, low_angle, high_angle):
    """Return inclusive beam indexes for an interval, or ``(1, 0)`` if empty."""
    low = _nearest_beam(scan_angles, low_angle)
    high = _nearest_beam(scan_angles, high_angle)
    if high < low:
        return 1, 0
    return low, high


@njit(cache=True)
def get_blocked_view_ranges(pose, vertices, scan_angles):
    """Return at most two beam ranges an opponent body could occlude.

    Min/max of corner bearings collapses to every beam when a body straddles
    the rear of the scan. Instead, the body's circular arc is the complement
    of the widest gap between its corner bearings, intersected with the sensor
    field of view.

    Each returned pair is inclusive. A range is empty when its high bound is
    below its low bound. The second range is live only when the body spans both
    ends of the scan.
    """
    num_beams = scan_angles.shape[0]
    bearings = np.empty(4)
    for index in range(4):
        angle = (
            np.arctan2(
                vertices[index, 1] - pose[1],
                vertices[index, 0] - pose[0],
            )
            - pose[2]
        )
        if angle > np.pi:
            angle -= 2.0 * np.pi
        elif angle <= -np.pi:
            angle += 2.0 * np.pi
        bearings[index] = angle
    bearings.sort()

    widest = -1.0
    at = 0
    for index in range(4):
        upper = (
            bearings[0] + 2.0 * np.pi
            if index == 3
            else bearings[index + 1]
        )
        gap = upper - bearings[index]
        if gap > widest:
            widest = gap
            at = index

    arc_low = bearings[(at + 1) % 4]
    arc_high = bearings[at]
    wrapped = at != 3
    fov_low = scan_angles[0]
    fov_high = scan_angles[num_beams - 1]

    if not wrapped:
        low = arc_low if arc_low > fov_low else fov_low
        high = arc_high if arc_high < fov_high else fov_high
        if low > high:
            return 1, 0, 1, 0
        return _beam_range(scan_angles, low, high) + (1, 0)

    upper_range = (1, 0)
    lower_range = (1, 0)
    if arc_low <= fov_high:
        low = arc_low if arc_low > fov_low else fov_low
        upper_range = _beam_range(scan_angles, low, fov_high)
    if arc_high >= fov_low:
        high = arc_high if arc_high < fov_high else fov_high
        lower_range = _beam_range(scan_angles, fov_low, high)
    return (
        lower_range[0],
        lower_range[1],
        upper_range[0],
        upper_range[1],
    )


@njit(cache=True)
def ray_cast(pose, scan, scan_angles, vertices):
    """Shorten scan beams that intersect another agent's body.

    ``scan`` is mutated in place and returned. The caller must pass four body
    vertices in cyclic order and run any logic that needs the wall-only scan
    before calling this function.
    """
    looped_vertices = np.empty((5, 2))
    looped_vertices[0:4, :] = vertices
    looped_vertices[4, :] = vertices[0, :]

    low_a, high_a, low_b, high_b = get_blocked_view_ranges(
        pose, vertices, scan_angles
    )
    for low, high in ((low_a, high_a), (low_b, high_b)):
        for beam_index in range(low, high + 1):
            for edge_index in range(4):
                scan_range = get_range(
                    pose,
                    pose[2] + scan_angles[beam_index],
                    looped_vertices[edge_index, :],
                    looped_vertices[edge_index + 1, :],
                )
                if scan_range < scan[beam_index]:
                    scan[beam_index] = scan_range
    return scan


__all__ = ["get_blocked_view_ranges", "get_range", "ray_cast"]
