"""
Layer 2 - Geometry Engine.

Pure mathematical representations independent of any topology. Everything that
needs linear algebra (transforms, surface evaluation) routes through numpy so the
topology layer never touches matrix math directly.

Conventions
    * A *point* is a location in space (Point3D).
    * A *vector* reuses Point3D but is interpreted as a direction/displacement.
    * Angles passed to rotations are in degrees (CLI friendly); converted to
      radians internally.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Points & vectors
# --------------------------------------------------------------------------- #
class Point3D:
    """A 3D location. Doubles as a direction vector when used as one."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    # --- conversions ------------------------------------------------------- #
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_array(cls, a: Sequence[float]) -> "Point3D":
        return cls(a[0], a[1], a[2])

    # --- vector algebra ---------------------------------------------------- #
    def __add__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Point3D":
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def dot(self, other: "Point3D") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Point3D") -> "Point3D":
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Point3D":
        n = self.length()
        if n < 1e-12:
            raise ValueError("cannot normalize a zero-length vector")
        return Point3D(self.x / n, self.y / n, self.z / n)

    def is_close(self, other: "Point3D", tol: float = 1e-9) -> bool:
        return (self - other).length() <= tol

    def __repr__(self) -> str:
        return f"({self.x:.4g}, {self.y:.4g}, {self.z:.4g})"


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
class Bezier:
    """A Bezier curve defined by its control points (degree = n - 1)."""

    def __init__(self, control_points: Sequence[Point3D]):
        if len(control_points) < 2:
            raise ValueError("a Bezier curve needs at least 2 control points")
        self.control_points: List[Point3D] = list(control_points)

    @property
    def degree(self) -> int:
        return len(self.control_points) - 1

    def evaluate(self, t: float) -> Point3D:
        """Evaluate the curve at parameter t in [0, 1] (De Casteljau)."""
        pts = [p for p in self.control_points]
        while len(pts) > 1:
            pts = [pts[i] * (1 - t) + pts[i + 1] * t for i in range(len(pts) - 1)]
        return pts[0]

    def split(self, t: float) -> "tuple[Bezier, Bezier]":
        """De Casteljau subdivision: returns the two sub-curves at parameter t."""
        left: List[Point3D] = []
        right: List[Point3D] = []
        pts = [p for p in self.control_points]
        left.append(pts[0])
        right.append(pts[-1])
        while len(pts) > 1:
            pts = [pts[i] * (1 - t) + pts[i + 1] * t for i in range(len(pts) - 1)]
            left.append(pts[0])
            right.append(pts[-1])
        right.reverse()
        return Bezier(left), Bezier(right)


# --------------------------------------------------------------------------- #
# Surfaces
# --------------------------------------------------------------------------- #
def _bspline_basis(i: int, k: int, u: float, knots: Sequence[float]) -> float:
    """Cox-de-Boor recursion for the i-th B-spline basis of degree k."""
    if k == 0:
        return 1.0 if (knots[i] <= u < knots[i + 1]) else 0.0
    left = 0.0
    denom_l = knots[i + k] - knots[i]
    if denom_l > 1e-12:
        left = (u - knots[i]) / denom_l * _bspline_basis(i, k - 1, u, knots)
    right = 0.0
    denom_r = knots[i + k + 1] - knots[i + 1]
    if denom_r > 1e-12:
        right = (knots[i + k + 1] - u) / denom_r * _bspline_basis(i + 1, k - 1, u, knots)
    return left + right


class NURBSSurface:
    """
    A NURBS surface over a rectangular control net.

    If knot vectors are not supplied a uniform clamped knot vector is generated,
    which (with unit weights) yields a Bezier/B-spline surface. The domain is
    normalized to (u, v) in [0, 1].
    """

    def __init__(
        self,
        control_net: Sequence[Sequence[Point3D]],
        degree_u: int,
        degree_v: int,
        weights: Sequence[Sequence[float]] | None = None,
        knots_u: Sequence[float] | None = None,
        knots_v: Sequence[float] | None = None,
    ):
        self.control_net = [list(row) for row in control_net]
        self.n_u = len(self.control_net)
        self.n_v = len(self.control_net[0]) if self.n_u else 0
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.weights = (
            [list(row) for row in weights]
            if weights is not None
            else [[1.0] * self.n_v for _ in range(self.n_u)]
        )
        self.knots_u = list(knots_u) if knots_u else self._clamped_knots(self.n_u, degree_u)
        self.knots_v = list(knots_v) if knots_v else self._clamped_knots(self.n_v, degree_v)

    @staticmethod
    def _clamped_knots(n_ctrl: int, degree: int) -> List[float]:
        """Uniform clamped knot vector normalized to [0, 1]."""
        m = n_ctrl + degree + 1
        knots = [0.0] * m
        interior = n_ctrl - degree - 1
        for j in range(1, interior + 1):
            knots[degree + j] = j / (interior + 1)
        for j in range(n_ctrl, m):
            knots[j] = 1.0
        return knots

    def evaluate(self, u: float, v: float) -> Point3D:
        """Evaluate the rational surface point S(u, v)."""
        # Clamp slightly below 1.0 so the half-open basis support includes the edge.
        u = min(max(u, 0.0), 1.0 - 1e-9)
        v = min(max(v, 0.0), 1.0 - 1e-9)
        numerator = np.zeros(3)
        denominator = 0.0
        for i in range(self.n_u):
            bu = _bspline_basis(i, self.degree_u, u, self.knots_u)
            if bu == 0.0:
                continue
            for j in range(self.n_v):
                bv = _bspline_basis(j, self.degree_v, v, self.knots_v)
                if bv == 0.0:
                    continue
                w = self.weights[i][j]
                factor = bu * bv * w
                numerator += factor * self.control_net[i][j].to_array()
                denominator += factor
        if denominator < 1e-12:
            return Point3D(0, 0, 0)
        return Point3D.from_array(numerator / denominator)

    # ------------------------------------------------------------------ #
    # Surface subdivision (De Casteljau, degree 1 or 2 Bézier only)
    # ------------------------------------------------------------------ #
    def _casteljau_row(self, row: List["Point3D"], t: float) -> "tuple[list, list]":
        """Split a degree-1 or degree-2 Bézier row at t; return (left, right)."""
        n = len(row)
        if n == 2:          # degree 1 (linear)
            m = row[0] * (1.0 - t) + row[1] * t
            return [row[0], m], [m, row[1]]
        if n == 3:          # degree 2 (quadratic)
            q0 = row[0] * (1.0 - t) + row[1] * t
            q1 = row[1] * (1.0 - t) + row[2] * t
            r0 = q0 * (1.0 - t) + q1 * t
            return [row[0], q0, r0], [r0, q1, row[2]]
        raise NotImplementedError(
            f"NURBSSurface.split: only degree 1/2 supported (got {n - 1})")

    def split_u(self, t: float) -> "tuple[NURBSSurface, NURBSSurface]":
        """
        Split the surface at parameter *t* along the **u** direction using
        De Casteljau subdivision on each row of the control net.

        Returns ``(left_surface, right_surface)`` where left covers u ∈ [0, t]
        and right covers u ∈ [t, 1], both reparameterised to [0, 1].

        Only supported for degree-1 or degree-2 Bézier control nets.
        """
        left_net, right_net = [], []
        for row in self.control_net:
            l_row, r_row = self._casteljau_row(row, t)
            left_net.append(l_row)
            right_net.append(r_row)
        return (
            NURBSSurface(left_net, self.degree_u, self.degree_v),
            NURBSSurface(right_net, self.degree_u, self.degree_v),
        )

    def split_v(self, t: float) -> "tuple[NURBSSurface, NURBSSurface]":
        """
        Split the surface at parameter *t* along the **v** direction.

        Returns ``(low_surface, high_surface)`` where low covers v ∈ [0, t]
        and high covers v ∈ [t, 1].
        """
        # Transpose: treat each column as a row, split, transpose back
        n_v = self.n_v
        cols = [[self.control_net[i][j] for i in range(self.n_u)] for j in range(n_v)]
        low_cols, high_cols = [], []
        for col in cols:
            l_col, h_col = self._casteljau_row(col, t)
            low_cols.append(l_col)
            high_cols.append(h_col)
        # Transpose back: rows become the n_u direction
        def transpose(cols_list):
            n_r = len(cols_list[0])
            return [[cols_list[j][i] for j in range(len(cols_list))] for i in range(n_r)]
        return (
            NURBSSurface(transpose(low_cols), self.degree_u, self.degree_v),
            NURBSSurface(transpose(high_cols), self.degree_u, self.degree_v),
        )


# --------------------------------------------------------------------------- #
# Trim plane (half-space cutter)
# --------------------------------------------------------------------------- #
class TrimPlane:
    """
    An infinite cutting plane in Hessian normal form:  normal · P = d

    Convention: points where ``normal · P - d > 0`` are on the **keep** side.
    The normal is automatically normalised on construction so that
    ``signed_distance`` returns a true Euclidean distance.
    """

    def __init__(self, normal: "Point3D", d: float):
        n_len = normal.length()
        if n_len < 1e-12:
            raise ValueError("TrimPlane normal must be non-zero")
        self.normal = Point3D(normal.x / n_len, normal.y / n_len, normal.z / n_len)
        self.d = d / n_len

    # -- geometry ----------------------------------------------------------- #
    def signed_distance(self, p: "Point3D") -> float:
        """Signed distance from *p* to the plane (positive = keep side)."""
        return self.normal.dot(p) - self.d

    def intersect_segment(
        self, p0: "Point3D", p1: "Point3D"
    ) -> "tuple[float, Point3D] | None":
        """
        Intersect the segment p0..p1 with the plane.

        Returns ``(t, intersection_point)`` where ``t`` in (0, 1) is the
        parameter along the segment, or ``None`` if the segment does not cross
        the plane (both endpoints on the same side or both on the plane).
        """
        d0 = self.signed_distance(p0)
        d1 = self.signed_distance(p1)
        if d0 * d1 >= 0.0:
            return None
        t = d0 / (d0 - d1)
        pt = p0 * (1.0 - t) + p1 * t
        return t, pt

    def __repr__(self) -> str:
        n = self.normal
        return f"TrimPlane(n=({n.x:.4g},{n.y:.4g},{n.z:.4g}), d={self.d:.4g})"


# --------------------------------------------------------------------------- #
# Geometric intersection helpers (curve / surface  ∩  plane)
#
# These are what make a trim *geometry-aware*: instead of cutting the straight
# chord between two vertices or the flat polygon of a face, we intersect the real
# geometric support (a Bezier curve, a NURBS surface) with the cutting plane.
# --------------------------------------------------------------------------- #
def bezier_plane_param(curve: "Bezier", plane: "TrimPlane",
                       samples: int = 64, tol: float = 1e-10):
    """
    Find a parameter ``t`` in (0, 1) where ``curve`` crosses ``plane``.

    Scans ``samples`` sub-intervals for a sign change of the signed distance,
    then bisects that bracket. Returns ``t`` (the first crossing found), or
    ``None`` when the curve stays on one side. This is the curve analogue of
    :meth:`TrimPlane.intersect_segment`, but evaluated on the *actual* curve
    rather than its end-point chord.
    """
    def sd(t: float) -> float:
        return plane.signed_distance(curve.evaluate(t))

    prev_t = 0.0
    prev_d = sd(0.0)
    for i in range(1, samples + 1):
        cur_t = i / samples
        cur_d = sd(cur_t)
        if prev_d == 0.0 and 0.0 < prev_t < 1.0:
            return prev_t
        if prev_d * cur_d < 0.0:                 # bracketed a crossing
            lo, hi = prev_t, cur_t
            d_lo = prev_d
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                d_mid = sd(mid)
                if abs(d_mid) < tol:
                    return mid
                if d_lo * d_mid < 0.0:
                    hi = mid
                else:
                    lo, d_lo = mid, d_mid
            return 0.5 * (lo + hi)
        prev_t, prev_d = cur_t, cur_d
    return None


def _clip_triangle_keep(tri, plane: "TrimPlane", tol: float = 1e-9):
    """
    Clip a triangle (three :class:`Point3D`) against ``plane``, keeping the part
    on the +side (``signed_distance >= 0``).

    Returns a list of triangles (each a 3-tuple of Point3D): empty if the whole
    triangle is on the discard side, ``[tri]`` if entirely kept, otherwise the
    fan-triangulation of the clipped polygon (Sutherland–Hodgman for one plane).
    """
    ds = [plane.signed_distance(p) for p in tri]
    if all(d >= -tol for d in ds):
        return [tuple(tri)]
    if all(d <= tol for d in ds):
        return []
    poly: List[Point3D] = []
    for i in range(3):
        a, da = tri[i], ds[i]
        b, db = tri[(i + 1) % 3], ds[(i + 1) % 3]
        if da >= -tol:
            poly.append(a)
        if (da > tol and db < -tol) or (da < -tol and db > tol):
            t = da / (da - db)
            poly.append(a * (1.0 - t) + b * t)
    if len(poly) < 3:
        return []
    return [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]


def surface_grid(surf: "NURBSSurface", nu: int = 16, nv: int = 16):
    """Evaluate ``surf`` on a ``(nu+1)×(nv+1)`` grid of Point3D (row-major u)."""
    return [[surf.evaluate(i / nu, j / nv) for j in range(nv + 1)]
            for i in range(nu + 1)]


def surface_plane_side(surf: "NURBSSurface", plane: "TrimPlane",
                       nu: int = 12, nv: int = 12, tol: float = 1e-9):
    """
    Sample ``surf`` on a grid and report ``(has_pos, has_neg)`` — whether the
    surface reaches the +side and/or the −side of ``plane``.

    ``has_pos and has_neg`` means the surface genuinely straddles the plane
    (an interior/curved cut), even when every *boundary vertex* sits on one side
    — the case a polygon-only classifier misses (e.g. a dome cap).
    """
    has_pos = has_neg = False
    for row in surface_grid(surf, nu, nv):
        for p in row:
            d = plane.signed_distance(p)
            if d > tol:
                has_pos = True
            elif d < -tol:
                has_neg = True
            if has_pos and has_neg:
                return True, True
    return has_pos, has_neg


def line_plane_intersect(origin: "Point3D", direction: "Point3D",
                         plane: "TrimPlane", tol: float = 1e-12):
    """
    Intersect the ray/line ``origin + t·direction`` with ``plane``.

    Returns ``(t, point)`` or ``None`` when the line is parallel to the plane.
    ``t`` is signed; callers that mean a forward *ray* should require ``t > 0``.
    This is the closed-form line–plane intersection used to extend a curve or a
    swept vertex onto a planar target.
    """
    denom = plane.normal.dot(direction)
    if abs(denom) < tol:
        return None
    t = (plane.d - plane.normal.dot(origin)) / denom
    return t, origin + direction * t


def ray_triangle_intersect(origin: "Point3D", direction: "Point3D",
                           a: "Point3D", b: "Point3D", c: "Point3D",
                           tol: float = 1e-9):
    """
    Möller–Trumbore ray/triangle intersection (two-sided).

    Returns the forward hit distance ``t > 0`` or ``None``. Used to intersect a
    ray with a tessellated NURBS surface.
    """
    e1 = b - a
    e2 = c - a
    pvec = direction.cross(e2)
    det = e1.dot(pvec)
    if abs(det) < tol:
        return None
    inv = 1.0 / det
    tvec = origin - a
    u = tvec.dot(pvec) * inv
    if u < -tol or u > 1.0 + tol:
        return None
    qvec = tvec.cross(e1)
    v = direction.dot(qvec) * inv
    if v < -tol or u + v > 1.0 + tol:
        return None
    t = e2.dot(qvec) * inv
    return t if t > tol else None


def ray_surface_intersect(origin: "Point3D", direction: "Point3D",
                          surf: "NURBSSurface", nu: int = 24, nv: int = 24):
    """
    Intersect a forward ray with a NURBS surface via its tessellation.

    Tessellates the surface into a triangle grid and returns the nearest
    forward hit :class:`Point3D` (smallest ``t > 0``), or ``None``. This is the
    mesh-based realization of the line-to-surface intersection (cf. the iterative
    projection algorithm in Beer, *Algorithms for geometrical operations with
    NURBS surfaces*): robust and analysis-suitable for the faceted surfaces this
    kernel produces.
    """
    grid = surface_grid(surf, nu, nv)
    best_t = None
    best_p = None
    for i in range(nu):
        for j in range(nv):
            quad = [grid[i][j], grid[i][j + 1],
                    grid[i + 1][j + 1], grid[i + 1][j]]
            for tri in ((quad[0], quad[1], quad[2]),
                        (quad[0], quad[2], quad[3])):
                t = ray_triangle_intersect(origin, direction, *tri)
                if t is not None and (best_t is None or t < best_t):
                    best_t = t
                    best_p = origin + direction * t
    return best_p


def tessellate_surface_trim(surf: "NURBSSurface", plane: "TrimPlane",
                            nu: int = 16, nv: int = 16):
    """
    Tessellate the +side of ``surf`` clipped by ``plane`` into a triangle mesh.

    Each grid quad is split into two triangles, each clipped against the plane so
    the section follows the true surface–plane intersection (piecewise-linear).
    Returns ``(points, triangles)`` where ``points`` is a list of Point3D and
    ``triangles`` is a list of index 3-tuples. This is the geometry-aware,
    cap-cut-capable replacement for a rectangular NURBS crop.
    """
    grid = surface_grid(surf, nu, nv)
    points: List[Point3D] = []
    triangles: List[tuple] = []

    def _add(p: Point3D) -> int:
        points.append(p)
        return len(points) - 1

    for i in range(nu):
        for j in range(nv):
            quad = [grid[i][j], grid[i][j + 1],
                    grid[i + 1][j + 1], grid[i + 1][j]]
            for tri in ((quad[0], quad[1], quad[2]),
                        (quad[0], quad[2], quad[3])):
                for kept in _clip_triangle_keep(tri, plane):
                    triangles.append(tuple(_add(p) for p in kept))
    return points, triangles


# --------------------------------------------------------------------------- #
# Affine transformations (homogeneous 4x4 matrices)
# --------------------------------------------------------------------------- #
def translation_matrix(dx: float, dy: float, dz: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = [dx, dy, dz]
    return m


def scaling_matrix(factor: float, center: Point3D | None = None) -> np.ndarray:
    c = center.to_array() if center else np.zeros(3)
    m = np.eye(4)
    m[0, 0] = m[1, 1] = m[2, 2] = factor
    m[:3, 3] = c - factor * c  # scale about the given center
    return m


def rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Rotation about a principal axis ('x', 'y' or 'z') by angle_deg degrees."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    axis = axis.lower()
    if axis == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == "y":
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    elif axis == "z":
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    else:
        raise ValueError(f"unknown rotation axis: {axis!r} (use x, y or z)")
    return m


def apply_matrix(matrix: np.ndarray, point: Point3D) -> Point3D:
    """Apply a 4x4 homogeneous transform to a point."""
    v = np.array([point.x, point.y, point.z, 1.0])
    out = matrix @ v
    return Point3D(out[0], out[1], out[2])
