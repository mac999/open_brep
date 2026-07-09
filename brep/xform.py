"""
Spatial transformations applied to whole entities.

Topology and geometry always move together: translating a solid shifts every
owned vertex *and* the control nets / control points of any NURBS surface or
Bezier curve attached to its faces and edges. Otherwise the analytic geometry
would detach from the boundary it is supposed to describe.

Shared by the CLI (``controller.do_move`` and friends) and the authoring web app.
"""

from __future__ import annotations

from typing import List

from .geometry import Bezier, NURBSSurface, Point3D, apply_matrix
from .topology import Edge, Face, Solid, Vertex


def vertices_of(entity) -> List[Vertex]:
    """Every vertex owned by ``entity`` (a vertex, edge, face, or solid)."""
    if isinstance(entity, Vertex):
        return [entity]
    if isinstance(entity, Edge):
        return [entity.he1.vertex, entity.he1.end_vertex]
    if isinstance(entity, Face):
        seen, out = set(), []
        for loop in entity.loops:
            for v in loop.vertices():
                if v.oid not in seen:
                    seen.add(v.oid)
                    out.append(v)
        return out
    if isinstance(entity, Solid):
        return list(entity.vertices)
    raise TypeError(f"#{getattr(entity, 'oid', '?')} cannot be transformed")


def centroid(verts: List[Vertex]) -> Point3D:
    """Average position of the positioned vertices (origin if there are none)."""
    pts = [v.point for v in verts if v.point is not None]
    if not pts:
        return Point3D(0, 0, 0)
    n = len(pts)
    return Point3D(sum(p.x for p in pts) / n,
                   sum(p.y for p in pts) / n,
                   sum(p.z for p in pts) / n)


def apply_transform(entity, matrix) -> None:
    """Apply a 4x4 homogeneous ``matrix`` to ``entity``'s points and geometry."""
    for v in vertices_of(entity):
        if v.point is not None:
            v.point = apply_matrix(matrix, v.point)

    faces: List[Face] = []
    edges: List[Edge] = []
    if isinstance(entity, Solid):
        faces = list(entity.faces)
        edges = list(entity.edges)
    elif isinstance(entity, Face):
        faces = [entity]
        edges = [he.edge for lp in entity.loops
                 for he in lp.halfedges() if he.edge]
    elif isinstance(entity, Edge):
        edges = [entity]

    for f in faces:
        surf = getattr(f, "surface", None)
        if isinstance(surf, NURBSSurface):
            surf.control_net = [[apply_matrix(matrix, p) for p in row]
                                for row in surf.control_net]
    seen = set()
    for e in edges:
        if e is None or e.oid in seen:
            continue
        seen.add(e.oid)
        curve = getattr(e, "curve", None)
        if isinstance(curve, Bezier):
            curve.control_points = [apply_matrix(matrix, p)
                                    for p in curve.control_points]


def bounding_box(entity) -> tuple[Point3D, Point3D]:
    """Axis-aligned (min, max) corners over the entity's positioned vertices."""
    pts = [v.point for v in vertices_of(entity) if v.point is not None]
    if not pts:
        return Point3D(0, 0, 0), Point3D(0, 0, 0)
    return (
        Point3D(min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)),
        Point3D(max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)),
    )
