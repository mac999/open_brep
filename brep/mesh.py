"""
Mesh -> half-edge construction.

Some shapes (spheres, cylinders, anything with poles) are awkward to grow one
atomic Euler operator at a time. This module builds a *complete, valid* half-edge
solid directly from a polygon soup: a list of points plus a list of faces given as
vertex-index loops.

The resulting solid satisfies the same invariants the Euler operators guarantee,
so :func:`brep.validate.check_solid` passes on it. The input must describe a
closed, consistently-wound (CCW seen from outside), 2-manifold surface.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .geometry import NURBSSurface, Point3D
from .model import Kernel
from .topology import Solid


def build_solid_from_faces(
    kernel: Kernel,
    points: Sequence[Point3D],
    faces: Sequence[Sequence[int]],
    name: str = "",
    surfaces: Optional[Sequence[Optional[NURBSSurface]]] = None,
    closed: bool = True,
) -> Solid:
    """
    Construct a half-edge solid from ``points`` and ``faces``.

    ``faces[k]`` is a list of indices into ``points`` describing one face's outer
    loop, wound counter-clockwise as seen from outside the solid. Each interior
    edge must be shared by exactly two oppositely-wound faces (a closed manifold).

    ``surfaces[k]`` optionally attaches a NURBS surface to face k.

    With ``closed=False`` the input may describe an *open* shell (a lamina, or
    the surviving half of a trimmed solid): a half-edge with no opposite gets an
    edge that owns it alone (``edge.he2 is None``), and a directed edge reused by
    two same-wound faces is left unpaired instead of raising. Such a shell is not
    a closed 2-manifold, so ``check_solid`` will report the boundary honestly.
    """
    solid = kernel.new_solid(name)

    verts = []
    for p in points:
        v = kernel.new_vertex(p)
        solid.add_vertex(v)
        verts.append(v)

    # Build per-face loops; index every directed half-edge by (start, end).
    he_by_dir = {}
    unpaired = []   # half-edges that can never find a twin (closed=False only)
    for fi, idx_loop in enumerate(faces):
        n = len(idx_loop)
        if n < 3:
            raise ValueError(f"face {fi} has fewer than 3 vertices")
        face = kernel.new_face(solid)
        solid.add_face(face)
        loop = kernel.new_loop(face)
        face.add_loop(loop)

        face_hes = []
        for k in range(n):
            a, b = idx_loop[k], idx_loop[(k + 1) % n]
            duplicate = (a, b) in he_by_dir
            if duplicate and closed:
                raise ValueError(
                    f"directed edge ({a},{b}) used twice - inconsistent winding")
            he = kernel.new_halfedge()
            he.vertex = verts[a]
            he.loop = loop
            face_hes.append(he)
            if not duplicate:
                he_by_dir[(a, b)] = he
            else:
                unpaired.append(he)
            if verts[a].halfedge is None:
                verts[a].halfedge = he

        for k in range(n):
            face_hes[k].next = face_hes[(k + 1) % n]
            face_hes[k].prev = face_hes[(k - 1) % n]
        loop.halfedge = face_hes[0]

        if surfaces is not None and surfaces[fi] is not None:
            face.surface = surfaces[fi]

    # Pair each half-edge with its opposite to form edges.
    for (a, b), he in he_by_dir.items():
        if he.edge is not None:
            continue
        twin = he_by_dir.get((b, a))
        if twin is None:
            if closed:
                raise ValueError(
                    f"open boundary: edge ({a},{b}) has no opposite face - not closed")
            unpaired.append(he)
            continue
        edge = kernel.new_edge()
        edge.he1, edge.he2 = he, twin
        he.edge = twin.edge = edge
        solid.add_edge(edge)

    # Boundary half-edges of an open shell: an edge with a single use.
    for he in unpaired:
        edge = kernel.new_edge()
        edge.he1, edge.he2 = he, None
        he.edge = edge
        solid.add_edge(edge)

    return solid
