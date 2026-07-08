"""
Layer 3 - Macro Modeler.

High-level modeling commands implemented purely as sequences of Micro Euler
operators plus geometry assignment. Nothing here pokes at half-edge pointers
directly; it goes through euler_ops so topology stays valid.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from . import euler_ops as eu
from .geometry import (Bezier, NURBSSurface, Point3D, SurfaceCutter, TrimPlane,
                       bezier_plane_param, line_plane_intersect,
                       ray_surface_intersect, ray_surface_intersect_ex,
                       surface_closest_point, surface_directional_derivs,
                       surface_normal, surface_plane_section,
                       surface_plane_side, surface_surface_section)
from .mesh import build_solid_from_faces
from .model import Kernel
from .topology import Edge, Face, Solid, Vertex


# --------------------------------------------------------------------------- #
# Extrude (sweep a planar face along a vector)
# --------------------------------------------------------------------------- #
def extrude(kernel: Kernel, face: Face, direction: Point3D) -> Face:
    """
    Sweep ``face``'s outer loop along ``direction``, turning it into a prism.

    The passed face is pushed to become the *top* cap; its mate face stays put as
    the bottom. Side faces are generated one per profile edge. Returns the top
    face (same object as ``face``).

    Algorithm (classic Euler-operator sweep):
        1. MEV every profile vertex straight up by ``direction``.
        2. MEF consecutive top vertices to spin out each side face; the original
           loop shrinks until it is exactly the top cap.
    """
    loop = face.outer
    base_halfedges = loop.halfedges()
    base_vertices: List[Vertex] = [he.vertex for he in base_halfedges]
    n = len(base_vertices)
    if n < 3:
        raise ValueError("extrude needs a closed profile of at least 3 vertices")

    # 1) raise each base vertex, remembering the corresponding top vertex.
    top_vertices: List[Vertex] = []
    for he in base_halfedges:
        base_v = he.vertex
        top_point = base_v.point + direction
        _edge, top_v = eu.mev(kernel, base_v, top_point, he_ref=he)
        top_vertices.append(top_v)

    # 2) close side faces between consecutive top vertices.
    for i in range(n):
        t_curr = top_vertices[i]
        t_next = top_vertices[(i + 1) % n]
        he1 = eu.find_outgoing_in_loop(loop, t_curr)
        he2 = eu.find_outgoing_in_loop(loop, t_next)
        if he1 is None or he2 is None:
            raise RuntimeError("extrude lost track of a top vertex (internal error)")
        eu._mef(kernel, he1, he2)

    return face


# --------------------------------------------------------------------------- #
# Create box (mvfs -> mev x3 -> mef -> extrude)
# --------------------------------------------------------------------------- #
def create_box(
    kernel: Kernel,
    length: float,
    width: float,
    height: float,
    origin: Point3D = Point3D(0, 0, 0),
    name: str = "box",
    solid_oid: Optional[int] = None,
):
    """
    Build an axis-aligned box of size length(x) * width(y) * height(z).

    Returns the solid. Demonstrates the full macro pipeline described in the PRD:
    a square base laid down with MVFS/MEV/MEF, then swept upward with extrude.
    """
    ox, oy, oz = origin.x, origin.y, origin.z
    # Counter-clockwise base square (viewed from +z).
    p0 = Point3D(ox, oy, oz)
    p1 = Point3D(ox + length, oy, oz)
    p2 = Point3D(ox + length, oy + width, oz)
    p3 = Point3D(ox, oy + width, oz)

    solid, base_face, v0 = eu.mvfs(kernel, p0, name=name, solid_oid=solid_oid)
    _e, v1 = eu.mev(kernel, v0, p1)
    _e, v2 = eu.mev(kernel, v1, p2)
    _e, v3 = eu.mev(kernel, v2, p3)
    # Close the square: this creates the second face (the one we will sweep).
    _edge, top_face = eu.mef(kernel, v3, v0)

    extrude(kernel, top_face, Point3D(0, 0, height))
    return solid


# --------------------------------------------------------------------------- #
# Sphere (faceted UV sphere built directly from its mesh)
# --------------------------------------------------------------------------- #
def create_sphere(
    kernel: Kernel,
    radius: float,
    slices: int = 16,
    stacks: int = 8,
    center: Point3D = Point3D(0, 0, 0),
    name: str = "sphere",
    solid_oid: Optional[int] = None,
):
    """
    Build a UV sphere of ``radius`` with ``slices`` longitude and ``stacks``
    latitude divisions. Triangle fans cap the poles; quads fill the bands. The
    result is a closed, valid 2-manifold solid.
    """
    if slices < 3 or stacks < 2:
        raise ValueError("sphere needs slices>=3 and stacks>=2")

    points: List[Point3D] = []
    north = len(points)
    points.append(center + Point3D(0, 0, radius))

    ring_start: List[int] = []
    for i in range(1, stacks):
        theta = math.pi * i / stacks          # 0 (north) .. pi (south)
        z = radius * math.cos(theta)
        ring_r = radius * math.sin(theta)
        ring_start.append(len(points))
        for j in range(slices):
            phi = 2 * math.pi * j / slices
            points.append(center + Point3D(ring_r * math.cos(phi),
                                           ring_r * math.sin(phi), z))
    south = len(points)
    points.append(center + Point3D(0, 0, -radius))

    faces: List[List[int]] = []
    top = ring_start[0]
    for j in range(slices):                   # north cap fan
        faces.append([north, top + j, top + (j + 1) % slices])
    for i in range(stacks - 2):               # middle quad bands
        cur, nxt = ring_start[i], ring_start[i + 1]
        for j in range(slices):
            j2 = (j + 1) % slices
            faces.append([cur + j, nxt + j, nxt + j2, cur + j2])
    bot = ring_start[-1]
    for j in range(slices):                   # south cap fan
        faces.append([south, bot + (j + 1) % slices, bot + j])

    solid = build_solid_from_faces(kernel, points, faces, name=name)
    if solid_oid is not None:
        kernel.registry.unregister(solid.oid)
        kernel.registry.register(solid, solid_oid)
    return solid


# --------------------------------------------------------------------------- #
# Cylinder (n-gon caps + side quads)
# --------------------------------------------------------------------------- #
def create_cylinder(
    kernel: Kernel,
    radius: float,
    height: float,
    slices: int = 16,
    center: Point3D = Point3D(0, 0, 0),
    name: str = "cylinder",
    solid_oid: Optional[int] = None,
):
    """Build a capped cylinder of ``radius`` and ``height`` along +z."""
    if slices < 3:
        raise ValueError("cylinder needs slices>=3")

    points: List[Point3D] = []
    for j in range(slices):                   # bottom ring 0..slices-1
        phi = 2 * math.pi * j / slices
        points.append(center + Point3D(radius * math.cos(phi),
                                       radius * math.sin(phi), 0))
    for j in range(slices):                   # top ring slices..2*slices-1
        phi = 2 * math.pi * j / slices
        points.append(center + Point3D(radius * math.cos(phi),
                                       radius * math.sin(phi), height))

    faces: List[List[int]] = []
    # bottom cap: wound so the outward normal points down (-z)
    faces.append([j for j in range(slices - 1, -1, -1)])
    # top cap: outward normal points up (+z)
    faces.append([slices + j for j in range(slices)])
    # side quads
    for j in range(slices):
        j2 = (j + 1) % slices
        faces.append([j, j2, slices + j2, slices + j])

    solid = build_solid_from_faces(kernel, points, faces, name=name)
    if solid_oid is not None:
        kernel.registry.unregister(solid.oid)
        kernel.registry.register(solid, solid_oid)
    return solid


# --------------------------------------------------------------------------- #
# NURBS dome (a square lamina carrying a curved NURBS surface)
# --------------------------------------------------------------------------- #
def create_nurbs_dome(
    kernel: Kernel,
    size: float,
    height: float,
    name: str = "nurbs",
    solid_oid: Optional[int] = None,
):
    """
    Build a square lamina (two faces sharing a 4-edge boundary) and attach a
    degree-2 NURBS dome surface to its front face. A lamina is the smallest valid
    closed object on which to demonstrate a free-form surface.
    """
    s = size / 2.0
    p0 = Point3D(-s, -s, 0)
    p1 = Point3D(s, -s, 0)
    p2 = Point3D(s, s, 0)
    p3 = Point3D(-s, s, 0)
    solid, base_face, v0 = eu.mvfs(kernel, p0, name=name, solid_oid=solid_oid)
    _e, v1 = eu.mev(kernel, v0, p1)
    _e, v2 = eu.mev(kernel, v1, p2)
    _e, v3 = eu.mev(kernel, v2, p3)
    _edge, front = eu.mef(kernel, v3, v0)

    # 3x3 control net: corners flat, edge midpoints raised half, centre raised full.
    m, h = 0.0, height
    net = [
        [Point3D(-s, -s, 0), Point3D(0, -s, h / 2), Point3D(s, -s, 0)],
        [Point3D(-s, m, h / 2), Point3D(0, 0, h), Point3D(s, m, h / 2)],
        [Point3D(-s, s, 0), Point3D(0, s, h / 2), Point3D(s, s, 0)],
    ]
    front.surface = NURBSSurface(net, degree_u=2, degree_v=2)
    return solid


# --------------------------------------------------------------------------- #
# Plane (flat rectangular lamina)
# --------------------------------------------------------------------------- #
def create_plane(
    kernel: Kernel,
    width: float,
    height: float,
    origin: Point3D = Point3D(0, 0, 0),
    name: str = "plane",
    solid_oid: Optional[int] = None,
) -> Solid:
    """
    Build a rectangular planar lamina of size width(x) × height(y) at z=origin.z.

    A lamina is a two-face solid: the front face and its mate share the same four
    boundary edges. This is the minimal closed B-Rep for a free-form surface and
    is the natural target for plane-on-plane trimming demos.
    """
    hw, hh = width / 2.0, height / 2.0
    p0 = Point3D(origin.x - hw, origin.y - hh, origin.z)
    p1 = Point3D(origin.x + hw, origin.y - hh, origin.z)
    p2 = Point3D(origin.x + hw, origin.y + hh, origin.z)
    p3 = Point3D(origin.x - hw, origin.y + hh, origin.z)

    solid, _base_face, v0 = eu.mvfs(kernel, p0, name=name, solid_oid=solid_oid)
    _e, v1 = eu.mev(kernel, v0, p1)
    _e, v2 = eu.mev(kernel, v1, p2)
    _e, v3 = eu.mev(kernel, v2, p3)
    eu.mef(kernel, v3, v0)
    return solid


# --------------------------------------------------------------------------- #
# Revolve (assign rotational surface math to a swept profile)
# --------------------------------------------------------------------------- #
def revolve(
    kernel: Kernel,
    face: Face,
    axis: str,
    angle_deg: float,
    segments: int = 4,
) -> Face:
    """
    Sweep a planar profile around a principal axis to approximate rotational
    geometry. The topology is generated as a faceted prism-of-revolution via
    repeated MEV/MEF, and each side face is tagged with cylindrical surface math.

    This is a teaching-grade revolve: it produces a closed, valid solid and
    attaches NURBS surfaces, not an analytic exact body.
    """
    loop = face.outer
    base_vertices = [he.vertex for he in loop.halfedges()]
    n = len(base_vertices)
    if n < 2:
        raise ValueError("revolve needs a profile of at least 2 vertices")

    from .geometry import rotation_matrix, apply_matrix

    step = angle_deg / segments
    current_face = face
    for _s in range(segments):
        m = rotation_matrix(axis, step)
        ring_loop = current_face.outer
        ring_hes = ring_loop.halfedges()
        ring_vs = [he.vertex for he in ring_hes]
        m_count = len(ring_vs)

        new_tops = []
        for he in ring_hes:
            rotated = apply_matrix(m, he.vertex.point)
            _e, tv = eu.mev(kernel, he.vertex, rotated, he_ref=he)
            new_tops.append(tv)
        for i in range(m_count):
            t_curr = new_tops[i]
            t_next = new_tops[(i + 1) % m_count]
            he1 = eu.find_outgoing_in_loop(ring_loop, t_curr)
            he2 = eu.find_outgoing_in_loop(ring_loop, t_next)
            _e, side = eu._mef(kernel, he1, he2)
            side.surface = _cylinder_surface(side)
    return current_face


def _cylinder_surface(face: Face) -> NURBSSurface:
    """Build a simple bilinear NURBS patch from a quad face's corner points."""
    pts = [he.vertex.point for he in face.outer.halfedges()]
    if len(pts) < 4:
        # Degenerate; fall back to whatever points exist.
        pts = (pts + pts)[:4]
    net = [[pts[0], pts[1]], [pts[3], pts[2]]]
    return NURBSSurface(net, degree_u=1, degree_v=1)


# --------------------------------------------------------------------------- #
# Trim
# --------------------------------------------------------------------------- #
def trim_curve(kernel: Kernel, edge: Edge, u: float) -> Tuple[Vertex, Edge]:
    """
    Split ``edge`` topologically (and geometrically when a Bezier is attached) at
    parameter ``u`` in (0, 1), inserting a new vertex M *on* the edge.

    The edge A↔B becomes two collinear, connected segments A↔M and M↔B — never a
    dangling spike. Implemented with :func:`euler_ops.split_edge` (+1V +1E, ΔF=0,
    both adjacent loops kept coherent), followed by geometric subdivision of any
    attached Bezier curve so each segment carries its matching half.

    Returns ``(new_vertex, new_edge)`` where ``new_edge`` covers the M↔B segment.
    """
    if not (0.0 < u < 1.0):
        raise ValueError("trim parameter u must be strictly between 0 and 1")

    he = edge.he1
    v_start = he.vertex
    v_end = he.end_vertex

    # Geometric split point + Bezier subdivision (evaluated before topology
    # changes so vertex positions are still those of the original endpoints).
    if edge.curve is not None and isinstance(edge.curve, Bezier):
        split_point = edge.curve.evaluate(u)
        left, right = edge.curve.split(u)
    else:
        split_point = v_start.point * (1 - u) + v_end.point * u
        left = right = None

    # Topologically insert M on the edge: ``edge`` keeps A↔M, ``new_edge`` gets M↔B.
    new_edge, new_vertex = eu.split_edge(kernel, edge, split_point)
    if left is not None:
        edge.curve = left
        new_edge.curve = right
    return new_vertex, new_edge


def _crop_nurbs(surf: NURBSSurface, u0: float, u1: float,
                v0: float, v1: float) -> NURBSSurface:
    """
    Return the sub-surface of ``surf`` spanning the parametric rectangle
    ``[u0, u1] × [v0, v1]``, reparameterised back to the full [0, 1]² domain.

    Built from :meth:`NURBSSurface.split_u` / :meth:`split_v` (De Casteljau),
    so it is exact for the degree-1/2 Bézier nets this kernel produces.
    """
    s = surf
    # Crop u to [u0, u1]. Trim the high end first, then the low end (whose
    # parameter must be rescaled by the shrunken domain u1).
    if u1 < 1.0 - 1e-12:
        s = s.split_u(u1)[0]
        if u0 > 1e-12:
            s = s.split_u(u0 / u1)[1]
    elif u0 > 1e-12:
        s = s.split_u(u0)[1]
    # Crop v to [v0, v1] the same way.
    if v1 < 1.0 - 1e-12:
        s = s.split_v(v1)[0]
        if v0 > 1e-12:
            s = s.split_v(v0 / v1)[1]
    elif v0 > 1e-12:
        s = s.split_v(v0)[1]
    return s


def trim_surface_region(kernel: Kernel, face: Face,
                        u0: float, u1: float, v0: float, v1: float) -> Face:
    """
    Trim a NURBS ``face`` to the parametric sub-rectangle ``[u0, u1] × [v0, v1]``.

    The face's ``surface`` is replaced by the cropped sub-surface (via
    :func:`_crop_nurbs`), so ``view`` and STEP ``save`` render only the retained
    region — a genuine surface trim, not a metadata tag. The kept window is also
    recorded on ``face.trim_uv`` for ``disp math`` inspection. Returns the face.
    """
    surf = getattr(face, "surface", None)
    if not isinstance(surf, NURBSSurface):
        raise ValueError(f"face #{face.oid} carries no NURBS surface to trim")
    for name, lo, hi in (("u", u0, u1), ("v", v0, v1)):
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError(
                f"{name} range must satisfy 0 <= {name}0 < {name}1 <= 1 "
                f"(got {lo}, {hi})")
    face.surface = _crop_nurbs(surf, u0, u1, v0, v1)
    face.trim_uv = (u0, u1, v0, v1)  # type: ignore[attr-defined]
    return face


def trim_surface(kernel: Kernel, face: Face, ring_loop_face_id: int) -> Face:
    """
    Legacy metadata form: tag ``face`` with a trimming-loop entity id.

    Kept for backward compatibility; prefer :func:`trim_surface_region` for a
    surface trim that actually crops the geometry. Returns the face.
    """
    face.trim_boundary = ring_loop_face_id  # type: ignore[attr-defined]
    return face


# --------------------------------------------------------------------------- #
# Trim result bookkeeping
# --------------------------------------------------------------------------- #
class TrimResult:
    """Summary returned by ``trim_solid_by_plane``."""

    def __init__(self, solid_oid: int, n_keep: int, n_cut: int, n_discard: int):
        self.solid_oid = solid_oid
        self.n_keep = n_keep        # faces entirely on the keep (+) side
        self.n_cut = n_cut          # faces that were topologically split
        self.n_discard = n_discard  # faces entirely on the discard (−) side
        self.is_topological = False  # True when edges/faces were actually modified

    def __repr__(self) -> str:
        kind = "topological" if self.is_topological else "parametric"
        return (f"TrimResult({kind}, solid=#{self.solid_oid}, "
                f"keep={self.n_keep}, cut={self.n_cut}, discard={self.n_discard})")


# --------------------------------------------------------------------------- #
# Trim solid by infinite plane
# --------------------------------------------------------------------------- #
def trim_solid_by_plane(
    kernel: Kernel,
    solid: Solid,
    nx: float,
    ny: float,
    nz: float,
    d: float,
    keep_below: bool = False,
) -> TrimResult:
    """
    Trim *solid* with the infinite half-space ``nx·x + ny·y + nz·z = d``.

    By default the ``above`` side (``nx·x + ny·y + nz·z > d``) is KEPT. Pass
    ``keep_below=True`` to keep the opposite half instead — this simply negates
    the plane (normal and ``d``) so the ``−`` side becomes the retained ``+``
    side, reusing the same machinery. The reference plane you specify is
    unchanged; only which side survives flips.

    **All solids** — laminas (``create plane`` / ``create nurbs``) *and* volumetric
    meshes (``create box`` / ``sphere`` / ``cylinder``) — receive a **full
    topological trim** via :func:`_trim_by_plane`:

    1. Every edge that strictly crosses the plane is split with
       :func:`euler_ops.split_edge`, inserting an exact intersection vertex.
    2. Every straddling face is split with MEF between its two on-plane vertices.
    3. The "discard" half of each cut face — and every face entirely on the − side
       — is tagged ``face.discarded = True``.
    4. The STEP exporter (``save``) emits an ``OPEN_SHELL`` of the surviving
       (+ side) faces — the trimmed open surface — with any NURBS surface
       reparameterized to span only the keep half.

    The mesh topology stays a valid closed 2-manifold throughout (the Euler
    invariant is preserved — split_edge: +V+E; MEF: +E+F), so ``check validity``
    passes after trimming; ``discarded`` is purely an export-time flag.

    Only when the plane **misses** the solid (fewer than two on-plane vertices
    arise) does it fall back to parametric ``face.trim_plane`` metadata.

    Returns a :class:`TrimResult` summary.
    """
    if keep_below:                      # flip the half-space we retain
        nx, ny, nz, d = -nx, -ny, -nz, -d
    plane = TrimPlane(Point3D(nx, ny, nz), d)
    return _trim_by_plane(kernel, solid, plane)


def _eval_bezier(pts: List[Point3D], t: float) -> Point3D:
    """De Casteljau evaluation of a Bézier curve — pure Point3D, no numpy."""
    pts = list(pts)
    while len(pts) > 1:
        pts = [pts[i] * (1.0 - t) + pts[i + 1] * t for i in range(len(pts) - 1)]
    return pts[0]


def _eval_surf_pt(ctrl_net: List[List[Point3D]], u: float, v: float) -> Point3D:
    """
    Evaluate a Bézier product surface at (u, v) using De Casteljau — no numpy.
    Each row is evaluated at u, then the resulting column is evaluated at v.
    """
    col = [_eval_bezier(row, u) for row in ctrl_net]
    return _eval_bezier(col, v)


def _nurbs_split_param(surf: NURBSSurface, plane: TrimPlane):
    """
    Binary-search for the u (or v) parameter where the NURBS surface intersects
    the trim plane, using pure Point3D De Casteljau (numpy-free).

    Returns ``(t, 'u')`` or ``(t, 'v')``, or ``(None, None)`` if no crossing
    is found in either direction.
    """
    _ITERS = 60
    net = surf.control_net

    def dist_u(u: float) -> float:
        p = _eval_surf_pt(net, u, 0.5)
        return plane.signed_distance(p)

    def dist_v(v: float) -> float:
        p = _eval_surf_pt(net, 0.5, v)
        return plane.signed_distance(p)

    def bisect(d_lo: float, d_hi: float, fn) -> float:
        lo, hi = (0.0, 1.0) if d_lo < 0 else (1.0, 0.0)
        for _ in range(_ITERS):
            mid = (lo + hi) / 2.0
            if fn(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    d0u, d1u = dist_u(0.0), dist_u(1.0)
    if d0u * d1u < 0.0:
        return bisect(d0u, d1u, dist_u), 'u'

    d0v, d1v = dist_v(0.0), dist_v(1.0)
    if d0v * d1v < 0.0:
        return bisect(d0v, d1v, dist_v), 'v'

    return None, None


def _reparameterize_keep_nurbs(face: Face, plane: TrimPlane) -> None:
    """
    If *face* carries a NURBSSurface, split it at the trim plane's intersection
    parameter and replace the surface with only the keep half.

    This ensures the STEP `B_SPLINE_SURFACE_WITH_KNOTS` exactly spans the keep
    portion of the geometry (u ∈ [0,1] ↔ x/y/z ∈ [trim_coord, max]), so STEP
    viewers render the correct half without needing pcurve trimming metadata.
    """
    surf = getattr(face, "surface", None)
    if not isinstance(surf, NURBSSurface):
        return
    t, axis = _nurbs_split_param(surf, plane)
    if t is None or not (1e-6 < t < 1.0 - 1e-6):
        return  # no clean split found; leave surface as-is

    if axis == 'u':
        left_s, right_s = surf.split_u(t)
        # Keep the half whose centre is on the positive (keep) side of the plane
        d_left_centre = plane.signed_distance(_eval_surf_pt(surf.control_net, t / 2.0, 0.5))
        face.surface = right_s if d_left_centre < 0.0 else left_s
    else:
        low_s, high_s = surf.split_v(t)
        d_low_centre = plane.signed_distance(_eval_surf_pt(surf.control_net, 0.5, t / 2.0))
        face.surface = high_s if d_low_centre < 0.0 else low_s


def _insert_section_ring(kernel: Kernel, face: Face,
                         section: List[Tuple[float, float, Point3D]]) -> Face:
    """
    Lift a closed surface–plane section curve into the half-edge topology of
    ``face`` as a real inner ring, splitting the face in two.

    Classic Mäntylä bridge construction, using only Euler operators so the
    invariant is preserved by construction (net: +nV +nE +1F +1R → ΔEuler = 0):

        1. MEV a *bridge* edge from a boundary vertex of the face to the first
           section point, then MEV-chain the remaining section points (spikes).
        2. MEF between the chain's last and first vertices — this closes the
           section loop and splits off the *cap* face bounded exactly by it.
        3. KEMR kills the bridge edge, turning the section cycle inside the
           remaining face into a proper inner **ring**.

    Each new vertex stores its surface parameters in ``vertex.on_surface_uv``
    (the parametric reconnection of the intersection); each section edge gets a
    straight chord Bezier. Returns the new cap face (bounded by the section);
    the original ``face`` keeps its outer boundary plus the new ring.
    """
    if len(section) < 3:
        raise ValueError("section ring needs at least 3 points")
    loop = face.outer

    # 1) Bridge from an outer-boundary corner to the first section point.
    anchor_he = loop.halfedge
    anchor_v = anchor_he.vertex
    u0, v0, p0 = section[0]
    bridge_edge, first_v = eu.mev(kernel, anchor_v, p0, he_ref=anchor_he)
    first_v.on_surface_uv = (u0, v0)          # type: ignore[attr-defined]

    verts = [first_v]
    prev = first_v
    for (su, sv, sp) in section[1:]:
        chain_edge, nv = eu.mev(kernel, prev, sp, he_ref=prev.halfedge)
        nv.on_surface_uv = (su, sv)           # type: ignore[attr-defined]
        chain_edge.curve = Bezier([prev.point, sp])
        verts.append(nv)
        prev = nv

    # 2) Close the loop: he1 = first->second (so the section cycle becomes the
    #    new cap face), he2 = the chain tip's only outgoing half-edge.
    he1 = None
    for he in loop.halfedges():
        if he.vertex is verts[0] and he.end_vertex is verts[1]:
            he1 = he
            break
    he2 = eu.find_outgoing_in_loop(loop, verts[-1])
    if he1 is None or he2 is None:
        raise RuntimeError("section ring lost its chain half-edges")
    closing_edge, cap_face = eu._mef(kernel, he1, he2)
    closing_edge.curve = Bezier([verts[-1].point, verts[0].point])

    # 3) Remove the bridge; its two half-edges now bound the same loop, so KEMR
    #    detaches the section cycle into an inner ring of the original face.
    eu.kemr(kernel, bridge_edge.he1)
    return cap_face


def _resample_closed(pts: List[Tuple[float, float, Point3D]],
                     n: int) -> List[Tuple[float, float, Point3D]]:
    """Evenly subsample a closed section polyline to at most ``n`` points."""
    # Drop consecutive near-duplicates first (marching artifacts at nodes).
    clean: List[Tuple[float, float, Point3D]] = []
    for item in pts:
        if not clean or (item[2] - clean[-1][2]).length() > 1e-9:
            clean.append(item)
    if len(clean) > 1 and (clean[0][2] - clean[-1][2]).length() <= 1e-9:
        clean.pop()
    if len(clean) <= n:
        return clean
    step = len(clean) / n
    return [clean[int(k * step)] for k in range(n)]


def _section_ring_trim(kernel: Kernel, face: Face,
                       plane: TrimPlane) -> Optional[Face]:
    """
    Topological realisation of an interior (cap) surface–plane cut.

    Extracts the section curve in the surface's ``(u, v)`` space
    (:func:`geometry.surface_plane_section`), lifts the single closed loop into
    the half-edge structure with :func:`_insert_section_ring` (cap face + inner
    ring), then flags the half on the − side as discarded. The keep half owns
    the NURBS surface plus the ``trim_plane`` tag that drives clipped
    tessellation in the viewer and STEP export.

    Returns the discarded half-face, or ``None`` when no single closed section
    loop exists (the face then stays tagged-only, as before).
    """
    surf = face.surface
    loops = surface_plane_section(surf, plane)
    closed = [pts for is_closed, pts in loops if is_closed and len(pts) >= 6]
    if len(closed) != 1:
        return None
    section = _resample_closed(closed[0], 24)
    if len(section) < 3:
        return None

    # Which side of the plane is *inside* the section loop? Probe the surface
    # at the loop's uv centroid (interior for a star-shaped cap loop).
    uc = sum(u for u, _v, _p in section) / len(section)
    vc = sum(v for _u, v, _p in section) / len(section)
    inner_sign = plane.signed_distance(surf.evaluate(uc, vc))

    try:
        cap = _insert_section_ring(kernel, face, section)
    except (ValueError, RuntimeError):
        return None                            # keep tagged-only fallback

    if inner_sign > 0.0:                       # cap region is the keep (+) side
        keep_half, discard_half = cap, face
    else:                                      # annulus keeps, cap discarded
        keep_half, discard_half = face, cap
    keep_half.surface = surf
    keep_half.trim_plane = plane               # type: ignore[attr-defined]
    if discard_half is not keep_half:
        discard_half.surface = None
        discard_half.discarded = True          # type: ignore[attr-defined]
    return discard_half


def _refine_cut_edge(kernel: Kernel, cut_edge: Edge, cutter,
                     support: TrimPlane, n_sub: int = 6) -> None:
    """
    Subdivide a straight MEF cut edge into a polyline that follows the TRUE
    intersection of a curved cutter with the (planar) face it lies in.

    A single chord between the two split vertices misses a curved section by
    up to the sagitta; here each interior chord point is converged onto the
    intersection *curve* by alternating projections — onto the cutter surface
    (closest point) and back onto the face's support plane — then inserted
    with :func:`euler_ops.split_edge`, so the FULL section curve is computed
    and traversable, not just its endpoints. Each new vertex stores its
    ``(u, v)`` on the cutter (``on_surface_uv``).
    """
    a = cut_edge.he1.vertex.point
    b = cut_edge.he1.end_vertex.point
    refined: List[Tuple[Point3D, Tuple[float, float]]] = []
    for k in range(1, n_sub):
        t = k / n_sub
        p = a * (1.0 - t) + b * t
        uv = (0.5, 0.5)
        ok = False
        for _ in range(30):
            u, v, foot = surface_closest_point(cutter.surf, p, *uv, iters=25)
            uv = (u, v)
            dsp = support.signed_distance(foot)
            p_new = foot - support.normal * dsp     # back onto the face plane
            if (p_new - p).length() < 1e-10:
                p = p_new
                ok = True
                break
            p = p_new
        if ok and abs(cutter.signed_distance(p)) < 1e-6:
            refined.append((p, uv))
    cur = cut_edge
    for p, uv in refined:
        cur, mv = eu.split_edge(kernel, cur, p)     # cur := the M..B remainder
        mv.on_surface_uv = uv                       # type: ignore[attr-defined]


def _trim_by_plane(kernel: Kernel, solid: Solid, plane: TrimPlane) -> TrimResult:
    """
    Topological half-space trim for **any** solid — a 2-face lamina, or a closed
    volumetric mesh (box / sphere / cylinder). Keeps the ``+`` side.

    Splits each edge that strictly straddles the plane (inserting an exact
    intersection vertex), then MEF-cuts each straddling face into keep/discard
    halves. Faces on the discard side are tagged ``face.discarded = True`` so the
    STEP exporter emits an OPEN_SHELL of the surviving surface. The in-memory
    topology stays a valid closed 2-manifold, so ``check validity`` still passes.

    Handles clean cuts through existing vertices (e.g. a sphere sliced at its
    equator ring): those vertices act as the section boundary and no edge split
    is needed.
    """
    _TOL = 1e-7

    # ── Step 1: split every edge that STRICTLY crosses the plane ─────────── #
    # Geometry-aware: an edge carrying a Bezier curve is intersected as the real
    # *curve* (the split vertex lands ON the curve and each half keeps its own
    # sub-curve); a straight edge uses its end-point chord.
    for edge in list(solid.edges):       # snapshot — the list grows as we split
        a = edge.he1.vertex
        b = edge.he1.end_vertex
        if a is None or b is None:
            continue
        curve = getattr(edge, "curve", None)
        if isinstance(curve, Bezier):
            u = bezier_plane_param(curve, plane)
            if u is None or not (1e-9 < u < 1.0 - 1e-9):
                continue
            split_point = curve.evaluate(u)
            left, right = curve.split(u)
            new_edge, _m = eu.split_edge(kernel, edge, split_point)
            edge.curve = left
            new_edge.curve = right
            continue
        da = plane.signed_distance(a.point)
        db = plane.signed_distance(b.point)
        if (da > _TOL and db < -_TOL) or (da < -_TOL and db > _TOL):
            if hasattr(plane, "intersect_segment"):
                # Planar cutter: the linear interpolant is the exact zero.
                t = da / (da - db)
                m_pt = a.point * (1.0 - t) + b.point * t
            else:
                # Curved cutter (SurfaceCutter): the zero along the segment is
                # not linear — bisect the signed distance on the true segment
                # so the split vertex lands ON the cutter surface.
                t = bezier_plane_param(Bezier([a.point, b.point]), plane)
                if t is None:
                    continue
                m_pt = a.point * (1.0 - t) + b.point * t
            eu.split_edge(kernel, edge, m_pt)

    # Section boundary = every vertex lying on the plane (fresh splits + any
    # original vertices the plane passes exactly through).
    on_plane = {v for v in solid.vertices
                if abs(plane.signed_distance(v.point)) <= _TOL}

    def _classify(verts: List[Vertex]) -> float:
        return sum(plane.signed_distance(v.point) for v in verts) / max(len(verts), 1)

    def _surface_interior_trim(face: Face):
        """
        Decide a *one-sided-boundary* face from its NURBS surface, not its flat
        polygon. Returns ``'cut'`` (surface straddles → tag for tessellation-
        level trim), ``'keep'``, ``'discard'``, or ``None`` (no surface / can't
        decide, fall back to the boundary sign).
        """
        surf = getattr(face, "surface", None)
        if not isinstance(surf, NURBSSurface):
            return None
        spos, sneg = surface_plane_side(surf, plane)
        if spos and sneg:
            face.trim_plane = plane           # type: ignore[attr-defined]
            return "cut"
        if spos:
            return "keep"
        if sneg:
            return "discard"
        return None

    # ── Step 2: classify / MEF-split each face ───────────────────────────── #
    n_cut = n_discard = n_keep = 0
    for face in list(solid.faces):
        loop = face.outer
        verts = [he.vertex for he in loop.halfedges()]
        has_pos = any(plane.signed_distance(v.point) > _TOL for v in verts)
        has_neg = any(plane.signed_distance(v.point) < -_TOL for v in verts)

        # ── One-sided boundary: the surface (if curved) still gets a vote, so an
        #    interior/cap cut a flat polygon cannot express is handled here. ──
        if not (has_pos and has_neg):
            decision = _surface_interior_trim(face)
            if decision == "cut":             # curved surface straddles plane
                # Lift the surface-plane section into the topology as a real
                # inner ring + cap face, so the trim boundary is traversable.
                discarded_half = _section_ring_trim(kernel, face, plane)
                n_cut += 1
                n_keep += 1
                if discarded_half is not None:
                    n_discard += 1
                continue
            if decision == "keep":
                n_keep += 1
                continue
            if decision == "discard":
                face.discarded = True         # type: ignore[attr-defined]
                n_discard += 1
                continue
            # Planar / surface undecided → decide by boundary sign (as before).
            if not has_neg:
                n_keep += 1
            else:
                face.discarded = True         # type: ignore[attr-defined]
                n_discard += 1
            continue

        # Straddling face: the two boundary vertices in this loop (loop order).
        bv: List[Vertex] = []
        for he in loop.halfedges():
            if he.vertex in on_plane and he.vertex not in bv:
                bv.append(he.vertex)

        M1 = bv[0] if len(bv) >= 1 else None
        M2 = bv[1] if len(bv) >= 2 else None
        he_m1 = eu.find_outgoing_in_loop(loop, M1) if M1 else None
        he_m2 = eu.find_outgoing_in_loop(loop, M2) if M2 else None

        # Fall back to a whole-face keep/discard decision when a clean 2-point
        # cut is not available (adjacent boundary verts would make a degenerate
        # edge; fewer than 2 boundary verts means nothing to split).
        if (he_m1 is None or he_m2 is None
                or he_m1.end_vertex is M2 or he_m2.end_vertex is M1):
            if _classify(verts) < 0.0:
                face.discarded = True         # type: ignore[attr-defined]
                n_discard += 1
            else:
                n_keep += 1
            continue

        # MEF: he_m1's cycle → new_face; he_m2's cycle stays in the old face.
        _cut_edge, new_face = eu._mef(kernel, he_m1, he_m2)
        n_cut += 1

        # Against a CURVED cutter the section across this face is a curve, not
        # a chord: refine the cut edge into a polyline on the true intersection
        # (planar faces only — a NURBS face gets its surface reparameterized).
        if (isinstance(plane, SurfaceCutter)
                and getattr(face, "surface", None) is None):
            support = face_plane(face)
            _refine_cut_edge(kernel, _cut_edge, plane, support)

        # Decide which resulting half is keep / discard by centroid sign.
        new_pts = [he.vertex for he in new_face.outer.halfedges()]
        if _classify(new_pts) >= 0.0:
            keep_face, discard_face = new_face, face
        else:
            keep_face, discard_face = face, new_face

        # Move any NURBS surface onto the keep half and crop it to that half.
        orig_surface = getattr(face, "surface", None)
        if orig_surface is not None and keep_face is not face:
            keep_face.surface = orig_surface
            face.surface = None
        _reparameterize_keep_nurbs(keep_face, plane)

        discard_face.discarded = True         # type: ignore[attr-defined]
        n_discard += 1
        n_keep += 1

    if n_cut == 0 and n_discard == 0:
        # Neither a boundary edge nor any surface actually crossed the plane —
        # it truly misses the solid; record parametric metadata instead.
        return _trim_metadata_by_plane(kernel, solid, plane)

    result = TrimResult(solid.oid, n_keep, n_cut, n_discard)
    result.is_topological = True
    return result


# --------------------------------------------------------------------------- #
# Extend (grow a source entity until it reaches a target entity)
#
# Reuses the same geometric-intersection logic as trim: a curve/ray is met with a
# plane (closed form) or a NURBS surface (ray–tessellation), and BOTH the geometry
# (new point on the target) and the topology (new vertex/edge, or a swept strip of
# faces) are updated. Targets are expressed as ('plane', TrimPlane) or
# ('surface', NURBSSurface); the controller resolves a #<face> into either.
# --------------------------------------------------------------------------- #
def face_plane(face: Face) -> TrimPlane:
    """Best-fit support plane of a (planar) face, via a Newell normal."""
    pts = [he.vertex.point for he in face.outer.halfedges()]
    nx = ny = nz = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        nx += (a.y - b.y) * (a.z + b.z)
        ny += (a.z - b.z) * (a.x + b.x)
        nz += (a.x - b.x) * (a.y + b.y)
    normal = Point3D(nx, ny, nz)
    if normal.length() < 1e-12:
        normal = Point3D(0, 0, 1)
    normal = normal.normalized()
    centroid = Point3D(
        sum(p.x for p in pts) / len(pts),
        sum(p.y for p in pts) / len(pts),
        sum(p.z for p in pts) / len(pts),
    )
    return TrimPlane(normal, normal.dot(centroid))


def _ray_contact(origin: Point3D, direction: Point3D, target):
    """
    Forward contact of ray ``origin + t·direction`` (t>0) with a target.

    ``target`` is ``('plane', TrimPlane)`` or ``('surface', NURBSSurface)``.
    Returns ``(point, uv)`` where ``uv`` is the target's surface parameter pair
    (``None`` for a plane target), or ``None`` if the ray never reaches the
    target ahead of ``origin``. Plane contacts are closed-form; surface contacts
    are tessellation-seeded then Newton-refined onto the true surface
    (:func:`geometry.ray_surface_intersect_ex`), so the point is exact — not a
    facet approximation — and carries its parametric address for reconnection.
    """
    kind, obj = target
    if kind == "plane":
        hit = line_plane_intersect(origin, direction, obj)
        if hit is None:
            return None
        t, pt = hit
        return (pt, None) if t > 1e-9 else None
    hit = ray_surface_intersect_ex(origin, direction, obj)
    if hit is None:
        return None
    pt, _t, u, v = hit
    return pt, (u, v)


def _edge_tangent_out(edge: Edge, at_end: bool) -> Point3D:
    """Outward unit tangent at an edge end (True = end vertex, False = start)."""
    curve = getattr(edge, "curve", None)
    a = edge.he1.vertex.point
    b = edge.he1.end_vertex.point
    if isinstance(curve, Bezier):
        cps = curve.control_points
        d = (cps[-1] - cps[-2]) if at_end else (cps[0] - cps[1])
    else:
        d = (b - a) if at_end else (a - b)
    if d.length() < 1e-12:
        d = (b - a) if at_end else (a - b)
    return d.normalized()


def extend_curve(kernel: Kernel, edge: Edge, target,
                 prefer: Optional[str] = None) -> Tuple[Vertex, Edge]:
    """
    Extend ``edge`` (a curve/segment) until its tangent ray meets ``target``.

    The end whose *forward* tangent ray reaches the target is chosen (``prefer``
    = ``'start'`` / ``'end'`` forces one), a new vertex is inserted at the exact
    contact point on the target, and a new edge is appended from that end to it
    with :func:`euler_ops.mev` (+1V +1E, topology stays valid). The new edge is a
    straight tangent continuation carrying a degree-1 Bezier.

    Returns ``(new_vertex, new_edge)``.
    """
    a = edge.he1.vertex
    b = edge.he1.end_vertex
    order = [(True, b), (False, a)]
    if prefer == "start":
        order = [(False, a)]
    elif prefer == "end":
        order = [(True, b)]
    for at_end, vtx in order:
        direction = _edge_tangent_out(edge, at_end)
        contact = _ray_contact(vtx.point, direction, target)
        if contact is not None:
            point, uv = contact
            new_edge, new_v = eu.mev(kernel, vtx, point, he_ref=vtx.halfedge)
            new_edge.curve = Bezier([vtx.point, point])
            if uv is not None:                 # parametric address on the target
                new_v.on_surface_uv = uv       # type: ignore[attr-defined]
            return new_v, new_edge
    raise ValueError(
        "curve does not reach the target along its tangent "
        "(try the other end, or reposition the target)")


def extend_face(kernel: Kernel, face: Face, target,
                direction: Optional[Point3D] = None) -> Face:
    """
    Extend a planar ``face`` (a sheet) up to ``target`` by sweeping it.

    Every boundary vertex is swept along ``direction`` (default: the face normal,
    auto-flipped toward the target) until it meets the target — a plane (uniform
    stop) or a NURBS surface (per-vertex stop, so the swept cap conforms to the
    surface). Side faces are spun out with MEF exactly as :func:`extrude` does, so
    the result is a valid closed solid whose new cap lies on the target.

    Returns the swept cap face (the same ``face`` object).
    """
    loop = face.outer
    base_hes = loop.halfedges()
    n = len(base_hes)
    if n < 3:
        raise ValueError("extend needs a face with a closed profile (>= 3 verts)")

    if direction is None:
        normal = face_plane(face).normal
        centroid = Point3D(
            sum(he.vertex.point.x for he in base_hes) / n,
            sum(he.vertex.point.y for he in base_hes) / n,
            sum(he.vertex.point.z for he in base_hes) / n,
        )
        if _ray_contact(centroid, normal, target) is not None:
            direction = normal
        elif _ray_contact(centroid, normal * -1.0, target) is not None:
            direction = normal * -1.0
        else:
            raise ValueError(
                "face does not face the target; give an explicit direction")
    else:
        direction = direction.normalized()

    tops: List[Vertex] = []
    for he in base_hes:
        contact = _ray_contact(he.vertex.point, direction, target)
        if contact is None:
            raise ValueError(
                f"vertex #{he.vertex.oid} never reaches the target along the "
                "sweep direction")
        point, uv = contact
        _e, tv = eu.mev(kernel, he.vertex, point, he_ref=he)
        if uv is not None:                     # parametric address on the target
            tv.on_surface_uv = uv              # type: ignore[attr-defined]
        tops.append(tv)

    for i in range(n):
        he1 = eu.find_outgoing_in_loop(loop, tops[i])
        he2 = eu.find_outgoing_in_loop(loop, tops[(i + 1) % n])
        if he1 is None or he2 is None:
            raise RuntimeError("extend lost track of a swept vertex")
        eu._mef(kernel, he1, he2)
    return face


# --------------------------------------------------------------------------- #
# Surface-surface intersection (NURBS ∩ NURBS) and curvature-continuous blend
# --------------------------------------------------------------------------- #
def _resample_branch(pts: List[tuple], n: int) -> List[tuple]:
    """Evenly subsample an SSI branch (tuples ending in a Point3D) to ≤ n."""
    clean: List[tuple] = []
    for item in pts:
        if not clean or (item[-1] - clean[-1][-1]).length() > 1e-9:
            clean.append(item)
    if len(clean) > 1 and (clean[0][-1] - clean[-1][-1]).length() <= 1e-9:
        clean.pop()
    if len(clean) <= n:
        return clean
    step = len(clean) / n
    return [clean[int(k * step)] for k in range(n)]


def intersect_surfaces(kernel: Kernel, face_a: Face, face_b: Face,
                       samples: int = 32, name: str = "intersection"):
    """
    NURBS ∩ NURBS: compute the intersection curve of two NURBS faces and lift
    it into the model as a wire solid (a vertex/edge chain; closed loops are
    MEF-closed into a lamina ribbon).

    The curve comes from :func:`geometry.surface_surface_section` — surface B
    turned into a signed-distance field over A's parameters, zero set marched
    in A's ``(u, v)`` space, every point tightened by alternating closest-point
    projections between both surfaces. Each wire vertex stores its parametric
    address on *both* surfaces (``on_surface_uv`` for A, ``on_surface_uv_b``
    for B) — the classic three-representation form of the intersection (3D
    curve + a pcurve in each parameter space).

    The full branch data is also recorded on both faces as ``face.ssi_curves``
    for ``disp math``. Returns ``(wire_solid, closed, n_points)``.
    """
    surf_a = getattr(face_a, "surface", None)
    surf_b = getattr(face_b, "surface", None)
    if not isinstance(surf_a, NURBSSurface) or not isinstance(surf_b, NURBSSurface):
        raise ValueError("intersect needs two faces that carry NURBS surfaces")
    branches = surface_surface_section(surf_a, surf_b)
    if not branches:
        raise ValueError("the two surfaces do not intersect")
    closed, pts = max(branches, key=lambda cp: len(cp[1]))
    pts = _resample_branch(pts, samples)
    if len(pts) < 2:
        raise ValueError("intersection curve degenerated to a point")

    face_a.ssi_curves = branches           # type: ignore[attr-defined]
    face_b.ssi_curves = branches           # type: ignore[attr-defined]

    ua, va, ub, vb, p0 = pts[0]
    wire, _f, v0 = eu.mvfs(kernel, p0, name=name)
    v0.on_surface_uv = (ua, va)            # type: ignore[attr-defined]
    v0.on_surface_uv_b = (ub, vb)          # type: ignore[attr-defined]
    prev = v0
    for (ua, va, ub, vb, p) in pts[1:]:
        edge, nv = eu.mev(kernel, prev, p, he_ref=prev.halfedge)
        edge.curve = Bezier([prev.point, p])
        nv.on_surface_uv = (ua, va)        # type: ignore[attr-defined]
        nv.on_surface_uv_b = (ub, vb)      # type: ignore[attr-defined]
        prev = nv
    if closed:
        closing_edge, _cap = eu.mef(kernel, prev, v0)
        closing_edge.curve = Bezier([prev.point, v0.point])
    return wire, closed, len(pts)


def trim_solid_by_surface(kernel: Kernel, solid: Solid, cutter_face: Face,
                          keep_below: bool = False) -> TrimResult:
    """
    Trim *solid* with a curved NURBS **surface** as the cutting tool
    (NURBS ∩ NURBS consumed by trim).

    The cutter face's surface is wrapped in a :class:`geometry.SurfaceCutter`
    whose ``signed_distance`` (closest-point projection + normal sign) makes it
    a drop-in replacement for :class:`TrimPlane` — the entire topological trim
    pipeline (curve-aware edge splitting, MEF face splitting, interior section
    rings) runs unchanged against the curved tool. ``keep above`` (default)
    retains the side the cutter's normal points to; ``keep_below`` the other.
    """
    surf = getattr(cutter_face, "surface", None)
    if not isinstance(surf, NURBSSurface):
        raise ValueError(f"cutter face #{cutter_face.oid} carries no NURBS surface")
    cutter = SurfaceCutter(surf, flip=keep_below)
    return _trim_by_plane(kernel, solid, cutter)


def _interpolate_rows(rows: List[List[Point3D]], degree: int) -> List[List[Point3D]]:
    """
    Solve for B-spline control rows that *interpolate* the given sample rows at
    uniform parameters (Beer Alg. 1: assemble the basis matrix ``A`` and solve
    ``A·c = x``). With control points equal to sample points a B-spline only
    approximates its samples; after this solve the assembled patch reproduces
    every sample row exactly, so each cross-section IS its quintic Hermite span.
    """
    m = len(rows)
    if m <= degree + 1:
        return rows                            # Bezier row count: leave as-is
    import numpy as np
    from .geometry import _bspline_basis
    knots = NURBSSurface._clamped_knots(m, degree)
    a_mat = np.zeros((m, m))
    for r in range(m):
        t = min(r / (m - 1), 1.0 - 1e-9)
        for i in range(m):
            a_mat[r, i] = _bspline_basis(i, degree, t, knots)
    n_cols = len(rows[0])
    rhs = np.zeros((m, 3 * n_cols))
    for r, row in enumerate(rows):
        for j, p in enumerate(row):
            rhs[r, 3 * j:3 * j + 3] = (p.x, p.y, p.z)
    sol = np.linalg.solve(a_mat, rhs)
    return [[Point3D(*sol[i, 3 * j:3 * j + 3]) for j in range(n_cols)]
            for i in range(m)]


def _blend_row(surf: NURBSSurface, u: float, v: float, into: Point3D,
               other_end: Point3D, chord: float):
    """
    Hermite end data for one blend row on one surface: rail point, first and
    second derivatives (w.r.t. the blend parameter) of the cross-boundary walk.
    """
    d1, d2, _du, _dv = surface_directional_derivs(surf, u, v, into)
    return d1 * chord, d2 * (chord * chord)


def blend_surfaces(kernel: Kernel, face_a: Face, face_b: Face,
                   width: float, samples: int = 9,
                   name: str = "blend") -> Solid:
    """
    Build a **curvature-continuous (G2) blend patch** across the intersection
    of two NURBS faces.

    Construction (per sample along the NURBS ∩ NURBS curve):

    1. The intersection curve is computed with
       :func:`geometry.surface_surface_section` and resampled.
    2. On each surface, a *rail* point is offset from the intersection by
       ``width`` along the in-surface direction perpendicular to the curve
       (normal × tangent, oriented away from the other surface, kept
       orientation-consistent along the curve).
    3. At each rail the cross-boundary walk toward the intersection is
       differentiated on the true surface (first + second directional
       derivatives, :func:`geometry.surface_directional_derivs`).
    4. A **quintic Hermite** span matches position, first and second derivative
       at both rails — curvature continuity with each host surface along the
       cross direction — and is written as a degree-5 Bezier row:
       ``P0=a, P1=a+a'/5, P2=a+2a'/5+a''/20, P3=b−2b'/5+b''/20, P4=b−b'/5, P5=b``.
    5. The rows assemble into a NURBS patch (degree 5 across × ≤3 along),
       attached to a new lamina solid exactly like ``create nurbs``.

    Returns the new blend solid.
    """
    surf_a = getattr(face_a, "surface", None)
    surf_b = getattr(face_b, "surface", None)
    if not isinstance(surf_a, NURBSSurface) or not isinstance(surf_b, NURBSSurface):
        raise ValueError("blend needs two faces that carry NURBS surfaces")
    if width <= 0:
        raise ValueError("blend width must be positive")

    branches = surface_surface_section(surf_a, surf_b)
    if not branches:
        raise ValueError("the two surfaces do not intersect - nothing to blend")
    closed, pts = max(branches, key=lambda cp: len(cp[1]))
    pts = _resample_branch(pts, samples)
    if closed:
        pts = pts + [pts[0]]               # wrap the strip around a closed curve
    m = len(pts)
    if m < 2:
        raise ValueError("intersection curve degenerated to a point")

    cutter_b = SurfaceCutter(surf_b)
    cutter_a = SurfaceCutter(surf_a)

    rows: List[List[Point3D]] = []
    prev_da: Optional[Point3D] = None
    prev_db: Optional[Point3D] = None
    for k in range(m):
        ua, va, ub, vb, p = pts[k]
        # Tangent of the intersection curve (central difference, wrap-aware).
        p_next = pts[(k + 1) % m if closed else min(k + 1, m - 1)][4]
        p_prev = pts[(k - 1) % m if closed else max(k - 1, 0)][4]
        t = (p_next - p_prev)
        t = t.normalized() if t.length() > 1e-12 else Point3D(1, 0, 0)

        # In-surface offset directions, perpendicular to the curve.
        da = surface_normal(surf_a, ua, va).cross(t)
        db = surface_normal(surf_b, ub, vb).cross(t)
        da = da.normalized() if da.length() > 1e-12 else Point3D(1, 0, 0)
        db = db.normalized() if db.length() > 1e-12 else Point3D(-1, 0, 0)
        if prev_da is not None:            # keep orientation consistent
            if da.dot(prev_da) < 0:
                da = da * -1.0
            if db.dot(prev_db) < 0:
                db = db * -1.0
        else:                              # first sample: offset away from the
            eps = 0.2 * width              # other surface, and to opposite sides
            if abs(cutter_b.signed_distance(p + da * eps)) < \
               abs(cutter_b.signed_distance(p - da * eps)):
                da = da * -1.0
            if abs(cutter_a.signed_distance(p + db * eps)) < \
               abs(cutter_a.signed_distance(p - db * eps)):
                db = db * -1.0
        prev_da, prev_db = da, db

        # Rails: walk `width` along the surface (first-order parameter step).
        _d1a, _d2a, du_a, dv_a = surface_directional_derivs(surf_a, ua, va, da)
        _d1b, _d2b, du_b, dv_b = surface_directional_derivs(surf_b, ub, vb, db)
        ra_u = min(max(ua + du_a * width, 0.0), 1.0)
        ra_v = min(max(va + dv_a * width, 0.0), 1.0)
        rb_u = min(max(ub + du_b * width, 0.0), 1.0)
        rb_v = min(max(vb + dv_b * width, 0.0), 1.0)
        a_pt = surf_a.evaluate(ra_u, ra_v)
        b_pt = surf_b.evaluate(rb_u, rb_v)
        chord = (b_pt - a_pt).length() or 1e-9

        # Hermite data: walk INTO the blend from each rail (toward the curve).
        a1, a2 = _blend_row(surf_a, ra_u, ra_v, da * -1.0, b_pt, chord)
        b1, b2 = _blend_row(surf_b, rb_u, rb_v, db * -1.0, a_pt, chord)
        # C(0)=a, C'(0)=a1, C''(0)=a2 ; C(1)=b, C'(1)=-b1, C''(1)=+b2
        rows.append([
            a_pt,
            a_pt + a1 * (1.0 / 5.0),
            a_pt + a1 * (2.0 / 5.0) + a2 * (1.0 / 20.0),
            b_pt + b1 * (2.0 / 5.0) + b2 * (1.0 / 20.0),
            b_pt + b1 * (1.0 / 5.0),
            b_pt,
        ])

    degree_u = min(3, m - 1)
    patch = NURBSSurface(_interpolate_rows(rows, degree_u), degree_u, 5)

    # Carry the patch on a lamina solid, exactly like create_nurbs_dome.
    c00, c05 = rows[0][0], rows[0][5]
    c10, c15 = rows[-1][0], rows[-1][5]
    solid, _base, v0 = eu.mvfs(kernel, c00, name=name)
    _e, v1 = eu.mev(kernel, v0, c10)
    _e, v2 = eu.mev(kernel, v1, c15)
    _e, v3 = eu.mev(kernel, v2, c05)
    _edge, front = eu.mef(kernel, v3, v0)
    front.surface = patch
    return solid


def _trim_metadata_by_plane(kernel: Kernel, solid: Solid, plane: TrimPlane) -> TrimResult:
    """
    Parametric trim: store intersection geometry on each face as metadata.
    No topology is modified; ``disp math #<face>`` reports the cut boundary.
    """
    _TOL = 1e-9
    n_keep = n_cut = n_discard = 0

    for face in solid.faces:
        verts = [he.vertex for he in face.outer.halfedges()]
        if not verts:
            continue
        dists = [plane.signed_distance(v.point) for v in verts]
        n_pos = sum(1 for dist in dists if dist > _TOL)
        n_neg = sum(1 for dist in dists if dist < -_TOL)

        if n_neg == 0:
            n_keep += 1
            continue

        if n_pos == 0:
            n_discard += 1
            face.trim_plane = plane        # type: ignore[attr-defined]
            face.trim_section = []         # type: ignore[attr-defined]
            continue

        n_cut += 1
        section: List[Point3D] = []
        n_v = len(verts)
        if hasattr(plane, "intersect_segment"):   # curved cutters have no closed form
            for i in range(n_v):
                p0 = verts[i].point
                p1 = verts[(i + 1) % n_v].point
                result = plane.intersect_segment(p0, p1)
                if result is not None:
                    section.append(result[1])
        face.trim_plane = plane        # type: ignore[attr-defined]
        face.trim_section = section    # type: ignore[attr-defined]

    return TrimResult(solid.oid, n_keep, n_cut, n_discard)
