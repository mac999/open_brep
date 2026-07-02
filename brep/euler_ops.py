"""
Layer 1 - Micro Euler Operators (atomic topology).

These are the *only* functions allowed to mutate the half-edge graph. Each one
preserves the Euler-Poincare invariant by construction. They know nothing about
geometry beyond storing a Point3D on a freshly created vertex.

Implemented operators
    mvfs       Make Vertex Face Solid   - seed a solid from a single point
    mev        Make Edge Vertex         - grow a wire by one edge + vertex
    mef        Make Edge Face           - close a loop, splitting off a new face
    kef        Kill Edge Face           - inverse of mef (merge two faces)
    mekr       Make Edge Kill Ring      - bridge an inner ring to its outer loop
    kemr       Kill Edge Make Ring      - split a loop into an outer loop + inner ring
    mfkr       Make Face Kill Ring      - turn an inner ring into its own face (cavity lid)
    kfmr       Kill Face Make Ring      - inverse of mfkr (open a handle / cavity)
    split_edge Insert a vertex at a point on an edge, updating both adj. loops

The CLI exposes mvfs / mev / mef directly; the rest are used by macros and
available programmatically.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .geometry import Point3D
from .model import Kernel
from .topology import Edge, Face, HalfEdge, Loop, Vertex


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def find_outgoing_in_loop(loop: Loop, vertex: Vertex) -> Optional[HalfEdge]:
    """Return the half-edge of ``loop`` that starts at ``vertex`` (or None)."""
    for he in loop.halfedges():
        if he.vertex is vertex:
            return he
    return None


def _reassign_loop(loop: Loop, start: HalfEdge) -> None:
    """Point ``loop`` at ``start`` and make every half-edge in its cycle own it."""
    loop.halfedge = start
    for he in loop.halfedges():
        he.loop = loop


# --------------------------------------------------------------------------- #
# MVFS - Make Vertex Face Solid
# --------------------------------------------------------------------------- #
def mvfs(
    kernel: Kernel,
    point: Point3D,
    name: str = "",
    solid_oid: Optional[int] = None,
    vertex_oid: Optional[int] = None,
    face_oid: Optional[int] = None,
) -> Tuple:
    """
    Create a minimal solid: one vertex, one face, one (empty) loop.

    The loop holds a single "seed" half-edge whose edge is None. The first
    :func:`mev` reuses this seed so no degenerate half-edge survives.

    Returns ``(solid, face, vertex)``.
    """
    solid = kernel.new_solid(name, solid_oid)
    face = kernel.new_face(solid, face_oid)
    solid.add_face(face)

    loop = kernel.new_loop(face)
    face.add_loop(loop)

    vertex = kernel.new_vertex(point, vertex_oid)
    solid.add_vertex(vertex)

    seed = kernel.new_halfedge()
    seed.vertex = vertex
    seed.loop = loop
    seed.next = seed.prev = seed  # a one-element cycle
    loop.halfedge = seed
    vertex.halfedge = seed
    return solid, face, vertex


# --------------------------------------------------------------------------- #
# MEV - Make Edge Vertex
# --------------------------------------------------------------------------- #
def mev(
    kernel: Kernel,
    v1: Vertex,
    new_point: Point3D,
    he_ref: Optional[HalfEdge] = None,
    edge_oid: Optional[int] = None,
    vertex_oid: Optional[int] = None,
) -> Tuple[Edge, Vertex]:
    """
    Add a new vertex ``v2`` at ``new_point`` and an edge ``v1``--``v2``.

    ``he_ref`` selects *which* corner at ``v1`` the spike is attached to (needed
    once ``v1`` belongs to several loops). When omitted, ``v1``'s stored
    half-edge is used.

    Returns ``(edge, v2)``.
    """
    he1 = he_ref or v1.halfedge
    if he1 is None or he1.vertex is not v1:
        raise ValueError(f"no half-edge starting at vertex #{v1.oid} to attach to")

    loop = he1.loop
    solid = kernel.solid_of(loop)
    v2 = kernel.new_vertex(new_point, vertex_oid)
    edge = kernel.new_edge(edge_oid)

    # Special case: the very first MEV reuses the MVFS seed half-edge so the
    # solid never carries a dangling edge-less half-edge.
    if he1.edge is None:
        ha = he1                       # becomes v1 -> v2
        hb = kernel.new_halfedge()     # becomes v2 -> v1
        ha.edge = hb.edge = edge
        edge.he1, edge.he2 = ha, hb
        hb.vertex = v2
        hb.loop = loop
        ha.next = hb
        hb.prev = ha
        hb.next = ha
        ha.prev = hb
    else:
        ha = kernel.new_halfedge()     # v1 -> v2
        hb = kernel.new_halfedge()     # v2 -> v1
        ha.edge = hb.edge = edge
        edge.he1, edge.he2 = ha, hb
        ha.vertex = v1
        hb.vertex = v2
        ha.loop = hb.loop = loop
        # Insert the spike (ha, hb) immediately before he1.
        prev = he1.prev
        prev.next = ha
        ha.prev = prev
        ha.next = hb
        hb.prev = ha
        hb.next = he1
        he1.prev = hb

    v2.halfedge = hb
    if v1.halfedge is None:
        v1.halfedge = ha
    solid.add_vertex(v2)
    solid.add_edge(edge)
    return edge, v2


# --------------------------------------------------------------------------- #
# MEF - Make Edge Face
# --------------------------------------------------------------------------- #
def _mef(kernel: Kernel, he1: HalfEdge, he2: HalfEdge) -> Tuple[Edge, Face]:
    """
    Core split: insert an edge from ``vtx(he1)`` to ``vtx(he2)`` (both in the
    same loop), splitting that loop in two. ``he1``'s cycle becomes a brand new
    face/loop; ``he2``'s cycle stays in the original loop.

    Returns ``(edge, new_face)``.
    """
    if he1.loop is not he2.loop:
        raise ValueError("MEF requires both half-edges to lie in the same loop")
    old_loop = he1.loop
    old_face = old_loop.face
    solid = kernel.solid_of(old_face)

    edge = kernel.new_edge()
    # ha runs A->B (A = vtx(he1)), hb runs B->A (B = vtx(he2)). Pairing the
    # vertex with the *opposite* half-edge's successor is what keeps the new
    # edge non-degenerate.
    ha = kernel.new_halfedge()  # A -> B, ends up in he2's (old) loop
    hb = kernel.new_halfedge()  # B -> A, ends up in he1's (new) loop
    ha.edge = hb.edge = edge
    edge.he1, edge.he2 = ha, hb
    ha.vertex = he1.vertex
    hb.vertex = he2.vertex

    he1_prev = he1.prev
    he2_prev = he2.prev

    # ha closes he2's cycle: ... he1_prev -> ha -> he2 ...
    he1_prev.next = ha
    ha.prev = he1_prev
    ha.next = he2
    he2.prev = ha
    # hb closes he1's cycle: ... he2_prev -> hb -> he1 ...
    he2_prev.next = hb
    hb.prev = he2_prev
    hb.next = he1
    he1.prev = hb

    # he1's cycle -> new face; he2's cycle stays in the old loop.
    new_face = kernel.new_face(solid)
    new_loop = kernel.new_loop(new_face)
    new_face.add_loop(new_loop)
    _reassign_loop(new_loop, he1)
    _reassign_loop(old_loop, he2)
    solid.add_face(new_face)
    solid.add_edge(edge)
    return edge, new_face


def mef(
    kernel: Kernel,
    v1: Vertex,
    v2: Vertex,
    loop: Optional[Loop] = None,
) -> Tuple[Edge, Face]:
    """
    CLI-friendly Make Edge Face: connect vertices ``v1`` and ``v2`` that share a
    loop, splitting it into a new face. If ``loop`` is not given, the common loop
    is discovered automatically.

    Returns ``(edge, new_face)``.
    """
    he1, he2 = _resolve_common_loop(v1, v2, loop)
    return _mef(kernel, he1, he2)


def _resolve_common_loop(
    v1: Vertex, v2: Vertex, loop: Optional[Loop]
) -> Tuple[HalfEdge, HalfEdge]:
    """Find half-edges starting at v1 and v2 within a shared loop."""
    candidate_loops = [loop] if loop else _loops_around(v1)
    for lp in candidate_loops:
        he1 = find_outgoing_in_loop(lp, v1)
        he2 = find_outgoing_in_loop(lp, v2)
        if he1 and he2:
            return he1, he2
    raise ValueError(
        f"vertices #{v1.oid} and #{v2.oid} do not share a loop for MEF"
    )


def _loops_around(vertex: Vertex):
    """All loops that contain an outgoing half-edge of ``vertex``."""
    seen = []
    start = vertex.halfedge
    if start is None:
        return seen
    he = start
    # Walk the half-edges emanating from this vertex via mate/next rotation.
    while True:
        if he.loop and he.loop not in seen:
            seen.append(he.loop)
        mate = he.mate
        if mate is None:
            break
        he = mate.next
        if he is start or he is None:
            break
    return seen


# --------------------------------------------------------------------------- #
# KEF - Kill Edge Face (inverse of MEF)
# --------------------------------------------------------------------------- #
def kef(kernel: Kernel, edge: Edge) -> None:
    """
    Remove ``edge`` and merge the two faces it separates back into one.

    The face owning the mate half-edge is destroyed; its loop's half-edges are
    spliced into the surviving loop.
    """
    ha, hb = edge.he1, edge.he2
    keep_loop = ha.loop
    kill_loop = hb.loop
    if keep_loop is kill_loop:
        raise ValueError("KEF expects the edge to border two different loops")
    keep_face = keep_loop.face
    kill_face = kill_loop.face
    solid = kernel.solid_of(keep_face)

    # Splice: remove ha and hb, joining their neighbours across the gap.
    ha.prev.next = hb.next
    hb.next.prev = ha.prev
    hb.prev.next = ha.next
    ha.next.prev = hb.prev

    _reassign_loop(keep_loop, ha.prev)  # ha.prev is guaranteed to survive
    # Fix any vertex pointers that referenced the doomed half-edges.
    if ha.vertex.halfedge is ha:
        ha.vertex.halfedge = hb.next
    if hb.vertex.halfedge is hb:
        hb.vertex.halfedge = ha.next

    solid.remove_face(kill_face)
    solid.remove_edge(edge)
    for dead in (ha, hb, edge, kill_loop, kill_face):
        kernel.destroy(dead)


# --------------------------------------------------------------------------- #
# Ring operators (inner loops / holes in a face)
# --------------------------------------------------------------------------- #
def kemr(kernel: Kernel, he: HalfEdge) -> Tuple[Loop, Edge]:
    """
    Kill Edge, Make Ring.

    Remove ``he``'s edge; the loop it lived in splits into two loops that share
    the face. The detached cycle becomes a new inner ring. Used to carve a hole
    boundary out of a single loop.

    Returns ``(new_ring_loop, removed_edge)``.
    """
    edge = he.edge
    ha, hb = edge.he1, edge.he2
    loop = ha.loop
    face = loop.face
    solid = kernel.solid_of(face)

    # Detach the edge, leaving two separate cycles.
    ha.prev.next = hb.next
    hb.next.prev = ha.prev
    hb.prev.next = ha.next
    ha.next.prev = hb.prev

    new_loop = kernel.new_loop(face)
    face.add_loop(new_loop)               # appended -> becomes an inner ring
    _reassign_loop(loop, ha.prev)
    _reassign_loop(new_loop, hb.prev)

    if ha.vertex.halfedge is ha:
        ha.vertex.halfedge = hb.next
    if hb.vertex.halfedge is hb:
        hb.vertex.halfedge = ha.next

    solid.remove_edge(edge)
    for dead in (ha, hb, edge):
        kernel.destroy(dead)
    return new_loop, edge


def mekr(
    kernel: Kernel, he_outer: HalfEdge, he_ring: HalfEdge
) -> Edge:
    """
    Make Edge, Kill Ring (inverse of :func:`kemr`).

    Bridge an inner ring loop to the outer loop of the same face with a new edge,
    merging the two loops into one and deleting the ring.

    Returns the new edge.
    """
    outer_loop = he_outer.loop
    ring_loop = he_ring.loop
    face = outer_loop.face
    if ring_loop.face is not face:
        raise ValueError("MEKR expects both loops to belong to the same face")
    solid = kernel.solid_of(face)

    edge = kernel.new_edge()
    ha = kernel.new_halfedge()  # vtx(he_outer) -> vtx(he_ring)
    hb = kernel.new_halfedge()  # vtx(he_ring)  -> vtx(he_outer)
    ha.edge = hb.edge = edge
    edge.he1, edge.he2 = ha, hb
    ha.vertex = he_outer.vertex
    hb.vertex = he_ring.vertex

    o_prev = he_outer.prev
    r_prev = he_ring.prev
    # Stitch the two cycles into one through the bridge edge.
    o_prev.next = ha
    ha.prev = o_prev
    ha.next = he_ring
    he_ring.prev = ha
    r_prev.next = hb
    hb.prev = r_prev
    hb.next = he_outer
    he_outer.prev = hb

    _reassign_loop(outer_loop, he_outer)
    face.loops.remove(ring_loop)
    solid.add_edge(edge)
    kernel.destroy(ring_loop)
    return edge


def kfmr(kernel: Kernel, face: Face) -> Loop:
    """
    Kill Face, Make Ring.

    Destroy ``face`` (which must have a single loop) and re-home its boundary
    loop as an inner ring of an adjacent face reached across one of its edges.
    This opens a handle/cavity, so the solid's genus is incremented.

    Returns the relocated ring loop.
    """
    if len(face.loops) != 1:
        raise ValueError("KFMR expects a face with exactly one loop")
    loop = face.outer
    solid = kernel.solid_of(face)

    # Find an adjacent face by crossing any edge of the loop.
    adjacent = None
    for he in loop.halfedges():
        mate = he.mate
        if mate and mate.loop and mate.loop.face is not face:
            adjacent = mate.loop.face
            break
    if adjacent is None:
        raise ValueError("KFMR could not find an adjacent face to host the ring")

    adjacent.add_loop(loop)  # demote to an inner ring of the neighbour
    solid.remove_face(face)
    solid.genus += 1
    kernel.destroy(face)
    return loop


def mfkr(kernel: Kernel, ring_loop: Loop) -> Face:
    """
    Make Face, Kill Ring (inverse of :func:`kfmr`).

    Promote an inner ring loop into its own face, closing a handle/cavity and
    decrementing the solid's genus.

    Returns the new face.
    """
    host = ring_loop.face
    if ring_loop not in host.inner:
        raise ValueError("MFKR expects an inner ring loop")
    solid = kernel.solid_of(host)

    host.loops.remove(ring_loop)
    new_face = kernel.new_face(solid)
    new_face.add_loop(ring_loop)
    solid.add_face(new_face)
    solid.genus = max(0, solid.genus - 1)
    return new_face


# --------------------------------------------------------------------------- #
# SPLIT_EDGE - Insert a vertex on an edge, updating both adjacent loops
# --------------------------------------------------------------------------- #
def split_edge(
    kernel: Kernel,
    edge: Edge,
    point: Point3D,
) -> Tuple[Edge, Vertex]:
    """
    Insert a new vertex ``M`` at ``point`` on ``edge``, splitting it into two.

    Both loops adjacent to the edge are updated coherently so no degenerate
    (spike) half-edges are created. After the call::

        Before:  L1: ...→ he1(A→B) →...    L2: ...→ he2(B→A) →...
        After:   L1: ...→ he1(A→M) → hm1(M→B) →...
                 L2: ...→ he2(B→M) → hm2(M→A) →...

    Edge pairing afterwards:
        - original ``edge`` covers A↔M  (he1, hm2 as mates)
        - returned ``new_edge``  covers M↔B  (hm1, he2 as mates)

    Returns ``(new_edge, new_vertex)``.

    Euler invariant: +1 V, +1 E, ΔF=0  →  V-E+F unchanged ✓
    """
    he1 = edge.he1   # A→B in loop L1
    he2 = edge.he2   # B→A in loop L2

    solid = kernel.solid_of(he1.loop)
    m = kernel.new_vertex(point)
    solid.add_vertex(m)

    # ── L1: insert hm1(M→B) between he1 and he1.next ────────────────────── #
    hm1 = kernel.new_halfedge()
    hm1.vertex = m
    hm1.loop = he1.loop
    he1_next = he1.next
    he1.next = hm1
    hm1.prev = he1
    hm1.next = he1_next
    he1_next.prev = hm1

    # ── L2: insert hm2(M→A) between he2 and he2.next ────────────────────── #
    hm2 = kernel.new_halfedge()
    hm2.vertex = m
    hm2.loop = he2.loop
    he2_next = he2.next
    he2.next = hm2
    hm2.prev = he2
    hm2.next = he2_next
    he2_next.prev = hm2

    # ── Re-pair edges ────────────────────────────────────────────────────── #
    # Original edge: A↔M  →  he1 (A→M in L1) mates hm2 (M→A in L2)
    edge.he2 = hm2
    hm2.edge = edge
    # he1.edge = edge already; he1 is unchanged.

    # New edge: M↔B  →  hm1 (M→B in L1) mates he2 (B→M in L2)
    new_edge = kernel.new_edge()
    new_edge.he1 = hm1
    new_edge.he2 = he2
    hm1.edge = new_edge
    he2.edge = new_edge

    m.halfedge = hm1
    solid.add_edge(new_edge)
    return new_edge, m
