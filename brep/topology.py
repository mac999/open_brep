"""
Layer 1 - Topology (Half-Edge data structure).

These classes hold *connectivity only*. Geometry (coordinates, curves, surfaces)
is attached as plain references so the topology stays valid even before any shape
is assigned. The structure follows Mantyla's half-edge model:

    Solid  -> Faces -> Loops -> HalfEdges
    Edge   -> two HalfEdges (a "twin"/mate pair)
    Vertex -> one outgoing HalfEdge

A HalfEdge is a directed use of an edge belonging to exactly one loop. Walking
``next`` around a loop returns to the start; the mate (the edge's other
half-edge) belongs to the neighbouring loop and runs in the opposite direction.

Every entity carries an ``oid`` (object id) once it has been registered.
"""

from __future__ import annotations

from typing import List, Optional

from .geometry import NURBSSurface, Point3D


class Vertex:
    """A topological vertex, optionally positioned by ``point``."""

    def __init__(self, point: Optional[Point3D] = None):
        self.oid: int = -1
        self.point: Optional[Point3D] = point
        self.halfedge: Optional["HalfEdge"] = None  # one outgoing half-edge

    def __repr__(self) -> str:
        return f"V#{self.oid}{self.point if self.point else ''}"


class HalfEdge:
    """A directed use of an edge inside one loop."""

    def __init__(self):
        self.oid: int = -1
        self.vertex: Optional[Vertex] = None      # start vertex of this half-edge
        self.edge: Optional["Edge"] = None         # parent edge (None only for MVFS seed)
        self.loop: Optional["Loop"] = None         # owning loop
        self.next: Optional["HalfEdge"] = None     # next half-edge around the loop
        self.prev: Optional["HalfEdge"] = None     # previous half-edge around the loop

    @property
    def mate(self) -> Optional["HalfEdge"]:
        """The edge's other half-edge (opposite direction), or None for the seed."""
        if self.edge is None:
            return None
        return self.edge.he2 if self.edge.he1 is self else self.edge.he1

    @property
    def end_vertex(self) -> Optional[Vertex]:
        """Destination vertex = start vertex of the next half-edge in the loop."""
        return self.next.vertex if self.next else None

    def __repr__(self) -> str:
        s = self.vertex.oid if self.vertex else "?"
        e = self.end_vertex.oid if self.end_vertex else "?"
        return f"he#{self.oid}(V{s}->V{e})"


class Edge:
    """An undirected edge owning exactly two half-edges (a mate pair)."""

    def __init__(self):
        self.oid: int = -1
        self.he1: Optional[HalfEdge] = None
        self.he2: Optional[HalfEdge] = None
        self.curve = None  # optional geometry (e.g. a Bezier) assigned later

    def __repr__(self) -> str:
        return f"E#{self.oid}"


class Loop:
    """A closed ring of half-edges bounding a region of a face."""

    def __init__(self, face: Optional["Face"] = None):
        self.oid: int = -1
        self.halfedge: Optional[HalfEdge] = None  # an arbitrary half-edge in the loop
        self.face: Optional[Face] = face

    def halfedges(self) -> List[HalfEdge]:
        """Return the loop's half-edges in order, starting from ``self.halfedge``."""
        result: List[HalfEdge] = []
        start = self.halfedge
        if start is None:
            return result
        he = start
        while True:
            result.append(he)
            he = he.next
            if he is start or he is None:
                break
        return result

    def vertices(self) -> List[Vertex]:
        return [he.vertex for he in self.halfedges()]

    def __repr__(self) -> str:
        return f"L#{self.oid}"


class Face:
    """A face with one outer loop and zero or more inner loops (rings/holes)."""

    def __init__(self, solid: Optional["Solid"] = None):
        self.oid: int = -1
        self.loops: List[Loop] = []          # loops[0] is the outer loop
        self.solid: Optional[Solid] = solid
        self.surface: Optional[NURBSSurface] = None  # optional geometry

    @property
    def outer(self) -> Optional[Loop]:
        return self.loops[0] if self.loops else None

    @property
    def inner(self) -> List[Loop]:
        return self.loops[1:]

    def add_loop(self, loop: Loop) -> None:
        loop.face = self
        self.loops.append(loop)

    def __repr__(self) -> str:
        return f"F#{self.oid}"


class Solid:
    """A solid: the top-level container of all topology for one shape."""

    def __init__(self, name: str = ""):
        self.oid: int = -1
        self.name = name
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.faces: List[Face] = []
        # Topological invariants tracked for the Euler-Poincare check.
        self.shells = 1   # connected boundary shells
        self.genus = 0    # number of through-holes (handles)

    # --- bookkeeping helpers ---------------------------------------------- #
    def add_vertex(self, v: Vertex) -> None:
        self.vertices.append(v)

    def add_edge(self, e: Edge) -> None:
        self.edges.append(e)

    def add_face(self, f: Face) -> None:
        f.solid = self
        self.faces.append(f)

    def remove_edge(self, e: Edge) -> None:
        if e in self.edges:
            self.edges.remove(e)

    def remove_vertex(self, v: Vertex) -> None:
        if v in self.vertices:
            self.vertices.remove(v)

    def remove_face(self, f: Face) -> None:
        if f in self.faces:
            self.faces.remove(f)

    # --- counts ----------------------------------------------------------- #
    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_loops(self) -> int:
        return sum(len(f.loops) for f in self.faces)

    @property
    def num_rings(self) -> int:
        """Inner loops = total loops minus one outer loop per face."""
        return self.num_loops - self.num_faces

    def __repr__(self) -> str:
        return f"S#{self.oid}('{self.name}')"
