"""
The Kernel - the 'Model' of the MVC design.

It owns the single :class:`IdRegistry` and the list of solids, and offers thin
helpers to create-and-register topology entities so the Euler operators never
talk to the registry directly. All higher layers (euler_ops, macro, controller)
operate through a Kernel instance.
"""

from __future__ import annotations

from typing import List, Optional

from .registry import IdRegistry
from .topology import Edge, Face, HalfEdge, Loop, Solid, Vertex


class Kernel:
    """Holds global modeling state: the ID registry and all solids."""

    def __init__(self):
        self.registry = IdRegistry(start=100)
        self.solids: List[Solid] = []

    # --- creation helpers (create + register in one step) ----------------- #
    def new_vertex(self, point=None, oid: Optional[int] = None) -> Vertex:
        v = Vertex(point)
        self.registry.register(v, oid)
        return v

    def new_halfedge(self, oid: Optional[int] = None) -> HalfEdge:
        he = HalfEdge()
        self.registry.register(he, oid)
        return he

    def new_edge(self, oid: Optional[int] = None) -> Edge:
        e = Edge()
        self.registry.register(e, oid)
        return e

    def new_loop(self, face: Optional[Face] = None, oid: Optional[int] = None) -> Loop:
        loop = Loop(face)
        self.registry.register(loop, oid)
        return loop

    def new_face(self, solid: Optional[Solid] = None, oid: Optional[int] = None) -> Face:
        f = Face(solid)
        self.registry.register(f, oid)
        return f

    def new_solid(self, name: str = "", oid: Optional[int] = None) -> Solid:
        s = Solid(name)
        self.registry.register(s, oid)
        self.solids.append(s)
        return s

    # --- destruction helpers --------------------------------------------- #
    def destroy(self, entity) -> None:
        """Remove an entity from the registry (its topological unlinking is the
        caller's responsibility)."""
        self.registry.unregister(entity.oid)

    def delete_solid(self, solid: Solid) -> None:
        """Remove a whole solid and every entity it owns from the model."""
        for face in list(solid.faces):
            for loop in list(face.loops):
                for he in loop.halfedges():
                    self.destroy(he)
                self.destroy(loop)
            self.destroy(face)
        for edge in list(solid.edges):
            self.destroy(edge)
        for v in list(solid.vertices):
            self.destroy(v)
        self.destroy(solid)
        if solid in self.solids:
            self.solids.remove(solid)

    # --- lookup ----------------------------------------------------------- #
    def get(self, oid: int):
        return self.registry.get(oid)

    def find(self, oid: int):
        return self.registry.find(oid)

    def solid_of(self, entity) -> Optional[Solid]:
        """Best-effort walk from any entity up to its owning solid."""
        if isinstance(entity, Solid):
            return entity
        if isinstance(entity, Vertex):
            for s in self.solids:
                if entity in s.vertices:
                    return s
            return None
        if isinstance(entity, Edge):
            for s in self.solids:
                if entity in s.edges:
                    return s
            return None
        if isinstance(entity, Face):
            return entity.solid
        if isinstance(entity, Loop):
            return entity.face.solid if entity.face else None
        if isinstance(entity, HalfEdge):
            return self.solid_of(entity.loop) if entity.loop else None
        return None
