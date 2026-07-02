"""
Integrity validation.

Two independent checks:
    1. Euler-Poincare formula:  V - E + F - R = 2 * (S - G)
       where R = inner rings, S = shells, G = genus (handles).
    2. Pointer sanity: no dangling next/prev/twin, loops are closed, mates are
       reciprocal, and every non-seed half-edge has an edge.

Returns a structured report so the View layer can format it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .model import Kernel
from .topology import Solid


@dataclass
class ValidationReport:
    solid_id: int
    v: int = 0
    e: int = 0
    f: int = 0
    rings: int = 0
    shells: int = 0
    genus: int = 0
    euler_lhs: int = 0
    euler_rhs: int = 0
    euler_ok: bool = False
    pointer_errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.euler_ok and not self.pointer_errors


def check_solid(solid: Solid) -> ValidationReport:
    """Run both checks on a single solid."""
    report = ValidationReport(solid_id=solid.oid)
    report.v = solid.num_vertices
    report.e = solid.num_edges
    report.f = solid.num_faces
    report.rings = solid.num_rings
    report.shells = solid.shells
    report.genus = solid.genus

    report.euler_lhs = report.v - report.e + report.f - report.rings
    report.euler_rhs = 2 * (report.shells - report.genus)
    report.euler_ok = report.euler_lhs == report.euler_rhs

    report.pointer_errors = _check_pointers(solid)
    return report


def _check_pointers(solid: Solid) -> List[str]:
    """Validate half-edge connectivity; return a list of human-readable errors."""
    errors: List[str] = []

    for face in solid.faces:
        for loop in face.loops:
            hes = loop.halfedges()
            if not hes:
                errors.append(f"loop #{loop.oid} is empty")
                continue
            for he in hes:
                tag = f"half-edge #{he.oid}"
                if he.next is None or he.prev is None:
                    errors.append(f"{tag} has a None next/prev pointer")
                    continue
                if he.next.prev is not he:
                    errors.append(f"{tag}: next.prev does not point back")
                if he.prev.next is not he:
                    errors.append(f"{tag}: prev.next does not point back")
                if he.vertex is None:
                    errors.append(f"{tag} has no start vertex")
                if he.loop is not loop:
                    errors.append(f"{tag} loop pointer disagrees with its container")
                if he.edge is None:
                    errors.append(f"{tag} has no edge (dangling seed)")
                else:
                    mate = he.mate
                    if mate is None:
                        errors.append(f"{tag} edge has no mate")
                    elif mate.mate is not he:
                        errors.append(f"{tag}: mate is not reciprocal")

    # Every registered edge should own exactly two half-edges.
    for edge in solid.edges:
        if edge.he1 is None or edge.he2 is None:
            errors.append(f"edge #{edge.oid} is missing a half-edge")
    return errors


def check_all(kernel: Kernel) -> List[ValidationReport]:
    return [check_solid(s) for s in kernel.solids]
