"""
View - console formatting.

Pure formatting helpers that turn model objects and reports into readable strings.
No printing happens here except through the small ``echo`` helpers; the controller
decides what reaches stdout. Keeping this separate is what makes the design MVC.
"""

from __future__ import annotations

from typing import List

from .geometry import Bezier, NURBSSurface, Point3D, TrimPlane
from .topology import Edge, Face, Solid, Vertex
from .validate import ValidationReport


def _rule(width: int = 60, ch: str = "-") -> str:
    return ch * width


def format_topology(solid: Solid) -> str:
    """Render the full half-edge tree of a solid as an indented outline."""
    lines: List[str] = []
    lines.append(f"Solid #{solid.oid} '{solid.name}'  "
                 f"(V={solid.num_vertices} E={solid.num_edges} "
                 f"F={solid.num_faces} rings={solid.num_rings} "
                 f"shells={solid.shells} genus={solid.genus})")
    lines.append(_rule())
    for face in solid.faces:
        surf = "  surface=NURBS" if face.surface else ""
        lines.append(f"Face #{face.oid}{surf}")
        for li, loop in enumerate(face.loops):
            kind = "outer" if li == 0 else "ring "
            lines.append(f"  Loop #{loop.oid} [{kind}]")
            for he in loop.halfedges():
                start = he.vertex.oid if he.vertex else "?"
                end = he.end_vertex.oid if he.end_vertex else "?"
                edge = f"E#{he.edge.oid}" if he.edge else "E#-- (seed)"
                mate = he.mate
                mate_id = f"he#{mate.oid}" if mate else "--"
                lines.append(
                    f"    he#{he.oid:<4} V{start} -> V{end:<4} "
                    f"{edge:<12} next=he#{he.next.oid:<4} "
                    f"prev=he#{he.prev.oid:<4} twin={mate_id}"
                )
    return "\n".join(lines)


def format_vertices(solid: Solid) -> str:
    """A compact coordinate table for all vertices of a solid."""
    lines = [f"Vertices of Solid #{solid.oid}", _rule(48)]
    lines.append(f"{'id':>6} | {'x':>10} {'y':>10} {'z':>10}")
    lines.append(_rule(48))
    for v in sorted(solid.vertices, key=lambda x: x.oid):
        p = v.point or Point3D(0, 0, 0)
        lines.append(f"{('#' + str(v.oid)):>6} | {p.x:>10.4g} {p.y:>10.4g} {p.z:>10.4g}")
    return "\n".join(lines)


def format_math(entity) -> str:
    """Print the underlying geometric equation / control data of an entity."""
    if isinstance(entity, Vertex):
        p = entity.point or Point3D(0, 0, 0)
        return f"Vertex #{entity.oid}: point = {p}"

    if isinstance(entity, Edge):
        head = f"Edge #{entity.oid}"
        if isinstance(entity.curve, Bezier):
            cps = "\n".join(
                f"    CP[{i}] = {cp}" for i, cp in enumerate(entity.curve.control_points)
            )
            return f"{head}: Bezier degree {entity.curve.degree}\n{cps}"
        if entity.he1 and entity.he2:
            a = entity.he1.vertex
            b = entity.he1.end_vertex
            return (f"{head}: straight segment\n"
                    f"    P(t) = (1-t)*{a.point} + t*{b.point},  t in [0,1]")
        return f"{head}: (no geometry assigned)"

    if isinstance(entity, Face):
        head = f"Face #{entity.oid}"
        lines: List[str] = []
        if isinstance(entity.surface, NURBSSurface):
            s = entity.surface
            grid = "\n".join(
                "    " + "  ".join(str(cp) for cp in row) for row in s.control_net
            )
            lines.append(f"{head}: NURBS surface degree ({s.degree_u},{s.degree_v}), "
                         f"net {s.n_u}x{s.n_v}\n{grid}")
        else:
            lines.append(f"{head}: planar (no parametric surface assigned)")

        # Trim metadata from trim_solid_by_plane
        tp = getattr(entity, "trim_plane", None)
        if isinstance(tp, TrimPlane):
            n = tp.normal
            lines.append(
                f"  trim plane: ({n.x:.4g},{n.y:.4g},{n.z:.4g})·P = {tp.d:.4g}"
            )
            section = getattr(entity, "trim_section", [])
            if section:
                pts_str = "  ->  ".join(str(p) for p in section)
                lines.append(f"  trim section ({len(section)} pts): {pts_str}")
            elif getattr(entity, "discarded", False):
                lines.append("  trim section: face fully discarded (no keep region)")
            else:
                lines.append("  trim: kept +side, faceted along the "
                             "surface-plane intersection")

        # Parametric surface crop from trim_surface_region
        uv = getattr(entity, "trim_uv", None)
        if uv is not None:
            lines.append(
                f"  trim surface region: u=[{uv[0]:.4g},{uv[1]:.4g}] "
                f"v=[{uv[2]:.4g},{uv[3]:.4g}]"
            )

        # Legacy tag from trim_surface
        tb = getattr(entity, "trim_boundary", None)
        if tb is not None:
            lines.append(f"  trim boundary id: #{tb}")

        return "\n".join(lines)

    return f"Entity #{getattr(entity, 'oid', '?')}: no math representation"


def format_validation(report: ValidationReport) -> str:
    """Format a validation report as PASS/FAIL with the Euler breakdown."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"Validity of Solid #{report.solid_id}: {status}",
        _rule(48),
        f"  V={report.v}  E={report.e}  F={report.f}  "
        f"rings={report.rings}  shells={report.shells}  genus={report.genus}",
        f"  Euler-Poincare: V-E+F-R = {report.euler_lhs}  "
        f"vs  2(S-G) = {report.euler_rhs}  -> "
        f"{'OK' if report.euler_ok else 'MISMATCH'}",
    ]
    if report.pointer_errors:
        lines.append(f"  Pointer errors ({len(report.pointer_errors)}):")
        for err in report.pointer_errors:
            lines.append(f"    - {err}")
    else:
        lines.append("  Pointers: all next/prev/twin links consistent")
    return "\n".join(lines)


def format_entity_created(label: str, oid: int, extra: str = "") -> str:
    tail = f"  {extra}" if extra else ""
    return f"+ created {label} #{oid}{tail}"
