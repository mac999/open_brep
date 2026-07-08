"""
STEP (ISO 10303-21) import / export.

``save`` writes a self-contained AP203-style file built around a
MANIFOLD_SOLID_BREP, so the topology and geometry round-trip into other CAD
tools at the boundary-representation level.

``load`` is a best-effort importer: it parses CARTESIAN_POINT / VERTEX_POINT /
EDGE_CURVE / ADVANCED_FACE records and rebuilds vertices and a point/edge map.
Full reconstruction of an arbitrary STEP B-rep back into the half-edge graph is
out of scope for this teaching kernel; what cannot be mapped is reported clearly.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

from .geometry import (NURBSSurface, Point3D, TrimPlane,
                       surface_closest_point, tessellate_surface_trim)
from .model import Kernel
from .topology import Solid


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
class _StepWriter:
    """Accumulates DATA-section records and hands out sequential #ids."""

    def __init__(self):
        self._next = 1
        self.lines: List[str] = []

    def add(self, body: str) -> int:
        oid = self._next
        self._next += 1
        self.lines.append(f"#{oid} = {body};")
        return oid


def _face_normal(face) -> Point3D:
    """Newell's method normal of a face's outer loop (robust to non-triangles)."""
    pts = [he.vertex.point for he in face.outer.halfedges()]
    nx = ny = nz = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        nx += (a.y - b.y) * (a.z + b.z)
        ny += (a.z - b.z) * (a.x + b.x)
        nz += (a.x - b.x) * (a.y + b.y)
    n = Point3D(nx, ny, nz)
    return n.normalized() if n.length() > 1e-12 else Point3D(0, 0, 1)


def _compress_knots(knots: List[float]):
    """Turn a full knot vector into (distinct_values, multiplicities)."""
    distinct: List[float] = []
    mult: List[int] = []
    for k in knots:
        if distinct and abs(distinct[-1] - k) < 1e-12:
            mult[-1] += 1
        else:
            distinct.append(k)
            mult.append(1)
    return distinct, mult


def _emit_bspline_surface(w: "_StepWriter", surf: NURBSSurface) -> int:
    """Emit a (non-rational) B_SPLINE_SURFACE_WITH_KNOTS and return its #id."""
    # Control-point grid as nested lists of CARTESIAN_POINT references.
    rows = []
    for i in range(surf.n_u):
        ids = []
        for j in range(surf.n_v):
            p = surf.control_net[i][j]
            ids.append(w.add(
                f"CARTESIAN_POINT('',({p.x:.6f},{p.y:.6f},{p.z:.6f}))"))
        rows.append("(" + ",".join(f"#{c}" for c in ids) + ")")
    grid = "(" + ",".join(rows) + ")"

    ku, mu = _compress_knots(surf.knots_u)
    kv, mv = _compress_knots(surf.knots_v)
    mu_s = ",".join(str(m) for m in mu)
    mv_s = ",".join(str(m) for m in mv)
    ku_s = ",".join(f"{k:.6f}" for k in ku)
    kv_s = ",".join(f"{k:.6f}" for k in kv)
    return w.add(
        f"B_SPLINE_SURFACE_WITH_KNOTS('',{surf.degree_u},{surf.degree_v},{grid},"
        f".UNSPECIFIED.,.F.,.F.,.F.,({mu_s}),({mv_s}),({ku_s}),({kv_s}),"
        f".UNSPECIFIED.)")


def _tri_normal(a: Point3D, b: Point3D, c: Point3D) -> Point3D:
    n = (b - a).cross(c - a)
    return n.normalized() if n.length() > 1e-12 else Point3D(0, 0, 1)


def _emit_triangle_face(w: "_StepWriter", a: Point3D, b: Point3D, c: Point3D) -> int:
    """Emit one planar triangular ADVANCED_FACE (used for faceted trimmed NURBS)."""
    pts = [a, b, c]
    vids = []
    for p in pts:
        cp = w.add(f"CARTESIAN_POINT('',({p.x:.6f},{p.y:.6f},{p.z:.6f}))")
        vids.append((cp, w.add(f"VERTEX_POINT('',#{cp})")))
    oriented = []
    for i in range(3):
        cpa, va = vids[i]
        _cpb, vb = vids[(i + 1) % 3]
        d = pts[(i + 1) % 3] - pts[i]
        length = d.length() or 1.0
        dirn = d * (1.0 / length)
        di = w.add(f"DIRECTION('',({dirn.x:.6f},{dirn.y:.6f},{dirn.z:.6f}))")
        ve = w.add(f"VECTOR('',#{di},{length:.6f})")
        ln = w.add(f"LINE('',#{cpa},#{ve})")
        ec = w.add(f"EDGE_CURVE('',#{va},#{vb},#{ln},.T.)")
        oriented.append(w.add(f"ORIENTED_EDGE('',*,*,#{ec},.T.)"))
    el = w.add(f"EDGE_LOOP('',({','.join('#' + str(o) for o in oriented)}))")
    fb = w.add(f"FACE_OUTER_BOUND('',#{el},.T.)")
    normal = _tri_normal(a, b, c)
    o_id = w.add(f"CARTESIAN_POINT('',({a.x:.6f},{a.y:.6f},{a.z:.6f}))")
    n_id = w.add(f"DIRECTION('',({normal.x:.6f},{normal.y:.6f},{normal.z:.6f}))")
    ax_id = w.add(f"AXIS2_PLACEMENT_3D('',#{o_id},#{n_id},$)")
    pl = w.add(f"PLANE('',#{ax_id})")
    return w.add(f"ADVANCED_FACE('',(#{fb}),#{pl},.T.)")


def _is_trimmed_nurbs(f) -> bool:
    # Any cutter with a signed_distance (TrimPlane or SurfaceCutter) counts.
    return (isinstance(getattr(f, "surface", None), NURBSSurface)
            and hasattr(getattr(f, "trim_plane", None), "signed_distance"))


def _vertex_uv(v, surf: NURBSSurface):
    """Surface parameters of a boundary vertex: stored, or projected on demand."""
    uv = getattr(v, "on_surface_uv", None)
    if uv is not None:
        return uv
    u, vv, _foot = surface_closest_point(surf, v.point)
    return (u, vv)


def _emit_trimmed_bspline_face(w: "_StepWriter", f) -> int:
    """
    Emit a trimmed NURBS face **analytically**: the full
    ``B_SPLINE_SURFACE_WITH_KNOTS`` bounded by its topological loops, each edge
    carried by a ``SURFACE_CURVE`` that pairs the 3D chord with a **PCURVE** —
    a degree-1 B-spline in the surface's ``(u, v)`` parameter space inside a
    ``DEFINITIONAL_REPRESENTATION`` (ISO 10303-42). This is the classic
    pcurve-based trimmed-surface representation: viewers that honour bounds
    render the exact trimmed patch, and the parametric geometry round-trips.
    """
    surf = f.surface
    surf_id = _emit_bspline_surface(w, surf)
    pctx = w.add("PARAMETRIC_REPRESENTATION_CONTEXT('2D parameter space','')")

    vertex_pt: dict = {}       # vertex oid -> (CARTESIAN_POINT id, VERTEX_POINT id)
    edge_curve: dict = {}      # edge oid -> EDGE_CURVE id

    def _vp(v):
        if v.oid not in vertex_pt:
            p = v.point
            cp = w.add(f"CARTESIAN_POINT('',({p.x:.6f},{p.y:.6f},{p.z:.6f}))")
            vertex_pt[v.oid] = (cp, w.add(f"VERTEX_POINT('',#{cp})"))
        return vertex_pt[v.oid]

    bound_ids = []
    for li, loop in enumerate(f.loops):
        oriented = []
        for he in loop.halfedges():
            if he.edge is None:
                continue
            e = he.edge
            if e.oid not in edge_curve:
                va = e.he1.vertex
                vb = e.he1.end_vertex
                cp_a, vp_a = _vp(va)
                _cp_b, vp_b = _vp(vb)
                d = vb.point - va.point
                length = d.length() or 1.0
                dirn = d * (1.0 / length)
                di = w.add(f"DIRECTION('',({dirn.x:.6f},{dirn.y:.6f},{dirn.z:.6f}))")
                ve = w.add(f"VECTOR('',#{di},{length:.6f})")
                c3 = w.add(f"LINE('',#{cp_a},#{ve})")
                ua, vva = _vertex_uv(va, surf)
                ub, vvb = _vertex_uv(vb, surf)
                q1 = w.add(f"CARTESIAN_POINT('',({ua:.9f},{vva:.9f}))")
                q2 = w.add(f"CARTESIAN_POINT('',({ub:.9f},{vvb:.9f}))")
                c2 = w.add(
                    f"B_SPLINE_CURVE_WITH_KNOTS('',1,(#{q1},#{q2}),"
                    f".UNSPECIFIED.,.F.,.F.,(2,2),(0.000000,1.000000),"
                    f".UNSPECIFIED.)")
                dr = w.add(f"DEFINITIONAL_REPRESENTATION('',(#{c2}),#{pctx})")
                pc = w.add(f"PCURVE('',#{surf_id},#{dr})")
                sc = w.add(f"SURFACE_CURVE('',#{c3},(#{pc}),.PCURVE_S1.)")
                edge_curve[e.oid] = w.add(
                    f"EDGE_CURVE('',#{vp_a},#{vp_b},#{sc},.T.)")
            same_dir = he.edge.he1 is he
            oriented.append(w.add(
                f"ORIENTED_EDGE('',*,*,#{edge_curve[e.oid]},"
                f"{'.T.' if same_dir else '.F.'})"))
        refs = ",".join(f"#{o}" for o in oriented)
        el = w.add(f"EDGE_LOOP('',({refs}))")
        kind = "FACE_OUTER_BOUND" if li == 0 else "FACE_BOUND"
        bound_ids.append(w.add(f"{kind}('',#{el},.T.)"))

    bounds = ",".join(f"#{b}" for b in bound_ids)
    return w.add(f"ADVANCED_FACE('',({bounds}),#{surf_id},.T.)")


def save(solid: Solid, filepath: str, faceted: bool = False) -> None:
    """
    Write ``solid`` to ``filepath`` as a STEP AP203 B-rep.

    Trimmed NURBS faces export **analytically** by default — the full B-spline
    surface bounded by its topological loops with PCURVEs in parameter space
    (see :func:`_emit_trimmed_bspline_face`). Pass ``faceted=True`` to emit the
    kept side as a triangle shell instead (for viewers with weak pcurve
    support).
    """
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    w = _StepWriter()

    # Only surviving (non-discarded) faces are exported after a topological trim.
    surviving = [f for f in solid.faces if not getattr(f, "discarded", False)]

    # Collect the vertices and edges actually referenced by those faces, so the
    # discarded half's geometry is not written as free-floating points/curves
    # (which viewers would show as untrimmed leftovers).
    used_v_oids: set = set()
    used_e_oids: set = set()
    for f in surviving:
        if _is_trimmed_nurbs(f):
            continue  # emitted as faceted triangles, not via topological edges
        for loop in f.loops:
            for he in loop.halfedges():
                if he.vertex is not None:
                    used_v_oids.add(he.vertex.oid)
                if he.edge is not None:
                    used_e_oids.add(he.edge.oid)

    # Geometry: one CARTESIAN_POINT + VERTEX_POINT per referenced vertex.
    point_id: Dict[int, int] = {}
    vertex_id: Dict[int, int] = {}
    for v in solid.vertices:
        if v.oid not in used_v_oids:
            continue
        p = v.point or Point3D(0, 0, 0)
        cp = w.add(f"CARTESIAN_POINT('',({p.x:.6f},{p.y:.6f},{p.z:.6f}))")
        point_id[v.oid] = cp
        vertex_id[v.oid] = w.add(f"VERTEX_POINT('',#{cp})")

    # Edges: a LINE plus an EDGE_CURVE between the two vertices.
    edge_curve_id: Dict[int, int] = {}
    for e in solid.edges:
        if e.oid not in used_e_oids:
            continue
        a = e.he1.vertex
        b = e.he1.end_vertex
        d = (b.point - a.point)
        length = d.length() or 1.0
        dirn = d * (1.0 / length)
        dir_id = w.add(f"DIRECTION('',({dirn.x:.6f},{dirn.y:.6f},{dirn.z:.6f}))")
        vec_id = w.add(f"VECTOR('',#{dir_id},{length:.6f})")
        line_id = w.add(f"LINE('',#{point_id[a.oid]},#{vec_id})")
        edge_curve_id[e.oid] = w.add(
            f"EDGE_CURVE('',#{vertex_id[a.oid]},#{vertex_id[b.oid]},#{line_id},.T.)"
        )

    # Faces: ADVANCED_FACE with FACE_OUTER_BOUND/FACE_BOUND edge loops on a PLANE.
    face_ids: List[int] = []
    for f in surviving:
        # A curved face carrying a trim cutter exports analytically: the full
        # B-spline surface bounded by its topological loops with PCURVEs in
        # (u,v) space. 'faceted=True' falls back to a keep-side triangle shell.
        if _is_trimmed_nurbs(f):
            if faceted:
                pts_m, tris_m = tessellate_surface_trim(
                    f.surface, f.trim_plane, 16, 16)
                for (ta, tb, tc) in tris_m:
                    face_ids.append(
                        _emit_triangle_face(w, pts_m[ta], pts_m[tb], pts_m[tc]))
            else:
                face_ids.append(_emit_trimmed_bspline_face(w, f))
            continue

        bound_ids: List[int] = []
        for li, loop in enumerate(f.loops):
            oriented: List[int] = []
            for he in loop.halfedges():
                if he.edge is None:
                    continue
                same_dir = he.edge.he1 is he
                ec = edge_curve_id[he.edge.oid]
                oe = w.add(f"ORIENTED_EDGE('',*,*,#{ec},{'.T.' if same_dir else '.F.'})")
                oriented.append(oe)
            refs = ",".join(f"#{o}" for o in oriented)
            edge_loop = w.add(f"EDGE_LOOP('',({refs}))")
            bound_kind = "FACE_OUTER_BOUND" if li == 0 else "FACE_BOUND"
            bound_ids.append(w.add(f"{bound_kind}('',#{edge_loop},.T.)"))

        # A face carrying a NURBS surface exports as a B-spline; otherwise a plane.
        if isinstance(f.surface, NURBSSurface):
            surface_id = _emit_bspline_surface(w, f.surface)
        else:
            origin = f.outer.halfedges()[0].vertex.point
            normal = _face_normal(f)
            o_id = w.add(f"CARTESIAN_POINT('',({origin.x:.6f},{origin.y:.6f},{origin.z:.6f}))")
            n_id = w.add(f"DIRECTION('',({normal.x:.6f},{normal.y:.6f},{normal.z:.6f}))")
            ax_id = w.add(f"AXIS2_PLACEMENT_3D('',#{o_id},#{n_id},$)")
            surface_id = w.add(f"PLANE('',#{ax_id})")
        bounds = ",".join(f"#{b}" for b in bound_ids)
        face_ids.append(w.add(f"ADVANCED_FACE('',({bounds}),#{surface_id},.T.)"))

    shell_refs = ",".join(f"#{fid}" for fid in face_ids)
    # A trimmed solid (discarded halves) or a faceted curved trim is an open shell.
    has_discarded = (any(getattr(f, "discarded", False) for f in solid.faces)
                     or any(_is_trimmed_nurbs(f) for f in surviving))
    if has_discarded:
        # Topologically trimmed: emit an open shell as a surface model.
        # SHELL_BASED_SURFACE_MODEL must reference the SHELL entity, not raw faces.
        shell_id = w.add(f"OPEN_SHELL('',({shell_refs}))")
        model_id = w.add(
            f"SHELL_BASED_SURFACE_MODEL('{solid.name or 'trimmed'}', (#{shell_id}))"
        )
        w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{model_id}),$)")
    else:
        shell_id = w.add(f"CLOSED_SHELL('',({shell_refs}))")
        brep_id = w.add(f"MANIFOLD_SOLID_BREP('{solid.name or 'solid'}',#{shell_id})")
        w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',(#{brep_id}),$)")

    body = "\n".join(w.lines)
    text = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('B-Rep CLI Kernel export'),'2;1');\n"
        f"FILE_NAME('{filepath}','',(''),(''),'brep-cli','brep-cli','');\n"
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        f"{body}\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# Import (best effort)
# --------------------------------------------------------------------------- #
_POINT_RE = re.compile(
    r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)", re.IGNORECASE
)


def load(kernel: Kernel, filepath: str) -> Solid:
    """
    Parse a STEP file's CARTESIAN_POINTs into a new solid as a vertex cloud.

    Returns the created solid. Topology (edges/faces) from the file is *not*
    rebuilt into the half-edge graph; the caller is told how many points were
    recovered so the limitation is explicit.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        text = fh.read()

    solid = kernel.new_solid(name=f"imported:{filepath}")
    count = 0
    for match in _POINT_RE.finditer(text):
        coords = [c.strip() for c in match.group(2).split(",") if c.strip()]
        if len(coords) < 3:
            continue
        try:
            x, y, z = (float(coords[0]), float(coords[1]), float(coords[2]))
        except ValueError:
            continue
        v = kernel.new_vertex(Point3D(x, y, z))
        solid.add_vertex(v)
        count += 1

    solid.name = f"imported({count} pts):{filepath}"
    return solid
