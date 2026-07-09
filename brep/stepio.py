"""
STEP (ISO 10303-21) import / export.

``save`` / ``save_solids`` write a self-contained AP203-style file built around
one MANIFOLD_SOLID_BREP per solid (or a SHELL_BASED_SURFACE_MODEL for a trimmed
open shell), so topology and geometry round-trip into other CAD tools at the
boundary-representation level.

``load`` reads such a file back into real half-edge topology: it resolves
ADVANCED_FACE -> FACE_BOUND -> EDGE_LOOP -> ORIENTED_EDGE -> EDGE_CURVE ->
VERTEX_POINT -> CARTESIAN_POINT, recovers each face's boundary vertex ring, and
rebuilds the half-edge graph with :func:`brep.mesh.build_solid_from_faces`.
B_SPLINE_SURFACE_WITH_KNOTS records are re-attached to their faces as
:class:`NURBSSurface`. What the importer cannot map is reported, not guessed:

    * inner bounds (rings/holes) are counted and skipped -- the outer bound of
      each face is what gets rebuilt
    * rational (weighted) B-splines and the ISO complex-record surface form are
      not parsed; those faces come back untagged (planar)
    * a file with no ADVANCED_FACE records falls back to a vertex cloud
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import (NURBSSurface, Point3D, TrimPlane,
                       surface_closest_point, tessellate_surface_trim)
from .mesh import build_solid_from_faces
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


def _emit_solid(w: "_StepWriter", solid: Solid, faceted: bool) -> int:
    """Emit one solid's shell and return its representation-item #id."""
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
        return w.add(
            f"SHELL_BASED_SURFACE_MODEL('{_escape(solid.name) or 'trimmed'}',"
            f"(#{shell_id}))"
        )
    shell_id = w.add(f"CLOSED_SHELL('',({shell_refs}))")
    return w.add(
        f"MANIFOLD_SOLID_BREP('{_escape(solid.name) or 'solid'}',#{shell_id})")


def _escape(name: str) -> str:
    """STEP string literals escape a single quote by doubling it."""
    return (name or "").replace("'", "''")


def save_solids(solids: Sequence[Solid], filepath: str,
                faceted: bool = False) -> None:
    """
    Write every solid in ``solids`` into one STEP AP203 file.

    Each solid contributes its own MANIFOLD_SOLID_BREP (or, when it carries
    trimmed/discarded faces, a SHELL_BASED_SURFACE_MODEL over an OPEN_SHELL);
    a single ADVANCED_BREP_SHAPE_REPRESENTATION collects them all. This is what
    the authoring web app writes, and what :func:`load` reads back.
    """
    if not solids:
        raise ValueError("nothing to save: no solids given")
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    w = _StepWriter()
    item_ids = [_emit_solid(w, s, faceted) for s in solids]
    refs = ",".join(f"#{i}" for i in item_ids)
    w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('',({refs}),$)")

    body = "\n".join(w.lines)
    text = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('B-Rep CLI Kernel export'),'2;1');\n"
        f"FILE_NAME('{_escape(filepath)}','',(''),(''),'brep-cli','brep-cli','');\n"
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        f"{body}\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )
    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(text)


def save(solid: Solid, filepath: str, faceted: bool = False) -> None:
    """
    Write a single ``solid`` to ``filepath`` as a STEP AP203 B-rep.

    Trimmed NURBS faces export **analytically** by default — the full B-spline
    surface bounded by its topological loops with PCURVEs in parameter space
    (see :func:`_emit_trimmed_bspline_face`). Pass ``faceted=True`` to emit the
    kept side as a triangle shell instead (for viewers with weak pcurve
    support).
    """
    save_solids([solid], filepath, faceted=faceted)


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #
_POINT_RE = re.compile(
    r"#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(([^)]*)\)", re.IGNORECASE
)
_RECORD_RE = re.compile(r"#(\d+)\s*=\s*(.*)", re.DOTALL)
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)


def _split_args(text: str) -> List[str]:
    """
    Split the top-level, comma-separated arguments of a STEP parameter list.

    Nested parentheses and quoted literals (where ``''`` is an escaped quote)
    are respected, so ``'a,b',(1,2),#3`` yields three arguments.
    """
    args: List[str] = []
    depth = 0
    in_str = False
    current: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    current.append("''")
                    i += 2
                    continue
                in_str = False
            current.append(ch)
        elif ch == "'":
            in_str = True
            current.append(ch)
        elif ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _parse_records(text: str) -> Dict[int, Tuple[str, List[str]]]:
    """Map every ``#id`` in the DATA section to ``(ENTITY_NAME, [raw args])``."""
    text = _COMMENT_RE.sub(" ", text)
    start = text.upper().find("DATA;")
    if start != -1:
        text = text[start + 5:]
    end = text.upper().rfind("ENDSEC;")
    if end != -1:
        text = text[:end]

    records: Dict[int, Tuple[str, List[str]]] = {}
    for chunk in _split_statements(text):
        match = _RECORD_RE.match(chunk.strip())
        if not match:
            continue
        body = match.group(2).strip()
        paren = body.find("(")
        if paren == -1 or not body.endswith(")"):
            continue
        name = body[:paren].strip().upper()
        if not name:      # ISO complex record: '( A(..) B(..) )' - not supported
            continue
        records[int(match.group(1))] = (name, _split_args(body[paren + 1:-1]))
    return records


def _split_statements(text: str) -> List[str]:
    """Split on ';' outside of quoted literals."""
    out, current, in_str = [], [], False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    current.append("''")
                    i += 2
                    continue
                in_str = False
        elif ch == "'":
            in_str = True
        elif ch == ";":
            out.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    if "".join(current).strip():
        out.append("".join(current))
    return out


def _ref(arg: str) -> Optional[int]:
    """``'#12'`` -> 12; anything else (``*``, ``$``, a literal) -> None."""
    arg = arg.strip()
    return int(arg[1:]) if arg.startswith("#") and arg[1:].isdigit() else None


def _refs(arg: str) -> List[int]:
    """``'(#1,#2)'`` -> [1, 2]."""
    inner = arg.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    out = []
    for part in _split_args(inner):
        r = _ref(part)
        if r is not None:
            out.append(r)
    return out


def _floats(arg: str) -> List[float]:
    """``'(0.,1.5)'`` -> [0.0, 1.5]."""
    inner = arg.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    out = []
    for part in _split_args(inner):
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def _unquote(arg: str) -> str:
    arg = arg.strip()
    if arg.startswith("'") and arg.endswith("'") and len(arg) >= 2:
        return arg[1:-1].replace("''", "'")
    return ""


def _expand_knots(knots: List[float], mults: List[int]) -> List[float]:
    """Rebuild a full knot vector, rescaled to the [0, 1] domain NURBSSurface uses."""
    full: List[float] = []
    for k, m in zip(knots, mults):
        full.extend([k] * int(m))
    if not full:
        return full
    lo, hi = full[0], full[-1]
    span = hi - lo
    if span <= 1e-12:
        return full
    return [(k - lo) / span for k in full]


def _parse_bspline_surface(args: List[str],
                           points: Dict[int, Point3D]) -> Optional[NURBSSurface]:
    """Rebuild a NURBSSurface from B_SPLINE_SURFACE_WITH_KNOTS arguments."""
    if len(args) < 12:
        return None
    try:
        degree_u, degree_v = int(args[1]), int(args[2])
    except ValueError:
        return None

    grid = args[3].strip()
    if not (grid.startswith("(") and grid.endswith(")")):
        return None
    net: List[List[Point3D]] = []
    for row in _split_args(grid[1:-1]):
        pts = [points.get(r) for r in _refs(row)]
        if not pts or any(p is None for p in pts):
            return None
        net.append(pts)
    if not net or any(len(r) != len(net[0]) for r in net):
        return None

    mult_u = [int(m) for m in _floats(args[8])]
    mult_v = [int(m) for m in _floats(args[9])]
    knots_u = _expand_knots(_floats(args[10]), mult_u)
    knots_v = _expand_knots(_floats(args[11]), mult_v)
    if len(knots_u) != len(net) + degree_u + 1:
        knots_u = None
    if len(knots_v) != len(net[0]) + degree_v + 1:
        knots_v = None
    return NURBSSurface(net, degree_u, degree_v,
                        knots_u=knots_u, knots_v=knots_v)


class _StepModel:
    """The subset of a STEP DATA section this kernel understands."""

    def __init__(self, records: Dict[int, Tuple[str, List[str]]]):
        self.points: Dict[int, Point3D] = {}
        self.vertex_point: Dict[int, int] = {}       # VERTEX_POINT -> point id
        self.edge_curve: Dict[int, Tuple[int, int]] = {}  # -> (v_start, v_end)
        self.oriented: Dict[int, Tuple[int, bool]] = {}   # -> (edge_curve, forward)
        self.edge_loop: Dict[int, List[int]] = {}    # -> oriented edge ids
        self.bound: Dict[int, Tuple[int, bool]] = {}      # -> (loop id, is_outer)
        self.face: Dict[int, Tuple[List[int], Optional[int]]] = {}  # -> (bounds, surf)
        self.surface: Dict[int, NURBSSurface] = {}
        self.shell: Dict[int, Tuple[str, List[int]]] = {}  # -> (kind, face ids)
        self.shell_name: Dict[int, str] = {}
        self.skipped_inner_bounds = 0

        for oid, (name, args) in records.items():
            if name == "CARTESIAN_POINT":
                coords = _floats(args[1]) if len(args) > 1 else []
                if len(coords) >= 3:
                    self.points[oid] = Point3D(*coords[:3])
            elif name == "VERTEX_POINT" and len(args) > 1:
                p = _ref(args[1])
                if p is not None:
                    self.vertex_point[oid] = p
            elif name == "EDGE_CURVE" and len(args) >= 3:
                a, b = _ref(args[1]), _ref(args[2])
                if a is not None and b is not None:
                    self.edge_curve[oid] = (a, b)
            elif name == "ORIENTED_EDGE" and len(args) >= 5:
                ec = _ref(args[3])
                if ec is not None:
                    self.oriented[oid] = (ec, ".T." in args[4].upper())
            elif name == "EDGE_LOOP" and len(args) >= 2:
                self.edge_loop[oid] = _refs(args[1])
            elif name in ("FACE_OUTER_BOUND", "FACE_BOUND") and len(args) >= 2:
                lp = _ref(args[1])
                if lp is not None:
                    self.bound[oid] = (lp, name == "FACE_OUTER_BOUND")
            elif name in ("ADVANCED_FACE", "FACE_SURFACE") and len(args) >= 3:
                self.face[oid] = (_refs(args[1]), _ref(args[2]))
            elif name == "B_SPLINE_SURFACE_WITH_KNOTS":
                surf = _parse_bspline_surface(args, self.points)
                if surf is not None:
                    self.surface[oid] = surf
            elif name in ("CLOSED_SHELL", "OPEN_SHELL") and len(args) >= 2:
                self.shell[oid] = (name, _refs(args[1]))

        # Names travel on the solid/model record that owns the shell.
        for _oid, (name, args) in records.items():
            if name == "MANIFOLD_SOLID_BREP" and len(args) >= 2:
                sh = _ref(args[1])
                if sh is not None:
                    self.shell_name[sh] = _unquote(args[0])
            elif name == "SHELL_BASED_SURFACE_MODEL" and len(args) >= 2:
                for sh in _refs(args[1]):
                    self.shell_name[sh] = _unquote(args[0])

    def face_vertex_ring(self, face_oid: int) -> Optional[List[int]]:
        """The outer bound of a face as an ordered list of VERTEX_POINT ids."""
        bound_ids, _surf = self.face[face_oid]
        outer = None
        for b in bound_ids:
            if b not in self.bound:
                continue
            loop_id, is_outer = self.bound[b]
            if is_outer or outer is None:
                if outer is not None:
                    self.skipped_inner_bounds += 1
                outer = loop_id
            else:
                self.skipped_inner_bounds += 1
        if outer is None or outer not in self.edge_loop:
            return None

        ring: List[int] = []
        for oe in self.edge_loop[outer]:
            if oe not in self.oriented:
                return None
            ec, forward = self.oriented[oe]
            if ec not in self.edge_curve:
                return None
            v_start, v_end = self.edge_curve[ec]
            ring.append(v_start if forward else v_end)
        return ring


def load(kernel: Kernel, filepath: str) -> List[Solid]:
    """
    Read a STEP file back into the kernel as real half-edge solids.

    One solid is created per CLOSED_SHELL / OPEN_SHELL. Vertices are merged by
    coordinate (to 1e-6) so a shell whose writer duplicated its CARTESIAN_POINTs
    still closes up. A file carrying no ADVANCED_FACE records degrades to
    :func:`load_points`.

    Returns the created solids.
    """
    with open(filepath, "r", encoding="utf-8") as fh:
        text = fh.read()

    model = _StepModel(_parse_records(text))
    if not model.face or not model.shell:
        return [load_points(kernel, filepath)]

    base = os.path.basename(filepath)
    solids: List[Solid] = []
    for shell_oid, (kind, face_oids) in model.shell.items():
        points: List[Point3D] = []
        index_of: Dict[tuple, int] = {}   # rounded coords -> index in points

        def _index(vertex_point_oid: int) -> Optional[int]:
            pid = model.vertex_point.get(vertex_point_oid)
            p = model.points.get(pid) if pid is not None else None
            if p is None:
                return None
            key = (round(p.x, 6), round(p.y, 6), round(p.z, 6))
            if key not in index_of:
                index_of[key] = len(points)
                points.append(p)
            return index_of[key]

        rings: List[List[int]] = []
        surfaces: List[Optional[NURBSSurface]] = []
        for f_oid in face_oids:
            if f_oid not in model.face:
                continue
            ring = model.face_vertex_ring(f_oid)
            if ring is None:
                continue
            idx = [_index(v) for v in ring]
            if any(i is None for i in idx):
                continue
            # Drop repeated neighbours left behind by the coordinate merge.
            dedup = [i for k, i in enumerate(idx) if i != idx[k - 1]]
            if len(dedup) < 3:
                continue
            rings.append(dedup)
            surfaces.append(model.surface.get(model.face[f_oid][1]))

        if not rings:
            continue
        name = model.shell_name.get(shell_oid) or f"imported:{base}"
        closed = kind == "CLOSED_SHELL"
        try:
            solid = build_solid_from_faces(kernel, points, rings, name,
                                           surfaces, closed=closed)
        except ValueError:
            # The shell claimed to be closed but isn't (or is wound
            # inconsistently); rebuild it as an open shell rather than lose it.
            solid = build_solid_from_faces(kernel, points, rings, name,
                                           surfaces, closed=False)
        solids.append(solid)

    if not solids:
        return [load_points(kernel, filepath)]
    return solids


def load_points(kernel: Kernel, filepath: str) -> Solid:
    """
    Parse a STEP file's CARTESIAN_POINTs into a new solid as a vertex cloud.

    The fallback for files whose topology this kernel cannot map. No edges or
    faces are created; the caller is told how many points were recovered so the
    limitation is explicit.
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

    solid.name = f"imported({count} pts):{os.path.basename(filepath)}"
    return solid
