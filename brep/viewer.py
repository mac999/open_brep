"""
Layer 4 - 3-D Viewer

Interactive 3-D visualization for B-Rep solids.

Backend priority
----------------
1. plotly  (pip install plotly)  -- browser-based, numpy-free, NURBS-sampled
2. vedo    (pip install vedo)    -- VTK interactive window
3. tkinter (stdlib)              -- perspective wireframe, no extra deps

Usage
-----
    from brep.viewer import show_solid
    show_solid(solid)                  # auto-detect backend, shaded+wire
    show_solid(solid, mode='wire')     # wireframe only
    show_solid(solid, mode='points')   # vertex cloud only
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .topology import Solid, Face
from .geometry import NURBSSurface, Point3D, TrimPlane, tessellate_surface_trim


# --------------------------------------------------------------------------- #
# Geometry helpers (numpy-free)
# --------------------------------------------------------------------------- #

def _face_pts(face: Face) -> List[Point3D]:
    """Return the ordered boundary vertices of face's outer loop."""
    return [he.vertex.point for he in face.outer.halfedges()
            if he.vertex and he.vertex.point]


def _surviving_refs(solid: Solid):
    """
    Return ``(vertex_oids, edge_oids)`` referenced by non-discarded faces.

    A trim leaves the discarded half in the in-memory topology (``discarded`` is
    an export/render flag), so the viewer must skip edges/vertices that belong
    only to removed faces — otherwise the trimmed-away wireframe still shows.
    """
    v_oids: set = set()
    e_oids: set = set()
    for f in solid.faces:
        if getattr(f, "discarded", False):
            continue
        for loop in f.loops:
            for he in loop.halfedges():
                if he.vertex is not None:
                    v_oids.add(he.vertex.oid)
                if he.edge is not None:
                    e_oids.add(he.edge.oid)
    return v_oids, e_oids


def _nurbs_face_mesh(face: Face, nu: int, nv: int):
    """
    Tessellate a NURBS ``face`` into ``(points, triangles)``.

    If the face carries a ``trim_plane`` (an interior/cap plane cut the flat
    boundary can't express), the mesh is clipped to the kept +side so the viewer
    shows the true trimmed surface; otherwise the full surface is sampled.
    """
    surf = face.surface
    plane = getattr(face, "trim_plane", None)
    if isinstance(plane, TrimPlane):
        return tessellate_surface_trim(surf, plane, nu, nv)
    grid = _sample_nurbs(surf, nu, nv)
    points: List[Point3D] = []
    tris: List[tuple] = []
    for row in grid:
        points.extend(row)
    stride = nv + 1
    for i in range(nu):
        for j in range(nv):
            o = i * stride + j
            tris.append((o, o + 1, o + stride + 1))
            tris.append((o, o + stride + 1, o + stride))
    return points, tris


def _fan_tris(n: int) -> List[Tuple[int, int, int]]:
    """Fan-triangulate an n-vertex polygon: (0,i,i+1) for i in 1..n-2."""
    return [(0, i, i + 1) for i in range(1, n - 1)]


def _eval_bezier(pts: List[Point3D], t: float) -> Point3D:
    """De Casteljau evaluation — pure Point3D, no numpy."""
    pts = list(pts)
    while len(pts) > 1:
        pts = [pts[i] * (1.0 - t) + pts[i + 1] * t for i in range(len(pts) - 1)]
    return pts[0]


def _sample_nurbs(surf: NURBSSurface, nu: int = 14, nv: int = 14
                  ) -> List[List[Point3D]]:
    """
    Sample the NURBS surface at a (nu+1)×(nv+1) grid using pure Point3D
    De Casteljau arithmetic — does not call surf.evaluate() so numpy is
    not required.

    Returns rows of Point3D; row[i][j] = S(i/nu, j/nv).
    """
    rows: List[List[Point3D]] = []
    for i in range(nu + 1):
        u = i / nu
        u_col = [_eval_bezier(net_row, u) for net_row in surf.control_net]
        rows.append([_eval_bezier(u_col, j / nv) for j in range(nv + 1)])
    return rows


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def show_solid(solid: Solid, mode: str = "solid", title: str = "") -> None:
    """
    Open an interactive 3-D viewer for *solid*.

    Parameters
    ----------
    mode : 'solid'   shaded faces + edge overlay  (default)
           'wire'    edge wireframe only
           'points'  vertex cloud only
    title : window / page title; defaults to "Solid #N 'name'"
    """
    title = title or f"Solid #{solid.oid}  '{solid.name}'"

    for backend in (_show_plotly, _show_vedo, _show_tkinter):
        try:
            backend(solid, mode, title)
            return
        except ImportError:
            continue

    raise RuntimeError(
        "No 3-D viewer available.\n"
        "  Install plotly:  pip install plotly   (recommended)\n"
        "  Install vedo:    pip install vedo"
    )


# --------------------------------------------------------------------------- #
# Backend 1 – plotly  (browser-based, numpy-free)
# --------------------------------------------------------------------------- #

def _show_plotly(solid: Solid, mode: str, title: str) -> None:
    import plotly.graph_objects as go  # ImportError if not installed

    traces = []
    surv_v, surv_e = _surviving_refs(solid)

    # ── Edges ────────────────────────────────────────────────────────────── #
    ex: List[Optional[float]] = []
    ey: List[Optional[float]] = []
    ez: List[Optional[float]] = []
    for edge in solid.edges:
        if edge.oid not in surv_e:
            continue
        he = edge.he1
        if he is None or he.vertex is None:
            continue
        end = he.end_vertex
        if end is None:
            continue
        a, b = he.vertex.point, end.point
        ex += [a.x, b.x, None]
        ey += [a.y, b.y, None]
        ez += [a.z, b.z, None]

    if ex:
        traces.append(go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode='lines',
            line=dict(color='#333333', width=2),
            name='edges',
            hoverinfo='skip',
        ))

    # ── Shaded faces ─────────────────────────────────────────────────────── #
    if mode in ('solid', 'shaded', 'surf'):
        # Flat planar faces
        px: List[float] = []
        py: List[float] = []
        pz: List[float] = []
        pi: List[int] = []
        pj: List[int] = []
        pk: List[int] = []

        # NURBS sampled surfaces
        nx_: List[float] = []
        ny_: List[float] = []
        nz_: List[float] = []
        ni: List[int] = []
        nj: List[int] = []
        nk: List[int] = []

        for face in solid.faces:
            if getattr(face, 'discarded', False):
                continue

            surf = getattr(face, 'surface', None)
            if isinstance(surf, NURBSSurface):
                # ── NURBS face: sample (clipped to keep side if trimmed) ── #
                pts_m, tris_m = _nurbs_face_mesh(face, 14, 14)
                g_off = len(nx_)
                for p in pts_m:
                    nx_.append(p.x); ny_.append(p.y); nz_.append(p.z)
                for (ta, tb, tc) in tris_m:
                    ni.append(g_off + ta)
                    nj.append(g_off + tb)
                    nk.append(g_off + tc)
            else:
                # ── Planar face: fan triangulation ── #
                pts = _face_pts(face)
                if len(pts) < 3:
                    continue
                off = len(px)
                for p in pts:
                    px.append(p.x); py.append(p.y); pz.append(p.z)
                for (ti, tj, tk) in _fan_tris(len(pts)):
                    pi.append(off + ti)
                    pj.append(off + tj)
                    pk.append(off + tk)

        if px:
            traces.append(go.Mesh3d(
                x=px, y=py, z=pz, i=pi, j=pj, k=pk,
                color='#b0c8e8', opacity=0.55,
                flatshading=False, name='faces',
                lighting=dict(diffuse=0.8, specular=0.3, ambient=0.3),
                hoverinfo='skip',
            ))
        if nx_:
            traces.append(go.Mesh3d(
                x=nx_, y=ny_, z=nz_, i=ni, j=nj, k=nk,
                color='#e8c890', opacity=0.65,
                flatshading=False, name='NURBS surface',
                lighting=dict(diffuse=0.9, specular=0.5, ambient=0.2),
            ))

    # ── Vertices ─────────────────────────────────────────────────────────── #
    if mode == 'points':
        vpts = [v.point for v in solid.vertices if v.point and v.oid in surv_v]
        if vpts:
            traces.append(go.Scatter3d(
                x=[p.x for p in vpts],
                y=[p.y for p in vpts],
                z=[p.z for p in vpts],
                mode='markers',
                marker=dict(size=6, color='tomato'),
                name='vertices',
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            bgcolor='#f4f4f4',
        ),
        legend=dict(yanchor='top', y=0.97, xanchor='left', x=0.01),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.show()  # opens default browser


# --------------------------------------------------------------------------- #
# Backend 2 – vedo  (VTK interactive window)
# --------------------------------------------------------------------------- #

def _show_vedo(solid: Solid, mode: str, title: str) -> None:
    import vedo  # ImportError if not installed / if numpy broken

    objects = []
    surv_v, surv_e = _surviving_refs(solid)

    # Edges
    starts: List[List[float]] = []
    ends: List[List[float]] = []
    for edge in solid.edges:
        if edge.oid not in surv_e:
            continue
        he = edge.he1
        if he is None or he.vertex is None:
            continue
        end = he.end_vertex
        if end is None:
            continue
        a, b = he.vertex.point, end.point
        starts.append([a.x, a.y, a.z])
        ends.append([b.x, b.y, b.z])
    if starts:
        objects.append(vedo.Lines(starts, ends, c='gray4', lw=1))

    if mode in ('solid', 'shaded', 'surf'):
        all_pts: List[List[float]] = []
        all_cells: List[List[int]] = []

        for face in solid.faces:
            if getattr(face, 'discarded', False):
                continue
            surf = getattr(face, 'surface', None)
            if isinstance(surf, NURBSSurface):
                pts_m, tris_m = _nurbs_face_mesh(face, 12, 12)
                g_off = len(all_pts)
                for p in pts_m:
                    all_pts.append([p.x, p.y, p.z])
                for (ta, tb, tc) in tris_m:
                    all_cells.append([g_off + ta, g_off + tb, g_off + tc])
                continue

            pts = _face_pts(face)
            if len(pts) < 3:
                continue
            off = len(all_pts)
            for p in pts:
                all_pts.append([p.x, p.y, p.z])
            for (ti, tj, tk) in _fan_tris(len(pts)):
                all_cells.append([off + ti, off + tj, off + tk])

        if all_pts:
            mesh = vedo.Mesh([all_pts, all_cells])
            mesh.color('lightsteelblue').alpha(0.55).lighting('plastic')
            objects.append(mesh)

    elif mode == 'points':
        vpts = [v.point for v in solid.vertices if v.point and v.oid in surv_v]
        if vpts:
            objects.append(
                vedo.Points([[p.x, p.y, p.z] for p in vpts], r=8).color('tomato')
            )

    vedo.show(*objects, title=title, axes=1, bg='white', bg2='#fffff0')


# --------------------------------------------------------------------------- #
# Backend 3 – tkinter  (stdlib, always available)
# --------------------------------------------------------------------------- #

def _show_tkinter(solid: Solid, mode: str, title: str) -> None:
    """
    Perspective wireframe viewer using only Python's standard library tkinter.

    Controls
    --------
    Left-drag         rotate
    Shift + left-drag pan
    Mouse wheel       zoom
    """
    import tkinter as tk

    # ── collect geometry ── #
    surv_v, surv_e = _surviving_refs(solid)
    edges_3d: List[Tuple[Tuple, Tuple]] = []
    for edge in solid.edges:
        if edge.oid not in surv_e:
            continue
        he = edge.he1
        if he is None or he.vertex is None:
            continue
        end = he.end_vertex
        if end is None:
            continue
        a, b = he.vertex.point, end.point
        edges_3d.append(((a.x, a.y, a.z), (b.x, b.y, b.z)))

    face_polys: List[List[Tuple[float, float, float]]] = []
    if mode in ('solid', 'shaded'):
        for face in solid.faces:
            if getattr(face, 'discarded', False):
                continue
            surf = getattr(face, 'surface', None)
            if isinstance(surf, NURBSSurface):
                if isinstance(getattr(face, 'trim_plane', None), TrimPlane):
                    # Trimmed: draw the kept-side triangles' outlines.
                    pts_m, tris_m = _nurbs_face_mesh(face, 10, 10)
                    for (ta, tb, tc) in tris_m:
                        face_polys.append([(pts_m[k].x, pts_m[k].y, pts_m[k].z)
                                           for k in (ta, tb, tc)])
                    continue
                # Untrimmed: coarse 6×6 grid of sample lines.
                grid = _sample_nurbs(surf, 6, 6)
                for row in grid:
                    face_polys.append([(p.x, p.y, p.z) for p in row])
                for col_j in range(7):
                    face_polys.append([(grid[i][col_j].x, grid[i][col_j].y,
                                        grid[i][col_j].z) for i in range(7)])
                continue
            pts = _face_pts(face)
            if len(pts) >= 3:
                face_polys.append([(p.x, p.y, p.z) for p in pts])

    all_pts = [v.point for v in solid.vertices if v.point and v.oid in surv_v]
    if not all_pts:  # e.g. a fully-metadata trim leaves nothing flagged
        all_pts = [v.point for v in solid.vertices if v.point]
    if not all_pts:
        return

    cx = sum(p.x for p in all_pts) / len(all_pts)
    cy = sum(p.y for p in all_pts) / len(all_pts)
    cz = sum(p.z for p in all_pts) / len(all_pts)
    rng = max(
        max(p.x for p in all_pts) - min(p.x for p in all_pts),
        max(p.y for p in all_pts) - min(p.y for p in all_pts),
        max(p.z for p in all_pts) - min(p.z for p in all_pts), 1.0
    )

    W, H = 760, 620
    root = tk.Tk()
    root.title(title)
    canvas = tk.Canvas(root, width=W, height=H, bg='#ececec')
    canvas.pack(fill='both', expand=True)
    tk.Label(
        root,
        text="Left-drag: rotate   Shift+drag: pan   Scroll: zoom",
        fg='#555', font=('Consolas', 9)
    ).pack(side='bottom', pady=3)

    st = {'ax': 0.45, 'ay': -0.65, 'zoom': 1.0, 'ox': 0.0, 'oy': 0.0}

    def project(x: float, y: float, z: float) -> Tuple[float, float]:
        ay, ax = st['ay'], st['ax']
        # Rotation about Y
        rx = x * math.cos(ay) - z * math.sin(ay)
        ry = y
        rz = x * math.sin(ay) + z * math.cos(ay)
        # Rotation about X
        ry2 = ry * math.cos(ax) - rz * math.sin(ax)
        rz2 = ry * math.sin(ax) + rz * math.cos(ax)
        rx2 = rx
        # Perspective divide
        dist = rng * 5.0
        scale = (dist / max(dist + rz2, 0.1)) * (min(W, H) * 0.65) / rng * st['zoom']
        return (W / 2 + st['ox'] + rx2 * scale,
                H / 2 + st['oy'] - ry2 * scale)

    def draw() -> None:
        canvas.delete('all')

        # Face fills / NURBS grid lines
        for poly in face_polys:
            coords: List[float] = []
            for (x, y, z) in poly:
                sx, sy = project(x - cx, y - cy, z - cz)
                coords += [sx, sy]
            if len(coords) >= 6:
                if mode in ('solid', 'shaded') and len(poly) >= 3:
                    canvas.create_polygon(
                        coords, fill='#b0c8e0', outline='#8090a0', width=0, stipple='gray50'
                    )
                else:
                    canvas.create_line(coords, fill='#5080a0', width=1)

        # Edges
        for (a, b) in edges_3d:
            sx1, sy1 = project(a[0] - cx, a[1] - cy, a[2] - cz)
            sx2, sy2 = project(b[0] - cx, b[1] - cy, b[2] - cz)
            canvas.create_line(sx1, sy1, sx2, sy2, fill='#222', width=1)

        # Vertices
        for pt in all_pts:
            sx, sy = project(pt.x - cx, pt.y - cy, pt.z - cz)
            canvas.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill='#cc3333', outline='')

        # Axis indicators (dashed lines at centre)
        for (dx, dy, dz, col, lbl) in [
            (1, 0, 0, '#cc0000', 'X'),
            (0, 1, 0, '#008800', 'Y'),
            (0, 0, 1, '#0000cc', 'Z'),
        ]:
            r2 = rng * 0.45
            sx0, sy0 = project(-dx * r2, -dy * r2, -dz * r2)
            sx1, sy1 = project( dx * r2,  dy * r2,  dz * r2)
            canvas.create_line(sx0, sy0, sx1, sy1, fill=col, width=1, dash=(5, 3))
            canvas.create_text(sx1 + 6, sy1, text=lbl, fill=col,
                               font=('Consolas', 9, 'bold'))

        # Info
        canvas.create_text(8, 8, anchor='nw',
                           text=f"V={solid.num_vertices}  E={solid.num_edges}  F={solid.num_faces}",
                           fill='#444', font=('Consolas', 10))

    drag: dict = {}

    def on_press(e: tk.Event) -> None:
        drag.update(x=e.x, y=e.y, shift=bool(e.state & 0x0001))

    def on_drag(e: tk.Event) -> None:
        dx, dy = e.x - drag['x'], e.y - drag['y']
        drag.update(x=e.x, y=e.y)
        if drag.get('shift'):
            st['ox'] += dx; st['oy'] += dy
        else:
            st['ay'] += dx * 0.008
            st['ax'] -= dy * 0.008
        draw()

    def on_scroll(e: tk.Event) -> None:
        factor = 1.1 if (getattr(e, 'delta', 0) > 0 or e.num == 4) else 0.9
        st['zoom'] *= factor
        draw()

    canvas.bind('<ButtonPress-1>', on_press)
    canvas.bind('<B1-Motion>', on_drag)
    canvas.bind('<MouseWheel>', on_scroll)
    canvas.bind('<Button-4>', on_scroll)   # Linux scroll up
    canvas.bind('<Button-5>', on_scroll)   # Linux scroll down

    draw()
    root.mainloop()
